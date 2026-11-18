"""Frame sources for the media scanner.

The scanner needs the frames of a video as PIL images.  There are two
ways to get them, and both are offered:

``ffmpeg``
    The external binary (as before): ffmpeg decodes the file and hands
    the frames over - as raw RGB (``frames_format="raw"``, the default,
    no PNG round-trip) or as PNGs.  It uses the ffmpeg **the user has
    installed**, so the decoded pixels are exactly the ones every other
    tool on that machine produces.

``av``
    PyAV (the optional ``av`` package) links libav **into our process**:
    the frames never travel through a pipe, they land in our own memory
    as numpy arrays, and libav's decoder threads overlap with the model
    inference.  It is faster (measured ~2.7x on the decode) but it
    brings **its own** FFmpeg build, which decodes a little differently
    from the system ffmpeg - so the frames are *not* bit-identical and a
    borderline frame can land on another side of the threshold.

Both return the same thing - ``(index, timecode, PIL image)`` - so the
scanner above them does not care which one ran.
"""

import io
import os
import queue
import subprocess
import threading
import time

from PIL import Image

__all__ = [
    "frames",
    "backend_available",
    "last_hwaccel",
    "BACKENDS",
    "HWACCEL_CANDIDATES",
]


BACKENDS = ("auto", "av", "ffmpeg")
HWACCEL_CANDIDATES = ("cuda", "qsv", "d3d11va", "vaapi")
FFMPEG_BIN = os.environ.get("NSFW_FFMPEG", "ffmpeg")


class DecoderUnavailable(RuntimeError):
    """This frame source cannot handle the file - try the next one."""


# Which decoder the last run actually used ("cpu" for the software one).
_last_hwaccel = {"value": None}


def last_hwaccel():
    """The decoder the last frame source actually used (None = unknown)."""
    return _last_hwaccel["value"]


def backend_available(backend):
    """Is the requested backend usable here?"""
    if backend in ("auto", "av"):
        try:
            import av  # noqa: F401
            return True
        except ImportError:
            return backend == "auto" and False
    return True


def _hwaccel_candidates(mode):
    """The hardware decoders to try, in order (None = the software one)."""
    if not mode or mode == "none":
        return [None]
    if mode == "auto":
        return list(HWACCEL_CANDIDATES) + [None]
    return [mode, None]


def _read_exactly(stream, count):
    """Reads exactly `count` bytes (a pipe read may return less)."""
    chunks = []
    remaining = count
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _png_end(buffer, start):
    """Byte offset just after the PNG at `start`, or -1 when incomplete."""
    pos = start + 8
    while pos + 8 <= len(buffer):
        length = int.from_bytes(buffer[pos:pos + 4], "big")
        ctype = buffer[pos + 4:pos + 8]
        if ctype == b"IEND":
            return pos + 12 + length
        pos += 12 + length
    return -1


def _showinfo_times(stderr_stream, sink):
    """Reads ffmpeg's showinfo output and pushes the frame timestamps."""
    for raw in stderr_stream:
        line = raw.decode("utf-8", "replace")
        if "pts_time:" not in line:
            continue
        chunk = line.split("pts_time:", 1)[1].split()[0]
        try:
            sink.put(float(chunk))
        except (ValueError, IndexError):
            continue


# ---------------------------------------------------------------------------
# PyAV (in-process)
# ---------------------------------------------------------------------------

def _frames_av(path, fps, every_frame, max_frames, hwaccel):
    """The frames via PyAV - in-process, no pipe, numpy arrays."""
    import av

    options = {}
    if hwaccel and hwaccel not in ("none", "auto"):
        # "auto" means "try the list" - that is the ffmpeg backend's job;
        # libav only understands a concrete decoder name.
        options["hwaccel"] = hwaccel
    _last_hwaccel["value"] = (hwaccel if hwaccel not in ("none", "auto")
                              else "cpu")
    try:
        container = av.open(path, options=options or None)
    except Exception as exc:
        raise DecoderUnavailable("PyAV could not open the file: %s" % exc)
    try:
        stream = container.streams.video[0]
        try:
            stream.thread_type = "AUTO"
        except Exception:
            pass
        step = (1.0 / fps) if (fps and fps > 0 and not every_frame) else 0.0
        index = 0
        target = 0.0
        half = step / 2.0
        pending = None            # the last frame before the current slot
        for frame in container.decode(stream):
            try:
                timecode = float(frame.pts * stream.time_base)
            except (TypeError, ValueError):
                timecode = float(index * step)
            array = frame.to_ndarray(format="rgb24")
            if every_frame or not step:
                yield index, timecode, Image.fromarray(array)
                index += 1
                if index >= max_frames:
                    return
                continue
            # The same rule as ffmpeg's fps filter: the output slot N
            # takes the *last* input frame before N + half a slot.  (The
            # filter rounds the frame's time to the nearest slot and keeps
            # the last frame of every slot.)  Getting this exactly right
            # is what makes the two backends scan the same frames.
            while timecode >= target + half:
                if pending is not None:
                    yield index, pending[0], Image.fromarray(pending[1])
                    index += 1
                    if index >= max_frames:
                        return
                target += step
                pending = None
            pending = (timecode, array)
    finally:
        try:
            container.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# The external ffmpeg
# ---------------------------------------------------------------------------

def _frames_ffmpeg(path, fps, every_frame, max_frames, ffmpeg_bin, hwaccel,
                    frames_format, frame_size, stream_index):
    """The frames via the external ffmpeg (raw RGB or PNG over a pipe)."""
    cmd = [ffmpeg_bin or FFMPEG_BIN, "-v", "info"]
    if hwaccel:
        cmd += ["-hwaccel", hwaccel]
    cmd += ["-i", path]
    filters = []
    if not every_frame and fps and fps > 0:
        filters.append("fps=%s" % fps)
    filters.append("showinfo")
    cmd += ["-vf", ",".join(filters)]
    if stream_index is not None:
        cmd += ["-map", "0:%d" % int(stream_index)]
    use_raw = (frames_format == "raw" and frame_size
               and frame_size[0] > 0 and frame_size[1] > 0)
    if use_raw:
        cmd += ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    else:
        cmd += ["-f", "image2pipe", "-vcodec", "png", "-"]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, bufsize=1024 * 1024)
    times = queue.Queue()
    reader = threading.Thread(target=_showinfo_times, args=(proc.stderr, times),
                               daemon=True)
    reader.start()

    buffer = b""
    index = 0
    fallback_step = (1.0 / fps) if (fps and fps > 0 and not every_frame) else 0.0
    try:
        while index < max_frames:
            if use_raw:
                width, height = frame_size
                needed = width * height * 3
                data = _read_exactly(proc.stdout, needed)
                if not data:
                    break                      # the file ended - normal
                if len(data) < needed:
                    raise DecoderUnavailable(
                        "the raw frame stream ended early (%d of %d bytes)"
                        % (len(data), needed))
                try:
                    image = Image.frombuffer("RGB", (width, height), data,
                                             "raw", "RGB", 0, 1)
                except Exception:
                    image = None
                buffer = b""
            else:
                chunk = proc.stdout.read(262144)
                if not chunk:
                    break
                buffer += chunk
                while True:
                    start = buffer.find(b"\x89PNG\r\n\x1a\n")
                    if start < 0:
                        if len(buffer) > 8:
                            buffer = buffer[-8:]
                        break
                    if start > 0:
                        buffer = buffer[start:]
                    end = _png_end(buffer, 0)
                    if end < 0:
                        break
                    png, buffer = buffer[:end], buffer[end:]
                    try:
                        image = Image.open(io.BytesIO(png)).convert("RGB")
                    except Exception:
                        image = None
                    timecode = None
                    try:
                        timecode = times.get_nowait()
                    except queue.Empty:
                        timecode = None
                    if timecode is None:
                        timecode = index * fallback_step
                    yield index, timecode, image
                    index += 1
                    if index >= max_frames:
                        break
                continue
            timecode = None
            try:
                timecode = times.get_nowait()
            except queue.Empty:
                timecode = None
            if timecode is None:
                timecode = index * fallback_step
            yield index, timecode, image
            index += 1
            if index >= max_frames:
                break
    finally:
        try:
            proc.stdout.close()
        except OSError:
            pass
        try:
            proc.terminate()
            proc.wait(timeout=10)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# The public entry point
# ---------------------------------------------------------------------------

def frames(path, fps=1.0, every_frame=False, max_frames=2000, ffmpeg_bin=None,
           hwaccel=None, frames_format="raw", backend="auto", frame_size=None,
           stream_index=None):
    """Yields (index, timecode, PIL image) for the sampled frames.

    ``backend``: "auto" (PyAV when it is installed, else the external
    ffmpeg), "av" (PyAV only) or "ffmpeg" (the binary only).

    ``hwaccel`` picks the decoder for the ffmpeg backend: "none",
    "auto" (cuda -> qsv -> d3d11va -> vaapi -> software) or one of
    those names.  The hardware decoders are **not** bit-identical to the
    software one (a few levels of rounding), so a borderline frame can
    flip.

    ``frames_format``: "raw" (raw RGB over the pipe - no PNG encode, no
    PNG decode) or "png" (the PNG pipe).  The raw path needs the frame
    size; without it the PNG path is used.
    """
    if backend not in BACKENDS:
        backend = "auto"
    if backend in ("auto", "av"):
        try:
            import av  # noqa: F401
        except ImportError:
            if backend == "av":
                raise DecoderUnavailable(
                    "the 'av' backend needs PyAV: pip install "
                    "\"av>=13,<17\"")
        else:
            try:
                for item in _frames_av(path, fps, every_frame, max_frames,
                                       hwaccel):
                    yield item
                return
            except DecoderUnavailable:
                if backend == "av":
                    raise
            except Exception:
                if backend == "av":
                    raise
                # fall through to the external ffmpeg
    last_error = None
    for candidate in _hwaccel_candidates(hwaccel):
        produced = 0
        _last_hwaccel["value"] = (candidate or "cpu")
        try:
            for item in _frames_ffmpeg(path, fps, every_frame, max_frames,
                                       ffmpeg_bin, candidate, frames_format,
                                       frame_size, stream_index):
                produced += 1
                yield item
            return
        except DecoderUnavailable as exc:
            # The raw stream broke - redo the pass with the PNG pipe.
            last_error = exc
            if frames_format == "png":
                raise
            try:
                for item in _frames_ffmpeg(path, fps, every_frame, max_frames,
                                           ffmpeg_bin, candidate, "png",
                                           None, stream_index):
                    produced += 1
                    yield item
                return
            except Exception as inner:
                raise inner
        except Exception as exc:
            if produced == 0 and candidate is not None:
                last_error = exc
                continue
            raise
    if last_error is not None:
        raise last_error
