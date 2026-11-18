"""Media scanner: video (key frames) and audio, with subtitle analysis.

ffmpeg + ffprobe are **required** by this module (external binaries) - the
image and website scanners never need them.  When they are missing the
scanner fails loudly with :class:`MediaDependencyError` instead of quietly
returning a wrong answer.

Video
    Frames are sampled (``fps=1`` by default, one frame per second) and
    rated with the same models the image scanner uses.  A denser pass
    (``every_frame=True``) is available for hunting subliminal content:
    every frame above ``min_report_score`` is reported with its timestamp.

Subtitles
    Embedded tracks and sidecar files are parsed (SRT / VTT / ASS) and
    scanned for sensuality terms.  A hit is a *problem area* with the
    subtitle line number and its timecode.

The result mirrors the other scanners and adds the "where":
``problems`` lists every frame / subtitle line that stood out.
"""

import io
import json
import os
import queue
import shutil
import subprocess
import threading
import time

import hashlib

from PIL import Image, ImageFilter

from . import decoder as _decoder
from . import subtitles as _subs

__all__ = [
    "MediaDependencyError",
    "VIDEO_EXTENSIONS",
    "AUDIO_EXTENSIONS",
    "detect_kind",
    "probe",
    "require_ffmpeg",
    "scan_audio",
    "scan_media",
    "scan_video",
]


class MediaDependencyError(RuntimeError):
    """ffmpeg/ffprobe missing or unusable - the media scanner cannot run."""


FFMPEG_BIN = os.environ.get("NSFW_FFMPEG", "ffmpeg")
FFPROBE_BIN = os.environ.get("NSFW_FFPROBE", "ffprobe")

# Hardware-accelerated decoding.  "auto" tries the candidates in order and
# falls back to the software decoder when the hardware refuses the file
# (an old GPU, an exotic codec) - the scan then runs, only slower.
HWACCEL_CANDIDATES = ("cuda", "qsv", "d3d11va", "vaapi")
HWACCEL_MODES = ("none", "auto") + HWACCEL_CANDIDATES


class _HwaccelUnavailable(RuntimeError):
    """The hardware decoder refused the file - fall back to the CPU."""


class _RawFramesUnavailable(RuntimeError):
    """The raw frame stream broke - fall back to the PNG path."""

VIDEO_EXTENSIONS = (".mp4", ".m4v", ".mkv", ".webm", ".mov", ".avi",
                    ".mpg", ".mpeg", ".wmv", ".flv", ".ts", ".m2ts",
                    ".ogv", ".3gp", ".vob")
AUDIO_EXTENSIONS = (".mp3", ".flac", ".wav", ".m4a", ".aac", ".ogg",
                    ".oga", ".opus", ".wma", ".aiff", ".aif", ".amr")

# extension -> (media_type, mime, container)
_EXTENSION_MAP = {}
for _ext in VIDEO_EXTENSIONS:
    _EXTENSION_MAP[_ext] = ("video", "video/mp4", None)
_EXTENSION_MAP.update({
    ".mkv": ("video", "video/x-matroska", "matroska"),
    ".webm": ("video", "video/webm", "webm"),
    ".mov": ("video", "video/quicktime", "quicktime"),
    ".avi": ("video", "video/x-msvideo", "avi"),
    ".mpg": ("video", "video/mpeg", "mpeg"),
    ".mpeg": ("video", "video/mpeg", "mpeg"),
    ".wmv": ("video", "video/x-ms-wmv", "asf"),
    ".flv": ("video", "video/x-flv", "flv"),
    ".ts": ("video", "video/mp2t", "mpegts"),
    ".m2ts": ("video", "video/mp2t", "mpegts"),
    ".3gp": ("video", "video/3gpp", "3gpp"),
    ".ogv": ("video", "video/ogg", "ogg"),
    ".vob": ("video", "video/dvd", "vob"),
    ".mp3": ("audio", "audio/mpeg", "mp3"),
    ".flac": ("audio", "audio/flac", "flac"),
    ".wav": ("audio", "audio/wav", "wav"),
    ".m4a": ("audio", "audio/mp4", "mp4"),
    ".aac": ("audio", "audio/aac", "aac"),
    ".ogg": ("audio", "audio/ogg", "ogg"),
    ".oga": ("audio", "audio/ogg", "ogg"),
    ".opus": ("audio", "audio/opus", "opus"),
    ".wma": ("audio", "audio/x-ms-wma", "asf"),
    ".aiff": ("audio", "audio/aiff", "aiff"),
    ".aif": ("audio", "audio/aiff", "aiff"),
    ".amr": ("audio", "audio/amr", "amr"),
})

# header magic -> (media_type, mime, container)
_MAGIC_MAP = (
    (b"\x1a\x45\xdf\xa3", "video", "video/x-matroska", "matroska"),
    (b"RIFF", None, None, None),        # refined below (AVI / WAVE / WEBP)
    (b"FLV\x01", "video", "video/x-flv", "flv"),
    (b"\x30\x26\xb2\x75", "video", "video/x-ms-asf", "asf"),
    (b"fLaC", "audio", "audio/flac", "flac"),
    (b"OggS", "audio", "audio/ogg", "ogg"),
    (b"ID3", "audio", "audio/mpeg", "mp3"),
    (b"\xff\xfb", "audio", "audio/mpeg", "mp3"),
    (b"\xff\xf3", "audio", "audio/mpeg", "mp3"),
    (b"\xff\xf2", "audio", "audio/mpeg", "mp3"),
    (b"\xff\xf1", "audio", "audio/aac", "aac"),
    (b"\xff\xf9", "audio", "audio/aac", "aac"),
    (b"\x47", "video", "video/mp2t", "mpegts"),     # refined below
)

_IMAGE_MAGIC = (b"\xff\xd8\xff", b"\x89PNG", b"GIF8", b"BM",
                b"II*\x00", b"MM\x00*")


def require_ffmpeg():
    """Makes sure ffmpeg + ffprobe are usable; raises MediaDependencyError.

    Called by every media scan - never at import time, so the image and
    website scanners stay usable on a box without ffmpeg.
    """
    found = {}
    for name, binary in (("ffmpeg", FFMPEG_BIN), ("ffprobe", FFPROBE_BIN)):
        path = shutil.which(binary) or (
            binary if os.path.isfile(binary) else None)
        if not path:
            raise MediaDependencyError(
                "%s not found (looked for %r on PATH). The media scanner needs "
                "ffmpeg + ffprobe - install ffmpeg (Windows: 'winget install "
                "Gyan.FFmpeg', macOS: 'brew install ffmpeg', Debian: 'apt "
                "install ffmpeg') or point NSFW_FFMPEG / NSFW_FFPROBE at the "
                "binaries. The image and website scanners do not need them."
                % (name, binary))
        found[name] = path
    return found


def _read_head(path, size=4096):
    try:
        with open(path, "rb") as handle:
            return handle.read(size)
    except OSError:
        return b""


def _sniff(head):
    """Sniffs the media type from the file header.  None when unknown."""
    if not head:
        return None
    if head.startswith(b"RIFF"):
        if head[8:12] == b"WEBP":
            return ("image", "image/webp", "webp")
        if head[8:12] == b"AVI ":
            return ("video", "video/x-msvideo", "avi")
        if head[8:12] == b"WAVE":
            return ("audio", "audio/wav", "wav")
        return None
    if head[4:8] == b"ftyp":
        brand = head[8:12]
        if brand in (b"M4A ", b"M4B "):
            return ("audio", "audio/mp4", "mp4")
        if brand in (b"qt  "):
            return ("video", "video/quicktime", "quicktime")
        return ("video", "video/mp4", "mp4")
    if head.startswith(b"\x1a\x45\xdf\xa3"):
        # Matroska and WebM share the EBML header; the doctype decides,
        # but the extension is a good enough hint for the container name.
        if b"webm" in head[:64].lower():
            return ("video", "video/webm", "webm")
        return ("video", "video/x-matroska", "matroska")
    if head[:2] == b"\x47\x00" or (head[0] == 0x47 and head[1] == 0x00):
        return ("video", "video/mp2t", "mpegts")
    if head.startswith(_IMAGE_MAGIC):
        if head.startswith(b"\xff\xd8\xff"):
            return ("image", "image/jpeg", "jpeg")
        if head.startswith(b"\x89PNG"):
            return ("image", "image/png", "png")
        if head.startswith(b"GIF8"):
            return ("image", "image/gif", "gif")
        if head.startswith(b"BM"):
            return ("image", "image/bmp", "bmp")
        return ("image", "image/tiff", "tiff")
    for magic, media_type, mime, container in _MAGIC_MAP:
        if magic == b"RIFF" or magic == b"\x47":
            continue
        if head.startswith(magic):
            return (media_type, mime, container)
    return None


def detect_kind(target):
    """Detects what a file is: header first, extension as the fallback.

    Returns::

        {"kind": "video"|"audio"|"image"|"subtitle"|"unknown",
         "media_type": ..., "mime": ..., "mime_sniffed": ...,
         "mime_claimed": ..., "ext": ..., "container": ...,
         "type_mismatch": bool}

    The **header wins** for routing; a file that lies about its extension
    is still scanned as what it is, and the disagreement is flagged.
    """
    path = str(target or "")
    ext = os.path.splitext(path)[1].lower()
    head = _read_head(path)
    sniffed = _sniff(head)
    claimed = _EXTENSION_MAP.get(ext)
    if claimed is None:
        from .scanner import IMAGE_EXTENSIONS
        if ext in IMAGE_EXTENSIONS:
            claimed = ("image", "image/%s" % ext.lstrip("."), None)

    claimed_type = claimed[0] if claimed else None
    claimed_mime = claimed[1] if claimed else None
    if ext in _subs.SUBTITLE_EXTENSIONS:
        claimed_type, claimed_mime = "subtitle", "text/vtt"

    if sniffed:
        media_type, mime, container = sniffed
    elif claimed:
        media_type, mime, container = claimed_type, claimed_mime, None
    else:
        media_type, mime, container = "unknown", None, None

    mismatch = bool(sniffed and claimed_type and media_type != claimed_type)
    return {
        "kind": media_type,
        "media_type": media_type,
        "mime": mime,
        "mime_sniffed": sniffed[1] if sniffed else None,
        "mime_claimed": claimed_mime,
        "ext": ext or None,
        "container": container or (claimed[2] if claimed else None),
        "type_mismatch": mismatch,
    }


def probe(path, ffprobe_bin=None):
    """Runs ffprobe and returns the parsed format/streams info."""
    binaries = require_ffmpeg()
    cmd = [ffprobe_bin or binaries["ffprobe"], "-v", "quiet",
           "-print_format", "json", "-show_format", "-show_streams", path]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, timeout=120)
    except (OSError, subprocess.SubprocessError) as exc:
        raise MediaDependencyError("ffprobe failed on %r: %s" % (path, exc))
    if proc.returncode != 0:
        raise MediaDependencyError(
            "ffprobe could not read %r: %s"
            % (path, proc.stderr.decode("utf-8", "replace").strip()[:200]))
    try:
        return json.loads(proc.stdout.decode("utf-8", "replace"))
    except ValueError as exc:
        raise MediaDependencyError("ffprobe returned invalid JSON: %s" % exc)


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


def _hwaccel_candidates(mode):
    """The hardware decoders to try, in order ([] = the software decoder)."""
    if not mode or mode == "none":
        return [None]
    if mode == "auto":
        return list(HWACCEL_CANDIDATES) + [None]
    if mode not in HWACCEL_MODES:
        return [None]
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


def iter_frames(path, fps=1.0, every_frame=False, max_frames=2000,
                ffmpeg_bin=None, hwaccel=None, frames_format="raw",
                frame_size=None, stream_index=None, backend="auto"):
    """Yields (index, timecode, PIL image) for the sampled frames.

    The frames come from one of the two sources in
    :mod:`nsfw_scanner.decoder`: the external ffmpeg (the default when
    PyAV is not installed) or PyAV in-process.  See that module for the
    differences.
    """
    return _decoder.frames(path, fps=fps, every_frame=every_frame,
                           max_frames=max_frames, ffmpeg_bin=ffmpeg_bin,
                           hwaccel=hwaccel, frames_format=frames_format,
                           backend=backend, frame_size=frame_size,
                           stream_index=stream_index)


def _audio_stats(path, ffmpeg_bin=None):
    """Loudness/speech profile of the first audio track (ffmpeg astats)."""
    binaries = require_ffmpeg()
    cmd = [ffmpeg_bin or binaries["ffmpeg"], "-v", "info", "-i", path,
           "-af", "astats=metadata=1:reset=0", "-f", "null", "-"]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, timeout=300)
    except (OSError, subprocess.SubprocessError):
        return {}
    text = proc.stderr.decode("utf-8", "replace")
    stats = {}
    for key, cast in (("RMS level dB", float), ("Peak level dB", float),
                      ("Zero crossings rate", float)):
        values = []
        for line in text.splitlines():
            if key in line and "Overall" not in line:
                try:
                    values.append(cast(line.split(key, 1)[1].split()[0]))
                except (ValueError, IndexError):
                    continue
        if values:
            stats[key] = values[-1]
    return stats


def _classify_audio(stats):
    """Turns the astats numbers into a coarse description."""
    rms = stats.get("RMS level dB")
    peak = stats.get("Peak level dB")
    zcr = stats.get("Zero crossings rate")
    if rms is None:
        return {"level": "unknown"}
    if rms < -50:
        level = "silence"
    elif rms < -30:
        level = "quiet"
    elif rms < -8:
        level = "normal"
    else:
        level = "loud"
    speech = "unknown"
    if zcr is not None:
        if zcr < 50:
            speech = "none"
        elif zcr < 600:
            speech = "likely-speech"
        else:
            speech = "noisy-or-music"
    return {"level": level, "rms_db": rms, "peak_db": peak,
            "zero_crossings": zcr, "speech": speech}


def _subtitle_cues(path, probe_info=None, include_embedded=True,
                   include_sidecar=True):
    """Collects subtitle cues from embedded tracks and sidecar files."""
    cues, sources = [], []
    seen = set()

    def add(parsed, source):
        fresh = []
        for cue in parsed:
            key = (round(cue.get("start") or 0.0, 3), cue.get("text", ""))
            if key in seen:
                continue
            seen.add(key)
            fresh.append(cue)
        if fresh:
            cues.extend(fresh)
            source["cues"] = len(fresh)
            sources.append(source)

    if include_sidecar:
        for candidate in _subs.sidecar_subtitles(path):
            parsed = _subs.read_subtitle_file(candidate)
            if parsed and parsed[0]:
                add(parsed[0], {"kind": "sidecar",
                                "file": os.path.basename(candidate)})
    if include_embedded:
        streams = []
        if probe_info:
            streams = [s for s in probe_info.get("streams", [])
                       if s.get("codec_type") == "subtitle"]
        for stream in streams[:8]:
            index = stream.get("index")
            if index is None:
                continue
            raw = _subs.extract_embedded_subtitles(path, index)
            if not raw:
                continue
            parsed = _subs.parse_auto(raw)
            if not parsed:
                continue
            add(parsed, {"kind": "embedded", "stream": index,
                         "codec": stream.get("codec_name")})
    return cues, sources


def _scan_cue_text(cues, lexicon, max_problems):
    """Scans subtitle cues for sensuality terms."""
    from . import scanner as _scanner

    problems = []
    for cue in cues:
        hits = _scanner.find_sensuality_hits(cue.get("text", ""), lexicon)
        if not hits:
            continue
        problems.append({
            "kind": "subtitle",
            "timecode": cue.get("start"),
            "timecode_end": cue.get("end"),
            "line_no": cue.get("index"),
            "keyword": hits[0],
            "text": (cue.get("text", "") or "")[:200],
            "rating": 10,
        })
        if len(problems) >= max_problems:
            break
    return problems


def _sha256_file(path, chunk_size=1048576):
    """Hex sha256 of the scanned file - what a user reports back to the site."""
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            while True:
                chunk = handle.read(chunk_size)
                if not chunk:
                    break
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def _report(progress, stage, done=0, total=None, timecode=None, problems=0,
            started=None, extra=None):
    """Calls the progress callback (never raises: a broken callback must
    not kill the scan)."""
    if progress is None:
        return
    elapsed = time.perf_counter() - started if started else 0.0
    percent = None
    eta = None
    if total:
        percent = min(100.0, round(done * 100.0 / total, 1))
        if done and elapsed > 0:
            eta = round(elapsed / done * (total - done), 1)
    info = {"stage": stage, "done": done, "total": total, "percent": percent,
            "timecode": timecode, "elapsed": round(elapsed, 2), "eta": eta,
            "problems": problems}
    if extra:
        info.update(extra)
    try:
        progress(info)
    except Exception:
        pass


def _sample_synthetic(path, duration, samples=None, ffmpeg_bin=None,
                      face_session=None):
    """SKIPPED: synthetic gate disabled (face detector off by default)"""
    return {"ratio": 0.0, "samples": 0, "noise_free": 0, "faces": 0,
            "face_ratio": 0.0, "synthetic": False, "median_flat": 0.0}


def _frame_at(path, moment, ffmpeg_bin):
    """One frame at a given time (None when it cannot be read)."""
    cmd = [ffmpeg_bin, "-v", "error", "-ss", "%.3f" % moment, "-i", path,
           "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE,
                              stderr=subprocess.DEVNULL, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0 or not proc.stdout:
        return None
    try:
        return Image.open(io.BytesIO(proc.stdout)).convert("RGB")
    except Exception:
        return None


def _media_reason(rating, problems, worst_score=None, frame_count=0,
                  frame_classes=None):
    """The reason + evidence for a media verdict (the deciding problem)."""
    from .scanner import build_reason

    for problem in problems:
        if problem.get("kind") == "subtitle":
            return build_reason(rating, "subtitle", line=problem.get("line_no"),
                                timecode=problem.get("timecode"),
                                keyword=problem.get("keyword"),
                                text=problem.get("text"))
    frame_problems = [problem for problem in problems
                      if problem.get("kind") == "frame"]
    if frame_problems:
        # The deciding problem is the worst one: the highest rating first,
        # then the highest score.
        deciding = max(frame_problems,
                       key=lambda problem: (problem.get("rating") or 0,
                                            problem.get("score") or 0.0))
        classes = (frame_classes or {}).get(deciding.get("frame_index"))
        if deciding.get("class"):
            return build_reason(
                rating, "frame-class",
                score=deciding.get("score"),
                binary=deciding.get("binary"),
                index=deciding.get("frame_index"),
                timecode=deciding.get("timecode"),
                **{"class": deciding.get("class"),
                   "classes": classes})
        return build_reason(rating, "frame", score=deciding.get("score"),
                            index=deciding.get("frame_index"),
                            timecode=deciding.get("timecode"))
    if rating is not None and rating > 1:
        return build_reason(rating, "image", score=worst_score,
                            count=len(problems) or frame_count,
                            total=frame_count)
    if rating is None:
        return build_reason(None, "modesty-not-assessed",
                            reason="no usable image-model output")
    return build_reason(rating, "modesty-no-signal", score=worst_score)


def _audio_reason(rating, problems):
    """The reason for an audio verdict (only the subtitles can decide)."""
    from .scanner import build_reason

    for problem in problems:
        if problem.get("kind") == "subtitle":
            return build_reason(rating, "subtitle", line=problem.get("line_no"),
                                timecode=problem.get("timecode"),
                                keyword=problem.get("keyword"),
                                text=problem.get("text"))
    if rating is None:
        return ("unrated: no audio classifier yet - the rating would come from "
                "the subtitles only", {"stage": "audio-profile"})
    return build_reason(rating, "modesty-no-signal")


def _save_hit_frame(image, source_path, timecode, frame_index, rating=None):
    """Save a hit as an unblurred sample and a blurred preview."""
    source_path = os.path.abspath(source_path)
    source_dir = os.path.dirname(source_path)
    sample_dir = os.path.join(source_dir, "SAMPLES")
    os.makedirs(sample_dir, exist_ok=True)

    seconds = max(0.0, float(timecode if timecode is not None else 0.0))
    milliseconds = round(seconds * 1000)
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    whole_seconds, milliseconds = divmod(milliseconds, 1000)
    timestamp = "%02d.%02d.%02d.%03d" % (
        hours, minutes, whole_seconds, milliseconds)
    stem = os.path.splitext(os.path.basename(source_path))[0]
    base = "%s.frame.%s" % (stem, timestamp)

    suffix = 1
    while True:
        candidate = base if suffix == 1 else "%s-%d" % (base, suffix)
        original_path = os.path.join(sample_dir, candidate + ".png")
        metadata_path = os.path.join(sample_dir, candidate + ".json")
        blurred_path = os.path.join(source_dir, candidate + "-blurred.png")
        if (not os.path.exists(original_path) and
                not os.path.exists(metadata_path) and
                not os.path.exists(blurred_path)):
            break
        suffix += 1

    frame = image.convert("RGB")
    frame.save(original_path, format="PNG")
    blur_radius = max(12, min(frame.size) * 0.025)
    frame.filter(ImageFilter.GaussianBlur(radius=blur_radius)).save(
        blurred_path, format="PNG")
    with open(metadata_path, "w", encoding="utf-8") as metadata_file:
        json.dump({
            "rating": rating,
            "frame_index": frame_index,
            "timecode": timecode,
        }, metadata_file)
    return {"original": original_path, "blurred": blurred_path,
            "metadata": metadata_path, "frame_index": frame_index,
            "timecode": timecode, "rating": rating}


def scan_video(path, fps=1.0, every_frame=False, max_frames=2000,
               max_duration_sec=None, min_report_score=None, max_problems=500,
               details=False, debug=False, subtitles=True, progress=None,
               tiles="off", hwaccel=None, frames_format="raw",
               backend="auto", face_crop=False, face_detector=None,
               save_samples=True, plugins=None, include_frame_ratings=False):
    """Scans a video file: sampled frames + subtitles.

    Returns the standard scanner dict plus ``problems`` (the areas that
    stood out) and the detected file type. By default, each detected frame
    is saved unblurred under ``SAMPLES/`` beside the video and as a blurred
    preview beside the video; set ``save_samples=False`` to disable this.
    """
    from . import scanner as _scanner

    started = time.perf_counter()
    binaries = require_ffmpeg()
    kind = detect_kind(path)
    if min_report_score is None:
        min_report_score = _scanner.NSFW_THRESHOLD_REVEALING

    info = probe(path)
    duration = None
    try:
        duration = float(info.get("format", {}).get("duration"))
    except (TypeError, ValueError):
        duration = None
    if max_duration_sec and duration and duration > max_duration_sec:
        raise ValueError("video is %.1fs long, the limit is %.1fs"
                         % (duration, max_duration_sec))

    nsfw_session, gender_session = _scanner.get_sessions()
    # The plugins (the optional classifiers from the config file) - they
    # may raise the rating, replace it, or only add to the evidence.
    if plugins is None:
        from . import plugins as _plugins_module
        plugins = _plugins_module.load_plugins()
    # The binary (2-class) model is the second opinion the joint model
    # needs: the joint model's "porn" class is a skin detector (it fires
    # on an exposed neck, on a white dress, on a beard), so a rating of
    # 10 only goes through when this model agrees.
    binary_session = _scanner.get_binary_session()
    # The face detector is an **optional external path** (the bundled one
    # lives in nsfw_scanner/faces and can be replaced by any callable with
    # the same signature).  It is deprecated as a default: it costs 14-36
    # ms per frame and the *file rating* does not change without it - only
    # the fine-grained "5" grades get coarser (the gender model then runs
    # on the whole frame instead of a face crop).
    # So the media path does not use it per frame any more; the synthetic
    # gate still uses it (it samples a couple of dozen frames over the
    # whole file, which is negligible) because the "no real face" signal
    # is what separates a render from a filmed scene.
    face_session = _scanner.get_face_session()
    frame_face_session = face_session if face_crop else None
    lexicon = _scanner.sensuality_lexicon()

    # Is this camera footage or a render / screen capture?  The picture
    # model is trained on photographs, so for synthetic content its score
    # does not track nudity (a fully clothed game character scored 0.59).
    synthetic = _sample_synthetic(path, duration,
                                  ffmpeg_bin=binaries["ffmpeg"],
                                  face_session=face_session)
    is_synthetic = bool(synthetic.get("synthetic"))
    explicit_bar = (_scanner.NSFW_SYNTHETIC_EXPLICIT if is_synthetic
                    else _scanner.NSFW_THRESHOLD_EXPLICIT)
    revealing_bar = (_scanner.NSFW_SYNTHETIC_REVEALING if is_synthetic
                     else _scanner.NSFW_THRESHOLD_REVEALING)
    if min_report_score is None:
        min_report_score = revealing_bar
    native_fps = None
    frame_size = None
    stream_index = None
    for stream in info.get("streams", []):
        if stream.get("codec_type") != "video":
            continue
        try:
            num, _, den = str(stream.get("r_frame_rate", "0/1")).partition("/")
            native_fps = round(float(num) / float(den or 1), 3) or None
        except (TypeError, ValueError, ZeroDivisionError):
            native_fps = None
        try:
            width = int(stream.get("width") or 0)
            height = int(stream.get("height") or 0)
        except (TypeError, ValueError):
            width = height = 0
        if width > 0 and height > 0 and (
                frame_size is None or width * height > frame_size[0] * frame_size[1]):
            # The biggest video stream is the one ffmpeg would pick too.
            frame_size = (width, height)
            try:
                stream_index = int(stream.get("index"))
            except (TypeError, ValueError):
                stream_index = None
        if native_fps and frame_size:
            break
    total_frames = int(duration * fps) if (duration and fps and not every_frame) else None
    try:
        file_size = os.path.getsize(path)
    except OSError:
        file_size = None
    _report(progress, "probe", 0, None, started=started,
            extra={"synthetic": is_synthetic,
                   "synthetic_ratio": synthetic.get("ratio"),
                   "duration": duration, "fps": fps,
                   "native_fps": native_fps,
                   "size": file_size,
                   "target": str(path),
                   "every_frame": every_frame,
                   "total": total_frames})

    problems = []
    worst_rating = None
    worst_score = 0.0
    frame_count = 0
    scored = 0
    notable = 0
    romance = 0
    frame_classes = {}
    saved_samples = []
    frame_ratings = [] if include_frame_ratings else None
    loop_started = time.perf_counter()
    inference_seconds = 0.0
    for index, timecode, image in iter_frames(
            path, fps=fps, every_frame=every_frame, max_frames=max_frames,
            ffmpeg_bin=binaries["ffmpeg"], hwaccel=hwaccel,
            frames_format=frames_format, frame_size=frame_size,
            stream_index=stream_index, backend=backend):
        frame_count += 1
        if image is None:
            continue
        inference_started = time.perf_counter()
        detail = _scanner.analyze_pil_image_tiles(
            image, nsfw_session, gender_session, fallback_url=path,
            details=True, face_session=frame_face_session, debug=debug,
            tiles=tiles, face_detector=face_detector,
            binary_session=binary_session, plugins=plugins)
        inference_seconds += time.perf_counter() - inference_started
        score = detail.get("nsfw_score")
        rating = detail.get("rating")
        best_class = detail.get("class")
        if frame_ratings is not None:
            frame_ratings.append({
                "timecode": timecode,
                "rating": detail.get("rating"),
                "class": best_class,
            })
        if rating is not None and (worst_rating is None or rating > worst_rating):
            worst_rating = rating
        if score is not None and score > worst_score:
            worst_score = score
        # A frame is reported when it *decided* something (rating 6 or
        # more), or - for a synthetic file - when it is one of the
        # flagged frames the synthetic gate talks about.  A frame that
        # only shows skin (the joint model's "porn" class without the
        # second model's agreement) is people, not a problem area.
        over_bar = score is not None and score >= min_report_score
        is_romance = best_class == "sexy" and rating == 6
        interesting = ((rating is not None and rating >= 6)
                       or (is_synthetic and over_bar))
        if is_romance:
            romance += 1
        if interesting:
            if save_samples:
                sample = _save_hit_frame(
                    image, path, timecode, index, rating=rating)
                saved_samples.append(sample)
            if over_bar:
                scored += 1
            if score is not None and score > revealing_bar:
                notable += 1
            if len(problems) < max_problems:
                problems.append({
                    "kind": "frame",
                    "timecode": timecode,
                    "frame_index": index,
                    "score": round(float(score), 4) if score is not None else None,
                    "binary": (round(float(detail["binary_score"]), 4)
                               if detail.get("binary_score") is not None else None),
                    "rating": rating,
                    "class": best_class,
                    "gender": detail.get("gender"),
                    "synthetic": is_synthetic,
                })
                if detail.get("classes"):
                    # kept aside for the deciding frame only - the problem
                    # rows stay small (they are stored in the database)
                    frame_classes[index] = detail.get("classes")
        progress_extra = {"worst_score": round(worst_score, 4),
                          "fps": fps, "every_frame": every_frame}
        if interesting and save_samples:
            progress_extra["sample"] = {
                "filename": os.path.basename(sample["original"]),
                "rating": sample["rating"],
                "blurred_path": sample["blurred"],
                "original_path": sample["original"],
            }
        _report(progress, "frames", frame_count, total_frames,
                timecode=timecode, problems=len(problems), started=started,
                extra=progress_extra)

    cues, cue_sources = ([], [])
    subtitles_started = time.perf_counter()
    if subtitles:
        _report(progress, "subtitles", 0, None, started=started)
        cues, cue_sources = _subtitle_cues(path, info)
        sub_problems = _scan_cue_text(cues, lexicon, max_problems)
        if sub_problems:
            problems.extend(sub_problems[:max(0, max_problems - len(problems))])
            worst_rating = max(worst_rating or 1, 10)
    subtitle_seconds = time.perf_counter() - subtitles_started

    if is_synthetic and worst_score <= explicit_bar:
        # The picture model cannot judge a render, a drawing or a UI: report
        # it as such (2) and keep the frames (and their scores) as flagged
        # areas.
        reason = ("synthetic content (2): render / drawing / screen capture - "
                  "the image model is trained on photographs; %d frame(s) "
                  "flagged (worst %s)" % (scored,
                                          "%.2f" % worst_score if worst_score else "n/a"))
        evidence = {"stage": "synthetic", "frames_flagged": scored,
                    "notable_frames": notable,
                    "worst_score": round(worst_score, 4),
                    "synthetic_ratio": synthetic.get("ratio"),
                    "samples": synthetic.get("samples"),
                    "faces_detected": synthetic.get("faces"),
                    "face_ratio": synthetic.get("face_ratio"),
                    "explicit_bar": explicit_bar,
                    "revealing_bar": revealing_bar}
        picture_rating = 2
    else:
        reason, evidence = _media_reason(worst_rating, problems, worst_score,
                                        frame_count, frame_classes)
        evidence = dict(evidence)
        evidence["synthetic"] = is_synthetic
        evidence["synthetic_ratio"] = synthetic.get("ratio")
        evidence["face_ratio"] = synthetic.get("face_ratio")
        picture_rating = worst_rating
    rating = picture_rating
    if any(p.get("kind") == "subtitle" for p in problems):
        rating = 10        # a text verdict stands on its own

    _report(progress, "done", frame_count, total_frames, started=started,
            extra={"rating": rating, "problems": len(problems)})
    total_seconds = time.perf_counter() - started
    loop_seconds = time.perf_counter() - loop_started - subtitle_seconds
    model_kind = getattr(nsfw_session, "_nsfw_kind", "none")
    result = {
        "target": str(path),
        "scanner": "media-video",
        "rating": rating,
        "error": None,
        "reason": reason,
        "evidence": evidence,
        "synthetic": is_synthetic,
        "media_type": kind["media_type"],
        "mime": kind["mime"],
        "mime_sniffed": kind["mime_sniffed"],
        "mime_claimed": kind["mime_claimed"],
        "ext": kind["ext"],
        "container": kind["container"],
        "type_mismatch": kind["type_mismatch"],
        "duration_sec": duration,
        "native_fps": native_fps,
        "frame_count": frame_count,
        "scored_frames": scored,
        "romance_frames": romance,
        "hwaccel": _decoder.last_hwaccel() or (hwaccel or "none"),
        "frames_format": frames_format or "raw",
        "backend": backend or "auto",
        "face_crop": bool(face_crop),
        "model": model_kind,
        "fps": None if every_frame else fps,
        "every_frame": every_frame,
        "problem_count": len(problems),
        "saved_samples": saved_samples,
        "problems": problems,
        "subtitles": {"cues": len(cues), "sources": cue_sources},
        "sha256": _sha256_file(path),
    }
    if frame_ratings is not None:
        result["frame_ratings"] = frame_ratings
    if details:
        result["details"] = {"worst_score": round(worst_score, 4)}
    result["timings"] = {
        "total_sec": round(total_seconds, 3),
        "frame_loop_sec": round(loop_seconds, 3),
        "inference_sec": round(inference_seconds, 3),
        "subtitles_sec": round(subtitle_seconds, 3),
        "ms_per_frame_inference": (round(inference_seconds * 1000 / frame_count, 2)
                                   if frame_count else None),
        "frames_per_sec": (round(frame_count / total_seconds, 2)
                           if total_seconds else None),
        "realtime_x": (round(frame_count / (fps or 0) / total_seconds, 3)
                       if (fps and total_seconds) else None),
    }
    result["elapsed"] = round(total_seconds, 3)
    return result


def scan_audio(path, max_problems=500, details=False, debug=False,
               subtitles=True, progress=None):
    """Scans an audio file.

    Without an audio classifier the *content* cannot be rated honestly, so
    this reports the acoustic profile (loudness / speech likelihood), the
    subtitle verdict if the file carries subtitles, and leaves the rating
    to the subtitle text.
    """
    from . import scanner as _scanner

    started = time.perf_counter()
    require_ffmpeg()
    kind = detect_kind(path)
    lexicon = _scanner.sensuality_lexicon()
    _report(progress, "probe", 0, None, started=started)

    info = probe(path)
    duration = None
    try:
        duration = float(info.get("format", {}).get("duration"))
    except (TypeError, ValueError):
        duration = None
    streams = info.get("streams", [])
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]

    profile = _classify_audio(_audio_stats(path))
    problems = []
    rating = None
    cues, cue_sources = ([], [])
    if subtitles:
        cues, cue_sources = _subtitle_cues(path, info)
        sub_problems = _scan_cue_text(cues, lexicon, max_problems)
        problems.extend(sub_problems)
        if sub_problems:
            rating = 10

    result = {
        "target": str(path),
        "scanner": "media-audio",
        "rating": rating,
        "error": None,
        "reason": _audio_reason(rating, problems),
        "evidence": {"stage": "audio-profile", "level": profile.get("level"),
                     "speech": profile.get("speech")},
        "sha256": _sha256_file(path),
        "media_type": kind["media_type"],
        "mime": kind["mime"],
        "mime_sniffed": kind["mime_sniffed"],
        "mime_claimed": kind["mime_claimed"],
        "ext": kind["ext"],
        "container": kind["container"],
        "type_mismatch": kind["type_mismatch"],
        "duration_sec": duration,
        "audio_streams": len(audio_streams),
        "codecs": sorted({s.get("codec_name") for s in audio_streams
                          if s.get("codec_name")}),
        "profile": profile,
        "problem_count": len(problems),
        "problems": problems,
        "subtitles": {"cues": len(cues), "sources": cue_sources},
        "note": ("no audio classifier yet: the rating comes from the "
                 "subtitles only, the profile is informational"),
    }
    result["elapsed"] = round(time.perf_counter() - started, 3)
    return result


def scan_media(target, **options):
    """Scans any media file: video or audio.

    ``target`` is a local path.  Unknown types raise ValueError.
    """
    kind = detect_kind(target)
    if kind["media_type"] == "video":
        return scan_video(target, **options)
    if kind["media_type"] == "audio":
        options.pop("include_frame_ratings", None)
        return scan_audio(target, **options)
    if kind["media_type"] == "image":
        from . import scanner as _scanner
        nsfw_session, gender_session = _scanner.get_sessions()
        detail = _scanner.analyze_image_file(
            target, nsfw_session, gender_session, details=True,
            face_session=_scanner.get_face_session(),
            binary_session=_scanner.get_binary_session(),
            high_accuracy=options.get("high_accuracy", False))
        reason, evidence = _scanner._image_reason(detail)
        return {"target": str(target), "scanner": "image-file",
                "rating": detail.get("rating"), "error": None,
                "reason": reason, "evidence": evidence,
                "media_type": "image", "mime": kind["mime"],
                "mime_sniffed": kind["mime_sniffed"],
                "mime_claimed": kind["mime_claimed"], "ext": kind["ext"],
                "container": kind["container"],
                "type_mismatch": kind["type_mismatch"],
                "problem_count": 0, "problems": []}
    raise ValueError("not a media file: %r (detected %s)"
                     % (target, kind["media_type"]))
