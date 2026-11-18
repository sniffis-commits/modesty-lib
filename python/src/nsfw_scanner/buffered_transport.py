# Buffered transport for the NSFW scanner.
# Falls back to pipe-based transport when Rust daemon is unavailable.

import os
import subprocess

MEMD_ENABLED = os.environ.get("NSFW_MEMD", "1") == "1"

try:
    import nsfw_memd_bindings as _memd
    _MEMD_AVAILABLE = True
except ImportError:
    _MEMD_AVAILABLE = False


class Frame:
    __slots__ = ("pts_us", "width", "height", "format", "stride", "payload", "seq")
    def __init__(self, pts_us, width, height, format, stride, payload, seq):
        self.pts_us = pts_us
        self.width = width
        self.height = height
        self.format = format
        self.stride = stride
        self.payload = payload
        self.seq = seq


class BufferedTransport:
    def __init__(self, video_path, fps=1.0, capacity=100, max_payload=2000000):
        self.video_path = video_path
        self.fps = fps
        self.capacity = capacity
        self.max_payload = max_payload
        self._use_memd = MEMD_ENABLED and _MEMD_AVAILABLE
        self._ffmpeg_proc = None
        self._closed = False
        if not self._use_memd:
            self._init_pipe()

    def _init_pipe(self):
        self._ffmpeg_proc = subprocess.Popen(
            ['ffmpeg', '-i', self.video_path, '-vf', 'fps=' + str(self.fps),
             '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1'],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )

    def frames(self):
        if self._use_memd:
            yield from self._frames_memd()
        else:
            yield from self._frames_pipe()

    def _frames_pipe(self):
        width, height = 224, 224
        frame_size = width * height * 3
        while not self._closed:
            data = self._ffmpeg_proc.stdout.read(frame_size)
            if len(data) < frame_size:
                break
            yield Frame(0, width, height, 1, width * 3, data, 0)

    def _frames_memd(self):
        yield from []

    def close(self):
        if self._ffmpeg_proc:
            self._ffmpeg_proc.terminate()
            self._ffmpeg_proc.wait()
        self._closed = True

    @classmethod
    def is_available(cls):
        return _MEMD_AVAILABLE and MEMD_ENABLED

    @classmethod
    def status(cls):
        if not MEMD_ENABLED:
            return 'disabled (NSFW_MEMD=0)'
        if not _MEMD_AVAILABLE:
            return 'unavailable (no nsfw_memd bindings)'
        return 'available'
