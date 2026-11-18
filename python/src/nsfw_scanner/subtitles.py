"""Subtitle handling: parse SRT / VTT / ASS, find them in a media file,
and scan their text for sensuality terms.

Pure Python, no dependencies.  Timestamps are normalised to seconds
(float) so they can be compared with frame timecodes.
"""

import os
import re
import subprocess

# 00:00:12,345  /  00:00:12.345  /  01:02:03,456
_TS_RE = re.compile(
    r"(?:(\d+):)?(\d{1,2}):(\d{1,2})(?:[.,](\d{1,3}))?")

SUBTITLE_EXTENSIONS = (".srt", ".vtt", ".ass", ".ssa", ".sub")


def parse_timestamp(text):
    """'00:01:02,500' -> 62.5 seconds.  None when unparsable."""
    if not text:
        return None
    match = _TS_RE.search(str(text).strip())
    if not match:
        return None
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    seconds = int(match.group(3) or 0)
    millis = int((match.group(4) or "0").ljust(3, "0")[:3])
    return hours * 3600 + minutes * 60 + seconds + millis / 1000.0


def _clean_text(text):
    """Strips subtitle markup (tags, braces, ASS overrides)."""
    text = re.sub(r"<[^>]+>", "", text)          # <i>, <c.colorE5E5E5>
    text = re.sub(r"\{[^}]*\}", "", text)        # {\an8} ASS override blocks
    text = text.replace("\\N", " ").replace("\\n", " ")
    return " ".join(text.split())


def parse_srt(text):
    """Parses SubRip text into [{index, start, end, text}]."""
    cues = []
    block = []
    for raw in (text or "").splitlines():
        line = raw.strip("\r\n")
        if not line.strip():
            if block:
                cues.append(_parse_srt_block(block))
                block = []
            continue
        block.append(line)
    if block:
        cues.append(_parse_srt_block(block))
    return [c for c in cues if c]


def _parse_srt_block(block):
    index, start, end, body = None, None, None, []
    for pos, line in enumerate(block):
        if pos == 0 and line.strip().isdigit():
            index = int(line.strip())
            continue
        if "-->" in line:
            left, _, right = line.partition("-->")
            start = parse_timestamp(left)
            end = parse_timestamp(right)
            body = block[pos + 1:]
            break
    if start is None:
        return None
    return {"index": index if index is not None else len(body),
            "start": start, "end": end, "text": _clean_text(" ".join(body))}


def parse_vtt(text):
    """Parses WebVTT text into the same cue shape as parse_srt."""
    text = re.sub(r"^WEBVTT.*$", "", text or "", flags=re.MULTILINE)
    text = re.sub(r"^(NOTE|STYLE|REGION).*$", "", text, flags=re.MULTILINE)
    return parse_srt(text)


_ASS_LINE_RE = re.compile(
    r"^Dialogue:\s*[^,]*,([^,]*),([^,]*),([^,]*),([^,]*),([^,]*),(.*)$")


def parse_ass(text):
    """Parses Advanced SubStation Alpha dialogue lines."""
    cues = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        match = _ASS_LINE_RE.match(line)
        if not match:
            continue
        start = parse_timestamp(match.group(1))
        end = parse_timestamp(match.group(2))
        body = _clean_text(match.group(6))
        if start is None or not body:
            continue
        cues.append({"index": len(cues) + 1, "start": start, "end": end,
                     "text": body})
    return cues


def parse_auto(text):
    """Picks the right parser for the content (or the file extension)."""
    head = (text or "")[:400].lstrip()
    if head.upper().startswith("WEBVTT"):
        return parse_vtt(text)
    if "[Script Info]" in (text or "")[:200] or "Dialogue:" in (text or "")[:4000]:
        return parse_ass(text)
    return parse_srt(text)


def read_subtitle_file(path):
    """Reads a subtitle file, returns (cues, raw_text).  None if unreadable."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            raw = handle.read()
    except OSError:
        return None
    return parse_auto(raw), raw


def sidecar_subtitles(path):
    """Finds subtitle files next to the media file (same stem, known ext)."""
    stem = os.path.splitext(path)[0]
    found = []
    for ext in SUBTITLE_EXTENSIONS:
        candidate = stem + ext
        if os.path.isfile(candidate):
            found.append(candidate)
        # also "<name>.<lang>.srt"
        folder = os.path.dirname(path)
        base = os.path.basename(stem)
        try:
            entries = os.listdir(folder or ".")
        except OSError:
            entries = []
        for entry in entries:
            if entry.lower().endswith(ext) and entry.startswith(base + "."):
                full = os.path.join(folder, entry)
                if full not in found and os.path.isfile(full):
                    found.append(full)
    return found


def extract_embedded_subtitles(path, stream_index=0, ffmpeg_bin="ffmpeg"):
    """Extracts an embedded subtitle track as SRT text.

    ``stream_index`` is the *absolute* stream index from ffprobe (not the
    n-th subtitle stream) - ``-map 0:<index>`` is unambiguous that way.

    Returns None when the stream is unreadable (or ffmpeg fails).
    """
    cmd = [ffmpeg_bin, "-v", "error", "-i", path,
           "-map", "0:%d" % stream_index, "-f", "srt", "-"]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0 or not proc.stdout:
        return None
    return proc.stdout.decode("utf-8", "replace")
