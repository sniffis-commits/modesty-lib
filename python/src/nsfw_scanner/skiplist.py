"""SRT skiplist parser + writer for .skp skip files.

Format: time [action=skip/blur/mute/blackout (+combo)] [duration] [reason]
"""
import math
import os
import re
from typing import List, Dict, Optional, Any


# ---------- Parser (unchanged, but accepts bare [duration] too) ----------

def parse_line(line: str) -> Optional[Dict]:
    line = line.split("#")[0].strip()
    if not line or line.startswith("#"):
        return None
    m = re.match(r"(\d{1,2}:\d{2}:\d{2}(\.\d+)?)", line)
    if not m:
        return None
    ts = m.group(1)
    rest = line[m.end():].strip()
    actions = []
    dur = 0.0
    reason = ""
    am = re.search(r"\[action=([^\]]+)\]", rest)
    if am:
        actions = [a.strip() for a in am.group(1).split("+") if a.strip()]
        rest = rest[:am.start()] + rest[am.end():]
    # Accept [duration=15.0] OR bare [15.0]
    dm = re.search(r"\[duration=([\d\.]+)\]", rest)
    if not dm:
        dm = re.search(r"\[([\d\.]+)\]", rest)
    if dm:
        dur = float(dm.group(1))
        rest = rest[:dm.start()] + rest[dm.end():]
    # Extract [reason=...] specifically, then strip any remaining brackets
    rm = re.search(r"\[reason=([^\]]+)\]", rest)
    if rm:
        reason = rm.group(1)
        rest = rest[:rm.start()] + rest[rm.end():]
    reason = re.sub(r"[\[\]]", "", rest).strip() + (" " + reason if reason else "")
    reason = reason.strip()
    return {"time": ts, "actions": actions, "duration": dur, "reason": reason, "raw": line}


def read_skiplist(path: str) -> List[Dict]:
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            e = parse_line(line)
            if e:
                entries.append(e)
    return entries


# ---------- Writer / formatting ----------

DEFAULT_ACTIONS = {
    "10": ["blackout", "mute"],
    "9": ["blur", "mute"],
    "8": ["blur"],
    "7": ["blur"],
    "6": ["blur"],
    "subtitle": ["mute"],
}


def format_timecode(sec: float) -> str:
    """seconds -> HH:MM:SS.mmm"""
    sec = max(0.0, float(sec))
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec - h * 3600 - m * 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def _timecode_seconds(timecode: str) -> float:
    hours, minutes, seconds = timecode.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def _frame_rating(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric) or not numeric.is_integer():
        return None
    return int(numeric)


def merge_adjacent(entries: List[Dict], gap_sec: float = 0.0) -> List[Dict]:
    """Merge overlapping or nearby entries with identical actions."""
    if not entries:
        return entries
    out = [entries[0].copy()]
    for e in entries[1:]:
        last = out[-1]
        last_start = _timecode_seconds(last["time"])
        last_end = last_start + last.get("duration", 0.0)
        cur_start = _timecode_seconds(e["time"])
        if (cur_start - last_end) <= gap_sec and e["actions"] == last["actions"]:
            current_end = cur_start + e.get("duration", 0.0)
            last["duration"] = max(last_end, current_end) - last_start
        else:
            out.append(e.copy())
    return out


def entries_from_result(
    result: Dict,
    threshold: int = 6,
    actions_map: Optional[Dict[str, List[str]]] = None,
    gap_sec: float = 2.0,
) -> List[Dict]:
    """Convert a media scan result into timed .skp entries.

    Video CLI scans that request skip-file output include ``frame_ratings``;
    each qualifying frame is skipped through the next later rating-1 sample,
    or to EOF. Older/external results without that timeline retain the
    sampling-interval fallback.
    """
    if actions_map is None:
        actions_map = DEFAULT_ACTIONS
    problems = result.get("problems", [])
    frame_ratings = result.get("frame_ratings")
    if not problems and frame_ratings is None:
        return []
    try:
        if result.get("every_frame"):
            sample_fps = float(result.get("native_fps") or 0.0)
        else:
            sample_fps = float(result.get("fps") or 0.0)
    except (TypeError, ValueError):
        sample_fps = 0.0
    frame_duration = 1.0 / sample_fps if sample_fps > 0 else 1.0
    try:
        media_duration = float(
            result.get("duration_sec") or result.get("duration") or 0.0)
    except (TypeError, ValueError):
        media_duration = 0.0
    # Frame ratings are only collected when a skip file is requested, avoiding
    # a large per-frame payload for ordinary media scans.
    raw = []
    if frame_ratings is not None:
        ordered_frames = []
        try:
            frame_rows = iter(frame_ratings)
        except TypeError:
            frame_rows = iter(())
        for frame in frame_rows:
            if not isinstance(frame, dict):
                continue
            try:
                timecode = float(frame.get("timecode"))
            except (TypeError, ValueError):
                continue
            if math.isfinite(timecode):
                ordered_frames.append((timecode, frame))
        ordered_frames.sort(key=lambda item: item[0])
        scanned_end = max(
            (timecode for timecode, _ in ordered_frames),
            default=0.0,
        )
        eof = media_duration if media_duration > 0 else scanned_end + frame_duration
        ratings = [_frame_rating(frame.get("rating"))
                   for _, frame in ordered_frames]
        next_safe_after = [eof] * len(ordered_frames)
        next_safe = eof
        for index in range(len(ordered_frames) - 1, -1, -1):
            next_safe_after[index] = next_safe
            if ratings[index] == 1:
                next_safe = ordered_frames[index][0]
        for index, (timecode, frame) in enumerate(ordered_frames):
            rating = ratings[index]
            if rating is None:
                continue
            if rating < threshold:
                continue
            duration = max(0.0, next_safe_after[index] - timecode)
            action_key = str(rating)
            acts = actions_map.get(action_key, actions_map.get("6", []))
            raw.append({
                "time": format_timecode(timecode),
                "actions": acts,
                "duration": duration,
                "reason": f"rating {rating} ({frame.get('class', 'frame')})",
            })
    else:
        for problem in problems:
            if problem.get("kind") != "frame" or problem.get("timecode") is None:
                continue
            rating = problem.get("rating", 0)
            if rating < threshold:
                continue
            timecode = float(problem["timecode"])
            duration = frame_duration
            if media_duration > 0:
                remaining = media_duration - timecode
                duration = min(duration, remaining) if remaining > 0 else 0.001
            action_key = str(rating)
            acts = actions_map.get(action_key, actions_map.get("6", []))
            raw.append({
                "time": format_timecode(timecode),
                "actions": acts,
                "duration": duration,
                "reason": f"rating {rating} ({problem.get('class', 'frame')})",
            })

    # Subtitle entries retain their exact cue intervals and do not depend on
    # the video-frame timeline (audio scanning currently has no classifier).
    for p in problems:
        if p.get("kind") == "subtitle" and p.get("rating", 0) >= threshold:
            timecode = p.get("timecode")
            if timecode is None:
                continue
            try:
                dur = float(p.get("timecode_end") or 0.0) - float(timecode)
            except (TypeError, ValueError):
                dur = 0.0
            if dur <= 0:
                dur = 3.0
            acts = actions_map.get("subtitle", ["mute"])
            raw.append({
                "time": format_timecode(timecode),
                "actions": acts,
                "duration": dur,
                "reason": f"subtitle: {p.get('keyword', 'hit')[:50]}",
            })
    # Merge adjacent same-action frames
    raw.sort(key=lambda x: _timecode_seconds(x["time"]))
    return merge_adjacent(raw, gap_sec=0.0)


def sidecar_skipfile(video_path: str) -> str:
    """Return the .skp path next to the video file."""
    stem = os.path.splitext(video_path)[0]
    return stem + ".skp"


def generated_skipfile(video_path: str) -> str:
    """Return a separate generated .skp path without replacing a user's sidecar."""
    stem = os.path.splitext(video_path)[0]
    return stem + ".generated.skp"


def write_skiplist(
    path: str,
    entries: List[Dict],
    header: Optional[Dict[str, Any]] = None,
) -> None:
    """Write entries to a .skp file."""
    lines = []
    lines.append("# NSFW skip file")
    if header:
        parts = []
        if header.get("source"):
            parts.append(f"source: {header['source']}")
        if header.get("sha256"):
            parts.append(f"sha256: {header['sha256']}")
        if header.get("duration"):
            parts.append(f"duration: {header['duration']:.1f}")
        if parts:
            lines.append("# " + "  ".join(parts))
    lines.append("# time [action=...] [duration] [reason]")
    lines.append("")
    for e in entries:
        acts = "+".join(e["actions"]) if e["actions"] else "skip"
        dur_str = f" [duration={e['duration']:.3f}]" if e["duration"] > 0 else ""
        reason = e.get("reason", "")
        reason_str = f" [reason={reason}]" if reason else ""
        lines.append(f"{e['time']} [action={acts}]{dur_str}{reason_str}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def preview_entries_text(entries: List[Dict]) -> str:
    lines = ["# Skip-file preview — " + str(len(entries)) + " entries"]
    for e in entries:
        acts = "+".join(e.get("actions", [])) or "skip"
        lines.append(f"{e['time']} | {acts} | dur={e.get('duration', 0):.1f}s | {e.get('reason', '')}")
    return "\n".join(lines) + "\n"


def write_result_skiplist(
    video_path: str,
    result: Dict,
    threshold: int = 6,
    actions_map: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, Any]:
    """Persist the skip list derived from an existing scan result."""
    entries = entries_from_result(
        result,
        threshold=threshold,
        actions_map=actions_map,
    )
    path = generated_skipfile(video_path)
    duration = result.get("duration_sec") or result.get("duration")
    write_skiplist(
        path,
        entries,
        header={
            "source": os.path.basename(video_path),
            "sha256": result.get("sha256") or result.get("file_sha256"),
            "duration": duration,
        },
    )
    preview_path = path.replace(".skp", ".txt")
    with open(preview_path, "w", encoding="utf-8") as f:
        f.write(preview_entries_text(entries))
    return {"path": path, "preview_path": preview_path, "entry_count": len(entries)}


def parse_actions_string(s: str) -> Dict[str, List[str]]:
    """Parse '10=blackout+mute,7=blur,subtitle=mute' into actions map."""
    out = {}
    for part in s.split(","):
        part = part.strip()
        if not part or "=" not in part:
            continue
        k, v = part.split("=", 1)
        out[k.strip()] = [a.strip() for a in v.split("+") if a.strip()]
    return out