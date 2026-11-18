"""Project persistence for video scan results.
Stores: original filename, sha256 hash, flags, timeline (.skp), sample locations.
"""
from __future__ import annotations
import json
import os
import hashlib
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Dict, Any
from pathlib import Path
from datetime import datetime


@dataclass
class AppFlags:
    media_mode: str = "auto"
    skip_file: Optional[str] = None
    skip_threshold: int = 3
    skip_actions: List[str] = field(default_factory=lambda: ["censor", "report", "none"])
    threshold: int = 5
    with_memd: bool = True
    with_accessibility: bool = False
    with_overlay: bool = False
    force_reextract: bool = False


@dataclass
class SampleInfo:
    filename: str
    hash: str
    rating: Optional[int]
    is_blurred: bool
    size_bytes: int


@dataclass
class TimelineEntry:
    frame: str
    rating: int
    action: str


@dataclass
class VideoProject:
    project_id: str
    original_filename: str
    sha256: str
    timestamp: int
    flags: AppFlags
    samples: List[SampleInfo] = field(default_factory=list)
    timeline: List[TimelineEntry] = field(default_factory=list)

    @classmethod
    def from_scan(cls, video_path: str, flags: AppFlags, scan_result: Dict[str, Any]) -> "VideoProject":
        project_id = Path(video_path).stem
        sha256 = hashlib.sha256(Path(video_path).read_bytes()).hexdigest()
        timestamp = int(datetime.now().timestamp() * 1000)
        return cls(
            project_id=project_id,
            original_filename=os.path.basename(video_path),
            sha256=sha256,
            timestamp=timestamp,
            flags=flags,
            samples=[],
            timeline=[]
        )

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, data: str) -> "VideoProject":
        d = json.loads(data)
        d["flags"] = AppFlags(**d["flags"])
        d["samples"] = [SampleInfo(**s) for s in d.get("samples", [])]
        d["timeline"] = [TimelineEntry(**t) for t in d.get("timeline", [])]
        return cls(**d)

    def save(self, path: str) -> None:
        Path(path).write_text(self.to_json())

    @classmethod
    def load(cls, path: str) -> "VideoProject":
        return cls.from_json(Path(path).read_text())


def generate_sample_paths(project: VideoProject, base_dir: str = "sample_output") -> Dict[str, str]:
    """Generate sample file paths for a project."""
    project_dir = Path(base_dir) / project.project_id
    project_dir.mkdir(parents=True, exist_ok=True)
    blurred = project_dir / f"{project.project_id}_blur.png"
    original = project_dir / f"{project.project_id}.png"
    return {
        "blurred": str(blurred),
        "original": str(original),
        "project_dir": str(project_dir)
    }


def add_sample(project: VideoProject, sample_path: str, rating: Optional[int] = None) -> SampleInfo:
    """Add a sample to the project, generate hash."""
    data = Path(sample_path).read_bytes()
    sha256 = hashlib.sha256(data).hexdigest()
    filename = Path(sample_path).name
    is_blurred = "_blur" in filename
    sample = SampleInfo(
        filename=filename,
        hash=sha256,
        rating=rating,
        is_blurred=is_blurred,
        size_bytes=len(data)
    )
    project.samples.append(sample)
    return sample


def add_timeline(project: VideoProject, frame: str, rating: int, action: str) -> TimelineEntry:
    entry = TimelineEntry(frame=frame, rating=rating, action=action)
    project.timeline.append(entry)
    return entry


def load_skp_timeline(skp_path: str) -> List[TimelineEntry]:
    """Parse .skp file into timeline entries."""
    # .skp format: timecode,rating,action (per line)
    entries = []
    if os.path.exists(skp_path):
        for line in Path(skp_path).read_text().strip().splitlines():
            parts = line.strip().split(',')
            if len(parts) >= 2:
                try:
                    frame = parts[0]
                    rating = int(parts[1])
                    action = parts[2] if len(parts) > 2 else "cover"
                    entries.append(TimelineEntry(frame=frame, rating=rating, action=action))
                except ValueError:
                    continue
    return entries