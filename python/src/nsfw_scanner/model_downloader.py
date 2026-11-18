"""On-demand ONNX model download for the installed wheel.

The PyPI wheel does **not** bundle the model weights (they are ~230 MB in
total, above PyPI's 100 MB per-file limit). Instead, the models are hosted on
a GitHub release and downloaded into a per-user cache the first time a model
is needed.

Configuration (all optional):
    NSFW_MODEL_DIR       explicit directory to look for / place models
    NSFW_MODEL_BASE_URL  override the download base URL
    NSFW_AUTO_DOWNLOAD   "0" to disable automatic download (default: enabled)

The download is best-effort: if it fails, the loaders simply return None and
the scanner degrades to text/URL-keyword mode as before.
"""

import hashlib
import os
import sys
import urllib.request
from pathlib import Path

__all__ = ["MODEL_FILES", "model_cache_dir", "ensure_model", "base_url"]

# Default release that hosts the model files (set at publish time).
DEFAULT_BASE_URL = os.environ.get(
    "NSFW_MODEL_BASE_URL",
    "https://github.com/sniffis-commits/modesty-models/releases/download/models-v1.0.0/",
)

# filename -> (size in bytes, optional sha256 prefix check - '' disables)
MODEL_FILES = {
    "nsfw_joint_model.onnx": (89_387_308, ""),
    "fairface_int8.onnx": (88_686_233, ""),
    "nsfw_model.onnx": (23_590_724, ""),
    "gender_model.onnx": (23_935_566, ""),
    "face_model.onnx": (1_270_727, ""),
    "yolo_best.onnx": (12_279_359, ""),
}


def base_url() -> str:
    return os.environ.get("NSFW_MODEL_BASE_URL", DEFAULT_BASE_URL).rstrip("/") + "/"


def model_cache_dir() -> Path:
    """Directory where downloaded models are cached (per user)."""
    env = os.environ.get("NSFW_MODEL_DIR")
    if env:
        return Path(env).expanduser()
    if sys.platform == "win32":
        root = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif sys.platform == "darwin":
        root = Path.home() / "Library" / "Application Support"
    else:
        root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return root / "modesty-nsfw" / "models"


def _auto_enabled() -> bool:
    return os.environ.get("NSFW_AUTO_DOWNLOAD", "1") not in ("0", "false", "no")


def ensure_model(filename: str, verbose: bool = False) -> Path | None:
    """Ensure a model file exists locally; download it if needed.

    Search order: NSFW_MODEL_DIR > user cache > package_dir/models.
    Returns the absolute path when the file is available, else None.
    """
    # 1. Explicit env dir
    env = os.environ.get("NSFW_MODEL_DIR")
    if env:
        candidate = Path(env).expanduser() / filename
        if candidate.exists():
            return candidate.resolve()

    # 2. User cache
    cached = model_cache_dir() / filename
    if cached.exists() and cached.stat().st_size > 0:
        return cached.resolve()

    # 3. Bundled in the package (source checkout / unpacked wheel)
    package_models = Path(__file__).resolve().parent / "models"
    bundled = package_models / filename
    if bundled.exists():
        return bundled.resolve()

    # 4. Download on demand
    if not _auto_enabled() or filename not in MODEL_FILES:
        return None
    url = base_url() + filename
    expected_size, expected_sha = MODEL_FILES[filename]
    try:
        cached.parent.mkdir(parents=True, exist_ok=True)
        tmp = cached.with_suffix(cached.suffix + ".part")
        if verbose:
            print(f"[modesty] downloading {filename} from {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "modesty_nsfw_scanner/0.3"})
        with urllib.request.urlopen(req, timeout=60) as response, open(tmp, "wb") as fh:
            while True:
                chunk = response.read(1 << 16)
                if not chunk:
                    break
                fh.write(chunk)
        if expected_size and tmp.stat().st_size != expected_size:
            if verbose:
                print(f"[modesty] size mismatch for {filename}")
            tmp.unlink(missing_ok=True)
            return None
        if expected_sha:
            digest = hashlib.sha256(tmp.read_bytes()).hexdigest()
            if not digest.startswith(expected_sha):
                tmp.unlink(missing_ok=True)
                return None
        tmp.replace(cached)
        return cached.resolve()
    except Exception as error:  # network / permissions - degrade gracefully
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        if verbose:
            print(f"[modesty] could not download {filename}: {error}")
        return None
