"""Low-performance (sequential, low-memory) NSFW URL scanner.

Rates a website on a scale of 1 to 10 based on its textual content and
the images it embeds:

    1  = No unsafe signal detected by a successful scan
    5  = Women/girls detected, no stronger unsafe signal
    7  = Revealing but not fully explicit content
    10 = Explicit adult content

This is the sequential counterpart of ``NSFW_url_scan_threaded.py``:
images are analyzed one by one so memory/CPU usage stays flat, at the
cost of speed. Heavy ML dependencies (tensorflow, onnxruntime) are
optional -- if the ONNX model files are missing, the scanner gracefully
degrades to text/URL-keyword rating instead of crashing.
"""

import hashlib
import io
import os
import re
import time
from urllib.parse import urljoin, urlparse

import numpy as np
import requests
from bs4 import BeautifulSoup
from PIL import Image

from .badwords import badword_rating, domain_badwords
from .lexicons import (
    BUILTIN_SAFE_MASK,
    BUILTIN_SUBSTRING_TERMS,
    BUILTIN_TERMS,
    load_lexicon,
    sensuality_lexicon,
)

try:
    import onnxruntime as ort
except ImportError:  # optional: text-only mode still works
    ort = None

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default action map for .skp skip files (rating -> actions)
DEFAULT_ACTIONS = {
    "10": ["blackout", "mute"],
    "9": ["blur", "mute"],
    "8": ["blur"],
    "7": ["blur"],
    "6": ["blur"],
    "subtitle": ["mute"],
}

ADULT_KEYWORDS = list(BUILTIN_TERMS)

# Terms that are matched as substrings (long, unambiguous ones only) and
# the look-alike words that are masked before that happens.
SUBSTRING_KEYWORDS = list(BUILTIN_SUBSTRING_TERMS)
SAFE_MASK_WORDS = list(BUILTIN_SAFE_MASK)

#delete this its useless
SAFE_KEYWORDS = [
    # Retained for API compatibility only. Safe words NEVER force a safe
    # rating on their own -- they show up on adult pages too
    # ("family", "church", ... as fetish categories, disclaimers, etc.).
    "education", "cartoon", "kids", "family", "church", "bible",
    "parenting", "child",
    # Hebrew safe keywords.
    "ילדים", "ילד", "ילדה", "משפחה", "חינוך", "דת", "תורה", "הורים",
    "בית ספר",
]

# Hebrew attaches prepositions/articles directly to the word (בסקס, הפורנו),
# so Hebrew keywords allow one optional prefix letter.
HEBREW_PREFIXES = "בהוכלמש"

def _env_float(name, default):
    """A float from the environment, with a hard-coded failsafe default."""
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def _env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return int(default)


# --- thresholds ------------------------------------------------------------
# Every one of them is configurable (environment variable first, hard-coded
# failsafe default second) so a deployment can tune them without a rebuild.
# Threshold mode: "default", "test2" (lower revealing threshold, higher explicit bar)
# Can also be set via NSFW_THRESHOLD_MODE env var.
import os as _os
_THRESHOLD_MODE = _os.environ.get("NSFW_THRESHOLD_MODE", "default").lower()

if _THRESHOLD_MODE == "test2":
    # test2: lower binary threshold for revealing (6/7), raise explicit bar
    NSFW_THRESHOLD_EXPLICIT = _env_float("NSFW_THRESHOLD_EXPLICIT", 0.6)
    NSFW_THRESHOLD_REVEALING = _env_float("NSFW_THRESHOLD_REVEALING", 0.12)
    NSFW_THRESHOLD_MIDDLE = _env_float("NSFW_THRESHOLD_MIDDLE", 0.35)
    NSFW_JOINT_STRONG = _env_float("NSFW_JOINT_STRONG", 0.75)
    NSFW_SEXY_STRONG = _env_float("NSFW_SEXY_STRONG", 0.75)
    NSFW_SEXY_HIGH = _env_int("NSFW_SEXY_HIGH", 0)
    FEMALE_THRESHOLD = _env_float("NSFW_FEMALE_THRESHOLD", 0.5)
elif _THRESHOLD_MODE == "test3":
    # test3: middle band between revealing (6) and explicit (10)
    # binary > 0.5 -> 10, binary 0.35-0.5 -> 7/8, binary < 0.35 -> 6
    NSFW_THRESHOLD_EXPLICIT = _env_float("NSFW_THRESHOLD_EXPLICIT", 0.5)
    NSFW_THRESHOLD_REVEALING = _env_float("NSFW_THRESHOLD_REVEALING", 0.2)
    NSFW_THRESHOLD_MIDDLE = _env_float("NSFW_THRESHOLD_MIDDLE", 0.35)
    NSFW_JOINT_STRONG = _env_float("NSFW_JOINT_STRONG", 0.8)
    NSFW_SEXY_STRONG = _env_float("NSFW_SEXY_STRONG", 0.8)
    NSFW_SEXY_HIGH = _env_int("NSFW_SEXY_HIGH", 0)
    FEMALE_THRESHOLD = _env_float("NSFW_FEMALE_THRESHOLD", 0.5)
else:
    # default (original)
    NSFW_THRESHOLD_EXPLICIT = _env_float("NSFW_THRESHOLD_EXPLICIT", 0.5)
    NSFW_THRESHOLD_REVEALING = _env_float("NSFW_THRESHOLD_REVEALING", 0.2)
    NSFW_THRESHOLD_MIDDLE = _env_float("NSFW_THRESHOLD_MIDDLE", 0.35)
    NSFW_JOINT_STRONG = _env_float("NSFW_JOINT_STRONG", 0.8)
    NSFW_SEXY_STRONG = _env_float("NSFW_SEXY_STRONG", 0.8)
    NSFW_SEXY_HIGH = _env_int("NSFW_SEXY_HIGH", 0)
    FEMALE_THRESHOLD = _env_float("NSFW_FEMALE_THRESHOLD", 0.5)

# Synthetic content (3D render / UI screenshot / drawing): the picture model
# is trained on photographs, so its score means nothing there.  These decide
# when we call a file synthetic and how high the bar is for such content.
NSFW_SYNTHETIC_SAMPLES = _env_int("NSFW_SYNTHETIC_SAMPLES", 24)
NSFW_SYNTHETIC_NOISE_PCTL = _env_float("NSFW_SYNTHETIC_NOISE_PCTL", 25)
NSFW_SYNTHETIC_NOISE_MAX = _env_float("NSFW_SYNTHETIC_NOISE_MAX", 0)
NSFW_SYNTHETIC_RATIO = _env_float("NSFW_SYNTHETIC_RATIO", 0.7)
NSFW_SYNTHETIC_FACE_RATIO = _env_float("NSFW_SYNTHETIC_FACE_RATIO", 0.2)
NSFW_SYNTHETIC_EXPLICIT = _env_float("NSFW_SYNTHETIC_EXPLICIT", 0.85)
NSFW_SYNTHETIC_REVEALING = _env_float("NSFW_SYNTHETIC_REVEALING", 0.5)
NSFW_SYNTHETIC_FLAT_MIN = _env_float("NSFW_SYNTHETIC_FLAT_MIN", 0.35)

RATING_PHRASES = {
    1: "no unsafe signal detected",
    2: "synthetic - not rated, scan it yourself",
    5: "women/girls present, no stronger unsafe signal detected",
    6: "romance - clothed, affectionate",
    7: "revealing, not fully explicit",
    8: "strongly suggestive",
    10: "explicit adult content",
}

# One short word per rating - the vocabulary of the open API (so a client
# can show it in any language without parsing our sentences).
RATING_VERDICTS = {
    1: "modesty-no-signal",
    2: "synthetic",
    5: "people-female",
    6: "romance",
    7: "revealing",
    8: "suggestive",
    10: "explicit",
}


def verdict_for(rating):
    """The short verdict word of a rating (None when there is no rating)."""
    if rating is None:
        return None
    return RATING_VERDICTS.get(int(rating), "unknown")

# The five classes of the joint open_nsfw model, in output order.
#   drawings - a drawn/cartoon scene that is not explicit
#   hentai   - a drawn/cartoon scene that is explicit
#   neutral  - nothing to report
#   porn     - a photographic sexual act
#   sexy     - sensual / suggestive, but not an act
NSFW_CLASSES = ("drawings", "hentai", "neutral", "porn", "sexy")

# What a class means for the rating.  "drawings" and "neutral" decide
# nothing on their own - the gender model may still add 5 (a woman in the
# picture), and the synthetic gate may still call the file 2.
#
# The "sexy" class has two levels - a low one (mildly sensual: a kiss, a
# dancer in a bikini) and a high one (strongly sensual, still not an act).
# The scale wants SEXY_LOW -> 6 (romance) and SEXY_HIGH -> 8 (the grade
# that is still reserved).  That split is NOT implemented: the whole
# class maps to 6, because the band cannot be calibrated reliably on
# real footage yet - see the ROADMAP (Phase 6).
NSFW_CLASS_RATINGS = {"porn": 10, "hentai": 10, "sexy": 6}

# Model file names.  Two NSFW models are supported:
#   the joint (5-class) one - the accurate path, it can tell "sexy" (6)
#   from "porn" (10), but it is a ResNet-50 at 299px (~32 ms/frame);
#   the binary (2-class) one - the fast path (MobileNetV2 at 224px,
#   ~7 ms/frame), no romance grade, but it runs on old phones.
NSFW_JOINT_MODEL = "nsfw_joint_model.onnx"
NSFW_BINARY_MODEL = "nsfw_model.onnx"
NSFW_STUDENT_MODEL = "nsfw_student.onnx"
BUILTIN_CLASSIFIERS = (
    ("nsfw-joint", NSFW_JOINT_MODEL, "five-class NSFW classifier"),
    ("nsfw-binary", NSFW_BINARY_MODEL, "binary NSFW classifier"),
    ("nsfw-student", NSFW_STUDENT_MODEL, "distilled NSFW classifier"),
    ("gender", "gender_model.onnx", "gender refinement classifier"),
    ("face-detector", "face_model.onnx", "face-crop detector"),
    ("yolo", "yolo_best.onnx", "optional localized-content refinement"),
)

# Which model to use: "auto" (the joint one when we have it, else the
# binary), "joint" or "binary".  Configurable by the NSFW_MODEL_MODE
# environment variable or by set_model_mode().
MODEL_MODES = ("auto", "joint", "binary", "student")
_MODEL_MODE = (os.environ.get("NSFW_MODEL_MODE") or "auto").strip().lower()
if _MODEL_MODE not in MODEL_MODES:
    _MODEL_MODE = "auto"


def rating_phrase(rating):
    """Human phrase for a numeric rating; 'unrated' for anything else."""
    return RATING_PHRASES.get(rating, "unrated")

DEFAULT_TIMEOUT = 8
DEFAULT_MAX_IMAGES = 20  # low-perf: bound the work per page

# Pretend to be a real browser by default (some sites block obvious bots).
BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0 Safari/537.36"
)
# Honest fallback identity. Switch with set_user_agent("scanner").
SCANNER_USER_AGENT = "NSFWScanner/1.0 (content-rating research scanner)"

USER_AGENT = BROWSER_USER_AGENT
DEFAULT_HEADERS = {"User-Agent": BROWSER_USER_AGENT}  # kept for compatibility


def _resolve_user_agent(agent):
    """Maps "browser"/"scanner" shortcuts to full strings; passes others through."""
    if isinstance(agent, str) and agent.lower() == "browser":
        return BROWSER_USER_AGENT
    if isinstance(agent, str) and agent.lower() == "scanner":
        return SCANNER_USER_AGENT
    return str(agent)


def set_user_agent(agent):
    """Sets the global User-Agent. Accepts "browser", "scanner", or any
    custom string. Returns the previous value."""
    global USER_AGENT
    previous = USER_AGENT
    USER_AGENT = _resolve_user_agent(agent)
    return previous


def _headers(user_agent=None):
    """Request headers, honoring a per-call override or the global UA."""
    return {"User-Agent": _resolve_user_agent(user_agent) if user_agent else USER_AGENT}

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff", ".svg")

# Cached sessions so rate_website() does not reload the models on every call.
_session_cache = {}

# Bundled weights ship inside the package (src/nsfw_scanner/models/).
PACKAGE_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "models")


def _default_model_path(filename):
    """Resolves a model file: NSFW_MODEL_DIR env > bundled models > cwd.

    The PyPI wheel does not bundle the weights; when a model is missing it
    is downloaded on demand into a per-user cache (see model_downloader).
    """
    env_dir = os.environ.get("NSFW_MODEL_DIR")
    if env_dir:
        candidate = os.path.join(env_dir, filename)
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    candidate = os.path.join(PACKAGE_MODELS_DIR, filename)
    if os.path.exists(candidate):
        return os.path.abspath(candidate)
    if os.path.exists(filename):
        return os.path.abspath(filename)
    # On-demand download for wheels (no bundled weights).
    try:
        from .model_downloader import ensure_model
        downloaded = ensure_model(filename)
        if downloaded is not None:
            return os.path.abspath(downloaded)
    except Exception:
        pass
    return os.path.abspath(candidate)  # canonical spot; loaders return None


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def _make_session(model_path):
    """Builds an InferenceSession with ORT chatter silenced.

    log_severity_level=3 hides verbose/info/warning logs (e.g. the
    'initializer appears in graph inputs' warnings from the old Caffe
    conversions); real errors still print.
    """
    options = ort.SessionOptions()
    options.log_severity_level = 3
    return ort.InferenceSession(model_path, sess_options=options,
                                providers=["CPUExecutionProvider"])


def model_mode():
    """The current NSFW model mode: 'auto' | 'joint' | 'binary'."""
    return _MODEL_MODE


def set_model_mode(mode):
    """Switches the NSFW model: 'auto', 'joint' or 'binary'.

    'joint'  - the 5-class model: the class decides (sexy -> 6), slower.
    'binary' - the 2-class model: the thresholds decide, ~4x faster, no
               romance grade - the path for old phones and small servers.
    'auto'   - the joint one when it is there, else the binary.

    Returns the previous mode.  The session cache is dropped, so the next
    scan loads the other model.
    """
    global _MODEL_MODE
    previous = _MODEL_MODE
    mode = (mode or "auto").strip().lower()
    if mode not in MODEL_MODES:
        mode = "auto"
    _MODEL_MODE = mode
    _session_cache.clear()
    return previous


def _resolve_nsfw_model_path(model_path=None):
    """The NSFW model to use.

    ``NSFW_MODEL`` (environment) picks a file by name - that is how the
    distilled student (``nsfw_student.onnx``) replaces the two teachers.
    Then the mode: ``joint`` (the 5-class model), ``binary`` (the fast
    2-class one) or ``student``.  ``auto`` takes the joint model when we
    have it, else the binary one.

    The joint model is strictly more informative than the old binary one
    (it can tell "sexy" from "porn", and it calls an innocent close-up
    "neutral" instead of "porn"), so it wins when both are around.
    """
    if model_path:
        return os.path.abspath(model_path)
    named = os.environ.get("NSFW_MODEL")
    if named:
        return _default_model_path(named)
    mode = _MODEL_MODE
    if mode == "binary":
        return _default_model_path(NSFW_BINARY_MODEL)
    if mode == "student":
        return _default_model_path(NSFW_STUDENT_MODEL)
    if mode == "joint":
        return _default_model_path(NSFW_JOINT_MODEL)
    joint = _default_model_path(NSFW_JOINT_MODEL)
    if os.path.exists(joint):
        return joint
    return _default_model_path(NSFW_BINARY_MODEL)


def _model_kind(session):
    """'joint' (5 classes), 'student' (5 classes + 1 nsfw score) or
    'binary' (2 classes)."""
    try:
        outputs = session.get_outputs()
        if len(outputs) == 2:
            return "student"
        shape = outputs[0].shape
        count = int(shape[-1]) if shape else 0
    except Exception:
        count = 0
    if count == len(NSFW_CLASSES):
        return "joint"
    return "binary"


def load_nsfw_model(model_path=None):
    """Loads the NSFW model. Returns None if unavailable (text-only mode).

    The bundled model is the joint 5-class one; the session is tagged with
    its kind (``session._nsfw_kind``) so the scorer knows whether it gets
    a class vector, a class vector plus a calibrated score (the distilled
    student) or a single "nsfw" probability.

    ``NSFW_MODEL`` (environment) picks a file by name - that is how the
    distilled student (``nsfw_student.onnx``) replaces the two teachers.
    """
    if ort is None:
        return None
    model_path = _resolve_nsfw_model_path(model_path)
    if not os.path.exists(model_path):
        return None
    session = _make_session(model_path)
    try:
        session._nsfw_kind = _model_kind(session)
    except Exception:
        session._nsfw_kind = "binary"
    return session


def load_gender_model(model_path=None):
    """Loads a gender classification model. Returns None if unavailable.

    Defaults to the FairFace ViT INT8 model.  Set NSFW_GENDER_MODEL to
    override (e.g. back to the shipped Levi/Adience gender_model.onnx).
    """
    if ort is None:
        return None
    if model_path is None:
        model_path = os.environ.get("NSFW_GENDER_MODEL")
    if model_path is None or not os.path.exists(model_path):
        # Prefer the same FairFace model bundled by the Android build.
        candidates = [
            _default_model_path("fairface_int8.onnx"),
            os.path.join(
                os.getcwd(), "eval", "models", "fairface_onnx_community",
                "gender_int8.onnx",
            ),
        ]
        here = os.path.dirname(os.path.abspath(__file__))
        candidates.append(os.path.join(
            here, "..", "..", "..", "..", "..",
            "eval", "models", "fairface_onnx_community", "gender_int8.onnx",
        ))
        model_path = next((path for path in candidates if os.path.exists(path)),
                          _default_model_path("gender_model.onnx"))
    if not os.path.exists(model_path):
        return None
    session = _make_session(model_path)
    model_name = os.path.basename(model_path).lower()
    session._nsfw_gender_output_order = (
        ("female", "male")
        if model_name in ("fairface_int8.onnx", "gender_int8.onnx")
        else ("male", "female")
    )
    return session


_binary_cache = {}


def get_binary_session(model_path=None):
    """The binary (2-class) NSFW model - the calibrated "is this explicit?".

    It is the second opinion the joint model needs: the joint model's
    "porn" class is a skin detector (it fires on an exposed neck, on a
    white dress, on a beard), while the binary model's score stays low
    on ordinary people.  A rating of 10 needs this model to agree.
    """
    if ort is None:
        return None
    path = os.path.abspath(model_path or _default_model_path(NSFW_BINARY_MODEL))
    if path not in _binary_cache:
        session = load_nsfw_model(path)
        # Only the *binary* model may answer here - a joint model would
        # be scored as if it were the calibrated one.
        if session is not None and getattr(session, "_nsfw_kind",
                                          "binary") != "binary":
            session = None
        _binary_cache[path] = session
    return _binary_cache[path]


def get_sessions(nsfw_path=None, gender_path=None):
    """Returns cached (nsfw_session, gender_session), loading once."""
    nsfw_path = _resolve_nsfw_model_path(nsfw_path)
    if gender_path is None:
        gender_session = load_gender_model()  # Uses FF INT8 by default
        cache_key_gender = None
    else:
        gender_path = os.path.abspath(gender_path)
        gender_session = load_gender_model(gender_path)
        cache_key_gender = gender_path
    key = (nsfw_path, cache_key_gender)
    if key not in _session_cache:
        _session_cache[key] = (load_nsfw_model(nsfw_path), gender_session)
    return _session_cache[key]


# ---------------------------------------------------------------------------
# YOLO high-accuracy model (optional, loaded lazily)
# ---------------------------------------------------------------------------

_yolo_cache = {}


def _default_yolo_path():
    """Resolve the bundled YOLOv8 NSFW detection model path."""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "___TESTS", "yolo_best.pt"),
        os.path.join(here, "..", "..", "..", "___TESTS", "yolo_best.pt"),
        "___TESTS/yolo_best.pt",
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    return None


def get_yolo_model(model_path=None):
    """Returns a cached YOLOv8 model, loading once.

    Uses ``yolo_best.pt`` (custom NSFW-trained YOLOv8) by default. Returns
    None if ultralytics is not installed or the model file is missing.
    """
    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError:
        return None
    path = model_path or _default_yolo_path()
    if path is None or not os.path.exists(path):
        return None
    if path not in _yolo_cache:
        _yolo_cache[path] = YOLO(path)
    return _yolo_cache[path]


# YOLO class names that indicate exposed (vs covered) body parts, grouped by
# the rating tier they imply when detected with confidence > 0.45.
YOLO_EXPOSED_GENITALIA = {
    "FEMALE_GENITALIA_EXPOSED", "MALE_GENITALIA_EXPOSED",
}
YOLO_EXPOSED_BREASTS = {"FEMALE_BREAST_EXPOSED", "MALE_BREAST_EXPOSED"}
YOLO_EXPOSED_BUTTOCKS = {"BUTTOCKS_EXPOSED"}
YOLO_EXPOSED_BELLY = {"BELLY_EXPOSED"}
YOLO_COVERED_BREASTS = {"FEMALE_BREAST_COVERED"}
YOLO_COVERED_GENITALIA = {"FEMALE_GENITALIA_COVERED", "MALE_GENITALIA_COVERED"}
YOLO_COVERED_BUTTOCKS = {"BUTTOCKS_COVERED"}


def _yolo_refine_rating(rating, img, threshold=0.45):
    """Refine a scanner rating using YOLO localized body-part detection.

    Parameters
    ----------
    rating : int or None
        The rating already decided by the fast scanner (10 / 7 / 6 / 5 / 1).
    img : PIL.Image
        The image to run YOLO on.
    threshold : float
        Confidence threshold for YOLO detections.

    Returns
    -------
    int or None
        The refined rating. YOLO can only raise or lower within reason:
        * exposed genitalia/breasts/buttocks -> at least 7, genitalia -> 10
        * only covered parts -> never higher than 7
        * no body parts detected -> unchanged
    """
    model = get_yolo_model()
    if model is None or rating is None:
        return rating

    try:
        results = model(img, verbose=False)
    except Exception:
        return rating

    detections = []
    for box in results[0].boxes:
        cls_name = model.names[int(box.cls)]
        conf = float(box.conf)
        if conf >= threshold:
            detections.append(cls_name)

    exposed_genitalia = bool(YOLO_EXPOSED_GENITALIA & set(detections))
    exposed_breasts = bool(YOLO_EXPOSED_BREASTS & set(detections))
    exposed_buttocks = bool(YOLO_EXPOSED_BUTTOCKS & set(detections))
    covered_breasts = bool(YOLO_COVERED_BREASTS & set(detections))
    covered_genitalia = bool(YOLO_COVERED_GENITALIA & set(detections))

    # Exposed genitalia is always explicit (10).
    if exposed_genitalia:
        return 10

    # Exposed breasts or buttocks = at least risque (7).
    if exposed_breasts or exposed_buttocks:
        if rating is None or rating < 7:
            return 7
        return rating

    # Only covered body parts: cannot be explicit (10). Cap at 7.
    if covered_breasts or covered_genitalia or YOLO_COVERED_BUTTOCKS & set(detections):
        if rating is not None and rating > 7:
            return 7

    # No explicit body parts found: if scanner said 10 but YOLO sees only
    # covered parts, downgrade to 7 (middle tier).
    if rating == 10 and not (exposed_breasts or exposed_buttocks or exposed_genitalia):
        return 7

    return rating


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def preprocess_image(img):
    """Resizes to 224x224 and returns NCHW float32 array in [0, 1].

    Generic preprocessing, kept for backward compatibility. The bundled
    models use their own pipelines below (preprocess_nsfw/preprocess_gender).
    """
    img = img.convert("RGB").resize((224, 224))
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = np.transpose(img_array, (2, 0, 1))  # HWC -> CHW
    img_array = np.expand_dims(img_array, axis=0)  # -> NCHW
    return img_array


# Dataset mean (BGR order) shared by the Yahoo NSFW and Levi gender models.
MEAN_BGR = np.array([104.0, 117.0, 123.0], dtype=np.float32)


def preprocess_nsfw(img):
    """Yahoo open_nsfw pipeline: 256px resize, 224px center crop, BGR-mean.

    Returns NHWC float32 batch of shape (1, 224, 224, 3). The reference
    implementation applies a JPEG round-trip first; it is skipped here
    (shifts scores by <1e-2, deterministic and faster).
    """
    img = img.convert("RGB").resize((256, 256), resample=Image.BILINEAR)
    img = img.crop((16, 16, 240, 240))
    arr = np.asarray(img, dtype=np.float32)[:, :, ::-1] - MEAN_BGR
    return np.ascontiguousarray(arr[np.newaxis, ...], dtype=np.float32)


def preprocess_gender(img):
    """FairFace ViT gender pipeline: 224px resize, /255, mean/std 0.5.

    Returns NCHW float32 batch of shape (1, 3, 224, 224) in RGB order.

    --- OLD: Levi/Adience gender pipeline (commented out, keep for reference)
    img = img.convert("RGB").resize((224, 224), resample=Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) - MEAN_BGR
    arr = np.transpose(arr, (2, 0, 1))[np.newaxis, ...]
    return np.ascontiguousarray(arr, dtype=np.float32)
    ---
    """
    img = img.convert("RGB").resize((224, 224), resample=Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    mean_arr = np.asarray([0.5, 0.5, 0.5], dtype=np.float32)
    std_arr = np.asarray([0.5, 0.5, 0.5], dtype=np.float32)
    arr = (arr - mean_arr) / std_arr
    arr = np.transpose(arr, (2, 0, 1))[np.newaxis, ...]
    return np.ascontiguousarray(arr, dtype=np.float32)


# The joint open_nsfw model (299x299, five classes).
JOINT_INPUT_SIZE = (299, 299)
STUDENT_INPUT_SIZE = (224, 224)
# The student was trained with the ImageNet normalisation (a ViT/MobileNet
# recipe), the teachers with theirs.
STUDENT_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STUDENT_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_nsfw_student(img):
    """The distilled student's pipeline: 224px, ImageNet norm, NCHW."""
    img = img.convert("RGB").resize(STUDENT_INPUT_SIZE, resample=Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = ((arr - STUDENT_MEAN) / STUDENT_STD).transpose(2, 0, 1)
    return np.ascontiguousarray(arr[np.newaxis, ...], dtype=np.float32)


def preprocess_nsfw_joint(img):
    """open_nsfw joint pipeline: 299px resize, values in [0, 1], NHWC.

    Returns NHWC float32 batch of shape (1, 299, 299, 3).  The joint
    model was trained without a mean shift (unlike the Yahoo 224 model),
    so the two pipelines are not interchangeable.
    """
    img = img.convert("RGB").resize(JOINT_INPUT_SIZE, resample=Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.ascontiguousarray(arr[np.newaxis, ...], dtype=np.float32)


def classify_pil_image(img, session):
    """The five class probabilities of the joint model (or the student).

    Returns {"drawings": .., "hentai": .., "neutral": .., "porn": ..,
    "sexy": ..} or None when the session is not the joint model (or the
    model is unavailable).
    """
    if session is None:
        return None
    if getattr(session, "_nsfw_kind", "binary") not in ("joint", "student"):
        return None
    try:
        input_name = session.get_inputs()[0].name
        output_names = [o.name for o in session.get_outputs()]
        prep = (preprocess_nsfw_student(img)
                if getattr(session, "_nsfw_kind", "") == "student"
                else preprocess_nsfw_joint(img))
        outputs = session.run(output_names, {input_name: prep})
        flat = np.asarray(outputs[0]).reshape(-1)
    except Exception:
        return None
    if flat.size < len(NSFW_CLASSES):
        return None
    return {name: float(flat[index]) for index, name in enumerate(NSFW_CLASSES)}


def student_nsfw_score(img, session):
    """The distilled student's "nsfw" score (its second head).

    The student replaces both teachers in one file: the five classes and
    this calibrated score.  Returns None when the session is not a student.
    """
    if session is None:
        return None
    if getattr(session, "_nsfw_kind", "binary") != "student":
        return None
    try:
        input_name = session.get_inputs()[0].name
        output_names = [o.name for o in session.get_outputs()]
        outputs = session.run(output_names,
                              {input_name: preprocess_nsfw_student(img)})
        return float(np.asarray(outputs[1]).reshape(-1)[0])
    except Exception:
        return None


# Gender output orders: FairFace ViT is [female, male]; Levi/Adience is
# [male, female]. A loaded session records which model it uses.
GENDER_LABELS = ("male", "female")


def _run_gender_both(session, img_array):
    """Runs the gender model, returning (female_score, male_score).

    Uses the output order recorded by load_gender_model. Unannotated
    sessions retain the legacy Levi/Adience order.
    """
    if session is None:
        return None, None
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    prediction = session.run([output_name], {input_name: img_array})[0]
    arr = np.asarray(prediction)
    flat = arr.reshape(-1).astype(np.float64)
    if flat.size < 2:
        return None, None
    # Apply softmax to convert logits to probabilities
    flat = flat - flat.max()
    exp = np.exp(flat)
    flat = exp / exp.sum()
    labels = getattr(session, "_nsfw_gender_output_order", GENDER_LABELS)
    female = float(flat[labels.index("female")])
    male = float(flat[labels.index("male")])
    return female, male


def _gender_label(female_score, male_score=None):
    """Maps female/male probabilities to 'female' / 'male' / 'unknown'.

    With both scores, uses argmax. A lone female probability is interpreted
    with the existing threshold for face-crop reporting.
    """
    if female_score is None:
        return "unknown"
    if male_score is not None:
        return "female" if female_score > male_score else "male"
    return "female" if female_score > FEMALE_THRESHOLD else "male"


def _run_session(session, img_array):
    """Runs one ONNX session, returning the positive-class score or None."""
    if session is None:
        return None
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    prediction = session.run([output_name], {input_name: img_array})[0]
    return _extract_positive_score(prediction)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Face detection (UltraFace-320) - an optional extra, deprecated as a default
# ---------------------------------------------------------------------------
#
# The detector, the box decode and the face crop live in their own module
# (nsfw_scanner.faces) so they can be imported - or dropped - on their own.
# The names are re-exported here so the public API does not change.

from .faces import (  # noqa: E402
    FACE_INPUT_SIZE,
    FACE_MIN_SIZE,
    FACE_THRESHOLD,
    detect_faces,
    get_face_session,
    load_face_model,
    preprocess_face,
)
from .faces import _square_box  # noqa: E402  (the face crop helper)


def _extract_positive_score(prediction, positive_index=1):
    """Pulls a positive-class probability out of common ONNX output shapes.

    Handles (1, 2), (2,), (1, 1)/scalar (sigmoid), and dict outputs.
    """
    if isinstance(prediction, dict):
        prediction = next(iter(prediction.values()))
    arr = np.asarray(prediction)
    if arr.size == 1:
        return float(arr.flat[0])
    flat = arr.reshape(-1)
    if positive_index < flat.size:
        return float(flat[positive_index])
    return float(flat[-1])


def _run_session(session, img_array):
    """Runs one ONNX session, returning the positive-class score or None."""
    if session is None:
        return None
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    prediction = session.run([output_name], {input_name: img_array})[0]
    return _extract_positive_score(prediction)


def _keyword_pattern(word):
    """Regex for one keyword: word boundaries, plus optional Hebrew prefixes.

    Hebrew attaches prepositions/articles to each word (בסקס, בית הספר),
    so every Hebrew token allows up to two prefix letters.
    """
    parts = []
    for token in word.split(" "):
        if re.search(r"[\u0590-\u05FF]", token):
            parts.append(r"[" + HEBREW_PREFIXES + r"]{0,2}" + re.escape(token))
        else:
            parts.append(re.escape(token))
    return r"\b" + r"\s+".join(parts) + r"\b"


def _contains_keyword(haystack, keywords):
    """Word-boundary match so 'sex' doesn't fire on 'Essex'/'sextant'
    (and 'סקס' doesn't fire on 'סקסופון'/saxophone)."""
    for word in keywords:
        if re.search(_keyword_pattern(word), haystack):
            return True
    return False


def _find_keyword_hits(haystack, keywords):
    """Returns the actual matched strings (deduped, keyword-list order).

    Unlike _contains_keyword, this shows WHAT matched -- e.g. the surface
    form 'הסקס' for keyword 'סקס'. Used for debug output.
    """
    hits = []
    seen = set()
    for word in keywords:
        for match in re.finditer(_keyword_pattern(word), haystack):
            surface = match.group(0)
            if surface not in seen:
                seen.add(surface)
                hits.append(surface)
    return hits


def _mask_safe_words(text, mask):
    """Blanks out the look-alike words before substring matching.

    "analysis" must not fire on "anal", "essex" not on "sex".  The words
    are replaced by spaces so no new words are created.
    """
    for word in sorted(mask, key=len, reverse=True):
        if word and word in text:
            text = text.replace(word, " " * len(word))
    return text


def find_sensuality_hits(text, lexicon=None):
    """All sensuality matches in a text: word-boundary *and* substring.

    Word-boundary terms behave like the classic keyword list ("sex" never
    fires on "Essex").  Substring terms are the long, unambiguous ones
    ("pornhub", "onlyfans") and are matched after the safe-mask words were
    blanked out.

    Returns the matched surface forms, in lexicon order.
    """
    if not text:
        return []
    text = text.lower()
    lexicon = lexicon or sensuality_lexicon()
    hits = _find_keyword_hits(text, lexicon.get("terms", []))
    seen = {h.lower() for h in hits}
    substrings = lexicon.get("substring_terms", [])
    mask = lexicon.get("safe_mask", [])
    if substrings:
        masked = _mask_safe_words(text, mask)
        for term in substrings:
            if term in seen:
                continue
            if term in masked:
                hits.append(term)
                seen.add(term.lower())
    return hits


def _contains_sensuality(text, lexicon=None):
    """True when the text holds a sensuality term."""
    return bool(find_sensuality_hits(text, lexicon))


def format_timecode(seconds):
    """Seconds -> 'mm:ss.ss' (used in the reasons)."""
    if seconds is None:
        return "?"
    try:
        minutes, rest = divmod(float(seconds), 60)
    except (TypeError, ValueError):
        return "?"
    return "%02d:%05.2f" % (int(minutes), rest)


def build_reason(rating, stage, **facts):
    """Builds the verdict explanation.

    Returns ``(reason, evidence)``:

    * ``reason``   - the sentence an admin reads ("why is this blocked?")
    * ``evidence`` - the minimal machine-readable facts behind it (a small
      dict; no blobs, no page dumps)

    The reason is stored with the block (admin only) - the public API
    never returns it.
    """
    phrase = RATING_PHRASES.get(rating, "unrated")
    evidence = {"stage": stage}

    def score_of(value):
        try:
            return round(float(value), 3)
        except (TypeError, ValueError):
            return None

    if stage == "text-keywords":
        hits = list(facts.get("hits") or [])[:5]
        evidence["hits"] = hits
        detail = " / ".join('"%s"' % hit for hit in hits) or "a sensuality term"
        reason = "%s (%s): %s matched in the page text" % (phrase, rating, detail)
    elif stage == "image":
        score = score_of(facts.get("score"))
        count = int(facts.get("count") or 0)
        total = int(facts.get("total") or 0)
        evidence["score"] = score
        evidence["images"] = count
        reason = "%s (%s): the image model scored %s on %d of %d images" % (
            phrase, rating, "%.2f" % score if score is not None else "n/a",
            count, total)
    elif stage == "gender":
        score = score_of(facts.get("score"))
        evidence["score"] = score
        reason = "%s (%s): an image crop was classified female (score %s)" % (
            phrase, rating, "%.2f" % score if score is not None else "n/a")
    elif stage == "modesty-url-keyword":
        hits = list(facts.get("hits") or [])[:5]
        evidence["hits"] = hits
        reason = "%s (%s): the URL contains %s" % (
            phrase, rating, " / ".join('"%s"' % hit for hit in hits) or "a term")
    elif stage == "dns-badwords":
        matches = list(facts.get("matches") or [])[:5]
        evidence["matches"] = matches
        reason = "%s (%s): the hostname matched the badword list (%s)" % (
            phrase, rating, ", ".join(matches) or "a known term")
    elif stage == "subtitle":
        line = facts.get("line")
        timecode = facts.get("timecode")
        keyword = facts.get("keyword")
        text = (facts.get("text") or "")[:80]
        evidence["line"] = line
        evidence["tc"] = score_of(timecode)
        evidence["keyword"] = keyword
        reason = "%s (%s): subtitle line %s at %s says \"%s\"" % (
            phrase, rating, line, format_timecode(timecode), text)
    elif stage == "frame":
        score = score_of(facts.get("score"))
        index = facts.get("index")
        timecode = facts.get("timecode")
        evidence["score"] = score
        evidence["frame"] = index
        evidence["tc"] = score_of(timecode)
        reason = "%s (%s): the image model scored %s on frame %s at %s" % (
            phrase, rating, "%.2f" % score if score is not None else "n/a",
            index, format_timecode(timecode))
    elif stage in ("class", "frame-class"):
        # The joint model does not score "nsfw", it names a class.
        name = facts.get("class")
        score = score_of(facts.get("score"))
        binary = score_of(facts.get("binary"))
        classes = facts.get("classes")
        evidence["class"] = name
        evidence["score"] = score
        if binary is not None:
            evidence["binary"] = binary
        if classes:
            evidence["classes"] = {key: score_of(value)
                                   for key, value in classes.items()}
        if stage == "frame-class":
            index = facts.get("index")
            timecode = facts.get("timecode")
            evidence["frame"] = index
            evidence["tc"] = score_of(timecode)
            subject = "frame %s at %s" % (index, format_timecode(timecode))
        else:
            subject = "image"
        reason = "%s (%s): the model called the %s %s (%s%s)" % (
            phrase, rating, subject, name,
            "%.2f" % score if score is not None else "n/a",
            ", the second model agrees at %.2f" % binary
            if binary is not None and facts.get("binary_agrees")
            else (", the second model sees nothing there (%.2f)" % binary
                  if binary is not None else ""))
    elif stage == "modesty-no-signal":
        score = score_of(facts.get("score"))
        evidence["score"] = score
        reason = "%s (1): nothing reached the thresholds (highest score %s)" % (
            phrase, "%.2f" % score if score is not None else "n/a")
    elif stage == "modesty-not-assessed":
        evidence["reason"] = facts.get("reason") or "no usable image-model output"
        reason = "not assessed: %s" % evidence["reason"]
    else:
        reason = "%s (%s)" % (phrase, rating)
    return reason, evidence


def _thresholds_dict():
    return {"explicit": NSFW_THRESHOLD_EXPLICIT,
            "revealing": NSFW_THRESHOLD_REVEALING,
            "female": FEMALE_THRESHOLD}


def normalize_url(url):
    """Adds https:// scheme when missing."""
    url = (url or "").strip()
    if not url:
        return url
    if not re.match(r"^https?://", url, re.IGNORECASE):
        url = "https://" + url
    return url


def extract_image_urls(soup, base_url, max_images=DEFAULT_MAX_IMAGES):
    """Collects absolute http(s) image URLs from <img> tags (sequential order).

    Handles src, data-src (lazy loading) and srcset. Skips data:/blob: URIs.
    """
    urls = []
    seen = set()
    for img in soup.find_all("img"):
        candidates = []
        for attr in ("src", "data-src", "data-original"):
            value = img.get(attr)
            if value:
                candidates.append(value.strip())
        srcset = img.get("srcset")
        if srcset:  # take each URL part, ignore width descriptors
            for part in srcset.split(","):
                token = part.strip().split()
                if token:
                    candidates.append(token[0].strip())
        for candidate in candidates:
            if not candidate or candidate.startswith("data:") or candidate.startswith("blob:"):
                continue
            absolute = urljoin(base_url, candidate)
            parsed = urlparse(absolute)
            if parsed.scheme not in ("http", "https"):
                continue
            if absolute not in seen:
                seen.add(absolute)
                urls.append(absolute)
                if len(urls) >= max_images:
                    return urls
    return urls


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def analyze_pil_image(img, nsfw_session, gender_session, fallback_url=None,
                      details=False, face_session=None, debug=False,
                      synthetic_gate=False, face_detector=None,
                      binary_session=None, plugins=None,
                      high_accuracy=False):
    """Rates an already-opened PIL image: 10 / 6 / 5 / 1.

    With the joint (5-class) model the verdict comes from its strongest
    class: ``porn``/``hentai`` -> 10, ``sexy`` -> 6, ``drawings``/
    ``neutral`` -> nothing on their own (the gender model may add 5).
    With the old binary model the thresholds decide (10 / 7 / 5 / 1).

    Gender is only judged on a detected face crop (face_session); without
    faces the gender is "unknown" so logos/graphics can't rate 5. Pass
    face_session=None for legacy whole-image gender scoring.

    ``face_detector`` is the **optional external path**: any callable with
    the signature of :func:`nsfw_scanner.faces.detect_faces`
    (``(img, face_session) -> [(x1, y1, x2, y2, prob), ...]``).  The
    bundled one is used when it is not given, so an external
    implementation can be plugged in without touching the core.

    ``high_accuracy`` (default False): when True, runs YOLOv8 (`yolo_best.pt`)
    after the fast scanner to detect localized body parts (covered vs exposed
    breasts, belly, buttocks, genitalia). This resolves middle-tier
    over-protection: distinguishing rating 6 (revealing, covered skin) from
    rating 7 (risque, exposed breasts/belly) from rating 10 (explicit,
    exposed genitalia/breasts).  Costs ~570ms per image vs ~112ms for the
    fast path.  Only runs when the scanner has already produced a rating.

    With details=True returns {"rating", "nsfw_score", "female_score",
    "gender", "faces", "elapsed"} where
    gender is "female" / "male" / "unknown" (unknown = no gender model).
    Scores are None when the corresponding model is unavailable.
    The joint model also fills "classes" (the five probabilities) and
    "class" (the strongest one).
    With debug=True (implies details) a "debug" dict is added explaining
    the verdict: which stage decided, scores vs thresholds, face boxes,
    and URL-keyword hits. The models are plain classifiers -- they cannot
    describe content, so this is the closest to a description they give.

    ``synthetic_gate`` (single images only): when the image is clearly a
    render / a UI screenshot / a drawing, the picture model is not trusted
    and the rating becomes 2 ("synthetic - not rated, scan it yourself").
    The media scanner decides on the whole file instead and leaves this off.
    """
    if debug:
        details = True
    start = time.perf_counter()
    img = img.convert("RGB")

    nsfw_score = None
    classes = None
    best_class = None
    if nsfw_session is not None:
        classes = classify_pil_image(img, nsfw_session)
    binary_score = None
    if binary_session is not None:
        binary_score = _run_session(binary_session, preprocess_nsfw(img))
    if binary_score is None and nsfw_session is not None:
        # The distilled student carries the second opinion in its own
        # second head - one file replaces both teachers.
        binary_score = student_nsfw_score(img, nsfw_session)

    if classes is not None:
        # The joint model's "porn" class is a **skin hypothesis**, not an
        # "explicit" verdict: measured on real frames it fires at 0.48 on
        # a man singing with his neck exposed, at 0.92 on a *wedding* and
        # at 0.77 on a rabbi's white beard.  The binary model is the
        # calibrated "is this explicit?" signal (0.07 / 0.01 / 0.03 on
        # exactly those frames).  So:
        #   * 10 needs the binary model to agree (or the class to be
        #     "hentai" - an explicit drawing);
        #   * 7 needs both: a strong skin hypothesis *and* something the
        #     second model sees too;
        #   * a skin hypothesis alone means "people are here", nothing
        #     more - the rating falls through to the gender model (5) or
        #     to no unsafe signal detected (1).
        nsfw_score = classes["porn"] + classes["hentai"]
        best_class = max(classes, key=classes.get)
        best_prob = classes[best_class]
        if binary_score is None:
            # No second opinion available (the binary model is missing):
            # fall back to the conservative rule - a "porn" class is then
            # taken at face value.
            if (nsfw_score > NSFW_THRESHOLD_EXPLICIT
                    or (best_class in ("porn", "hentai")
                        and nsfw_score > NSFW_THRESHOLD_REVEALING)):
                rating, decided_by = 10, "modesty-class-%s" % best_class
            elif (best_class == "sexy"
                  and classes["sexy"] > NSFW_THRESHOLD_REVEALING):
                rating, decided_by = 6, "modesty-class-sexy"
            else:
                rating, decided_by = None, None
        elif binary_score > NSFW_THRESHOLD_EXPLICIT:
            rating, decided_by = 10, "modesty-explicit-binary"
        elif best_class == "hentai" and best_prob > NSFW_JOINT_STRONG:
            rating, decided_by = 10, "modesty-class-hentai"
        elif best_class in ("porn", "hentai"):
            if best_prob > NSFW_JOINT_STRONG:
                # A lot of skin. Three tiers based on binary model:
                # binary > 0.5 -> 10 (explicit)
                # binary 0.35-0.5 -> 7/8 (middle band: lingerie/risque)
                # binary < 0.35 -> 6 (revealing/skin present)
                if binary_score > NSFW_THRESHOLD_EXPLICIT:
                    rating, decided_by = 10, "modesty-explicit-binary-joint"
                elif binary_score > NSFW_THRESHOLD_MIDDLE:
                    rating, decided_by = 7, "modesty-middle-skin"
                else:
                    rating, decided_by = 6, "modesty-class-skin"
            else:
                rating, decided_by = None, None
        elif best_class == "sexy" and classes["sexy"] > NSFW_THRESHOLD_REVEALING:
            # The two levels of "sexy": a low one (6, romance) and a high
            # one (8, strongly suggestive).  The 8 is off by default.
            if (NSFW_SEXY_HIGH and classes["sexy"] > NSFW_SEXY_STRONG):
                rating, decided_by = NSFW_SEXY_HIGH, "modesty-class-sexy-high"
            else:
                rating, decided_by = 6, "modesty-class-sexy"
        else:
            rating, decided_by = None, None
    elif nsfw_session is not None:
        nsfw_score = _run_session(nsfw_session, preprocess_nsfw(img))
        if nsfw_score is not None:
            if nsfw_score > NSFW_THRESHOLD_EXPLICIT:
                rating, decided_by = 10, "modesty-explicit"
            elif nsfw_score > NSFW_THRESHOLD_REVEALING:
                rating, decided_by = 7, "modesty-revealing"
            else:
                rating, decided_by = None, None
        else:
            rating, decided_by = None, None
    else:
        rating, decided_by = None, None

    female_score = None
    male_score = None
    gender_label = "unknown"
    face_boxes = []
    gender_on = "skipped-no-model"
    if rating is None and gender_session is not None:
        # Gender always runs on 4-section quadrants (not face-crop only)
        gender_on = "quadrant-crop"
        best_female_crop = None
        best_male_crop = None
        for crop_img in crop_halves(img):
            try:
                f, m = _run_gender_both(
                    gender_session, preprocess_gender(crop_img))
                if f is not None and m is not None:
                    if f > m:
                        if best_female_crop is None or f > best_female_crop[0]:
                            best_female_crop = (f, m)
                    elif best_male_crop is None or m > best_male_crop[1]:
                        best_male_crop = (f, m)
            except Exception:
                pass
        selected_crop = best_female_crop or best_male_crop
        if selected_crop is not None:
            female_score, male_score = selected_crop
            gender_label = "female" if best_female_crop is not None else "male"
        if best_female_crop is not None:
            rating, decided_by = 5, "modesty-gender-female"

    url_hits = []
    if rating is None:
        # No models available (or scores below thresholds): check the URL itself.
        if (nsfw_score is None and female_score is None and fallback_url):
            url_hits = find_sensuality_hits(fallback_url.lower())
        if url_hits:
            rating, decided_by = 7, "modesty-url-keyword"
        elif nsfw_score is not None or classes is not None:
            rating, decided_by = 1, "modesty-no-signal"
        else:
            rating, decided_by = None, "modesty-not-assessed"

    plugin_evidence = []
    if plugins:
        # The plugins may raise the rating (censor), replace it (rate) or
        # only add to the evidence (report).
        from .plugins import merge_verdicts
        rating, plugin_evidence = merge_verdicts(rating, plugins, img,
                                                  "image")

    # YOLO high-accuracy refinement: localized body-part detection to resolve
    # middle-tier over-protection (revealing 6 vs risque 7 vs explicit 10).
    yolo_detections = None
    if high_accuracy and rating is not None:
        refined = _yolo_refine_rating(rating, img)
        if refined != rating:
            decided_by = decided_by + "+yolo-refined" if decided_by else "yolo-refined"
        rating = refined
        yolo_detections = refined

    # SKIPPED: synthetic gate disabled (face detector off by default)
    # if synthetic_gate:
    #     from .synthetic import is_synthetic_frame
    #     synthetic_stats = is_synthetic_frame(img, face_session=face_session)
    #     if synthetic_stats.get("synthetic"):
    #         rating, decided_by = 2, "synthetic"
    synthetic_stats = {"synthetic": False}
    synthetic_stats = None

    if details:
        def _r6(v):
            return None if v is None else round(float(v), 6)

        def _round_classes(c):
            if c is None:
                return None
            return {k: _r6(v) for k, v in c.items()}

        info = {"rating": rating, "nsfw_score": _r6(nsfw_score),
                "binary_score": _r6(binary_score),
                "female_score": _r6(female_score),
                "male_score": _r6(male_score),
                "gender": gender_label,
                "faces": len(face_boxes), "elapsed": time.perf_counter() - start}
        if plugin_evidence:
            info["plugins"] = plugin_evidence
        if high_accuracy:
            info["high_accuracy"] = True
            info["yolo_refined"] = yolo_detections
        if classes is not None:
            info["classes"] = _round_classes(classes)
            info["class"] = best_class
        if synthetic_stats is not None:
            info["synthetic"] = bool(synthetic_stats.get("synthetic"))
            info["synthetic_stats"] = {k: synthetic_stats[k] for k in
                                       ("noise", "flat") if k in synthetic_stats}
            if synthetic_stats.get("synthetic"):
                info["reason"] = ("synthetic content (2): render / screenshot / "
                                  "drawing - the image model is trained on "
                                  "photographs; scan it yourself")
                info["evidence"] = {"stage": "synthetic",
                                    "noise": synthetic_stats.get("noise"),
                                    "flat": synthetic_stats.get("flat"),
                                    "nsfw_score": _r6(nsfw_score)}
        if debug:
            info["debug"] = {"decided_by": decided_by,
                             "nsfw_score": _r6(nsfw_score),
                             "binary_score": _r6(binary_score),
                             "classes": _round_classes(classes),
                             "class": best_class,
                             "female_score": _r6(female_score),
                             "thresholds": _thresholds_dict(),
                             "gender_on": gender_on,
                             "face_boxes": face_boxes,
                             "url_hits": url_hits,
                             "high_accuracy": high_accuracy}
        return info
    return rating


# ============================================================================
# New: gender detection over ALL faces (alongside the largest-face one)
# ============================================================================

def gender_all_faces(img, gender_session, face_session=None, face_detector=None):
    """Runs the gender model on **every** detected face (not just the largest).

    This is an alternative to the largest-face-only gender scoring used
    inside :func:`analyze_pil_image`.  Use both to compare -- the existing
    path keeps the single biggest face, this one scores each face and
    aggregates.

    Returns
    -------
    dict
        {"faces": [{"box": [x1,y1,x2,y2], "prob": float,
                     "female_score": float or None, "gender": "female"/"male"/"unknown"},
                   ...],
         "best_female_score": float or None,   # highest female score across faces
         "any_female": bool,                   # True if any face > FEMALE_THRESHOLD
         "gender_label": "female"/"male"/"unknown",  # female if any face is female
         "gender_on": "face-crop"/"skipped-no-faces"/"skipped-no-model",
         "face_count": int}
        Returns all-None/unknown when the gender model is unavailable.
    """
    result = {"faces": [],
              "best_female_score": None,
              "any_female": False,
              "gender_label": "unknown",
              "gender_on": "skipped-no-model",
              "face_count": 0}
    if gender_session is None:
        return result

    img = img.convert("RGB")
    detector = face_detector or detect_faces
    try:
        boxes = detector(img, face_session)
    except Exception:
        boxes = []

    result["face_count"] = len(boxes)
    if not boxes:
        result["gender_on"] = "skipped-no-faces"
        return result

    result["gender_on"] = "face-crop"
    best = None
    for box in boxes:
        x1, y1, x2, y2, prob = box
        try:
            sx1, sy1, sx2, sy2, _p = _square_box(box, *img.size)
            female_score, male_score = _run_gender_both(
                gender_session,
                preprocess_gender(img.crop((sx1, sy1, sx2, sy2))),
            )
        except Exception:
            female_score, male_score = None, None
        label = _gender_label(female_score, male_score)
        result["faces"].append({"box": [x1, y1, x2, y2], "prob": float(prob),
                                "female_score": female_score,
                                "male_score": male_score, "gender": label})
        if female_score is not None:
            if best is None or female_score > best:
                best = female_score
            if label == "female":
                result["any_female"] = True

    result["best_female_score"] = best
    if result["any_female"]:
        result["gender_label"] = "female"
    elif best is not None:
        result["gender_label"] = "male"
    return result


# ---------------------------------------------------------------------------
# New: half-vertical / half-horizontal crops for low-quality images
# ---------------------------------------------------------------------------

# Low quality = either axis smaller than this pixel dimension.
# Configurable via NSFW_LOW_QUALITY_MAX_DIM environment variable (default 1024).
import os as _os
LOW_QUALITY_MAX_DIM = int(_os.environ.get("NSFW_LOW_QUALITY_MAX_DIM", "1024"))


def is_low_quality(img, max_dim=LOW_QUALITY_MAX_DIM):
    """True when the image is small enough that cropping may help focus the signal."""
    width, height = img.size
    return width < max_dim or height < max_dim


def crop_halves(img):
    """Crop an image into four quadrants: half-vertical + half-horizontal.

    Returns [top-left, top-right, bottom-left, bottom-right].
    """
    width, height = img.size
    half_w, half_h = width // 2, height // 2
    return [
        img.crop((0, 0, half_w, half_h)),        # top-left
        img.crop((half_w, 0, width, half_h)),    # top-right
        img.crop((0, half_h, half_w, height)),   # bottom-left
        img.crop((half_w, half_h, width, height)),  # bottom-right
    ]


# ============================================================================
# New: full evaluation - every model + every plugin, no conditions
# ============================================================================

def evaluate_image(img, nsfw_session=None, gender_session=None,
                   face_session=None, binary_session=None,
                   face_detector=None, plugins=None, fallback_url=None,
                   crops=False, low_quality_max_dim=None,
                   early_stop=False):
    """Runs every model and plugin without any threshold conditions.

    Unlike :func:`analyze_pil_image`, this function does **not** decide a
    rating, does **not** apply thresholds, and does **not** short-circuit.
    It runs all available models (joint/student NSFW classes, binary,
    gender on all faces, whole-image gender) and all plugins, then returns
    every raw result for offline evaluation / comparison.

    ``crops`` (default off): when the image is low quality (smaller than
    ``low_quality_max_dim`` on either axis), also runs the full evaluation
    on each of the four half-vertical / half-horizontal quadrant crops.
    This helps when a face or person is small in the frame and gets diluted.

    ``early_stop`` (default False): when ``crops`` is on, stop after the
    first crop that detects a female (``any_female`` or gender_label ==
    ``"female"``).  Use True for a quick "is there any female?" answer;
    use False to get facts for every crop.

    Returns
    -------
    dict
        {"nsfw_classes": {...} or None, "nsfw_score": float or None,
         "best_class": str or None, "binary_score": float or None,
         "gender": {...result of gender_all_faces...},
         "whole_image_gender": {"female_score": float or None,
                                "male_score": float or None, "gender": str},
         "plugins": [...], "url_hits": [...], "elapsed": float,
         "crops": [per-crop evaluate_image dicts] or None,
         "crop_count": int, "early_stopped": bool}
    """
    if low_quality_max_dim is None:
        low_quality_max_dim = LOW_QUALITY_MAX_DIM
    start = time.perf_counter()
    img = img.convert("RGB")

    # --- NSFW joint / student classes ---
    classes = None
    best_class = None
    if nsfw_session is not None:
        classes = classify_pil_image(img, nsfw_session)
    if classes is not None:
        best_class = max(classes, key=classes.get)
    nsfw_score = None
    if classes is not None:
        nsfw_score = classes.get("porn", 0.0) + classes.get("hentai", 0.0)

    # --- binary (second opinion) ---
    binary_score = None
    if binary_session is not None:
        binary_score = _run_session(binary_session, preprocess_nsfw(img))
    if binary_score is None and nsfw_session is not None:
        binary_score = student_nsfw_score(img, nsfw_session)

    # --- gender on ALL faces ---
    gender_all = gender_all_faces(img, gender_session,
                                  face_session=face_session,
                                  face_detector=face_detector)

    # --- whole-image gender (legacy path) ---
    whole_female = None
    whole_male = None
    if gender_session is not None:
        try:
            whole_female, whole_male = _run_gender_both(
                gender_session, preprocess_gender(img))
        except Exception:
            whole_female, whole_male = None, None

    # --- plugins (no rating decision, just evidence) ---
    plugin_evidence = []
    if plugins:
        from .plugins import merge_verdicts
        _dummy_rating, plugin_evidence = merge_verdicts(None, plugins, img, "image")

    # --- URL keyword hits ---
    url_hits = []
    if fallback_url:
        url_hits = find_sensuality_hits(fallback_url.lower())

    def _r6(v):
        return None if v is None else round(float(v), 6)

    result = {"nsfw_classes":
              {k: _r6(v) for k, v in classes.items()} if classes else None,
              "nsfw_score": _r6(nsfw_score),
              "best_class": best_class,
              "binary_score": _r6(binary_score),
              "gender": gender_all,
              "whole_image_gender": {"female_score": _r6(whole_female),
                                     "male_score": _r6(whole_male),
                                     "gender": _gender_label(whole_female, whole_male)},
              "plugins": plugin_evidence,
              "url_hits": url_hits,
              "elapsed": _r6(time.perf_counter() - start)}

    # --- half-vertical / half-horizontal crops for low-quality images ---
    early_stopped = False
    if crops and is_low_quality(img, low_quality_max_dim):
        crop_results = []
        for index, crop_img in enumerate(crop_halves(img)):
            crop_eval = evaluate_image(crop_img, nsfw_session=nsfw_session,
                                       gender_session=gender_session,
                                       face_session=face_session,
                                       binary_session=binary_session,
                                       face_detector=face_detector,
                                       plugins=plugins,
                                       fallback_url=fallback_url,
                                       crops=False,  # no recursion
                                       low_quality_max_dim=low_quality_max_dim,
                                       early_stop=False)  # crops never recurse
            crop_eval["crop"] = index
            crop_results.append(crop_eval)
            # early stop: first crop that detects a female
            if early_stop:
                cg = crop_eval.get("gender", {})
                if cg.get("any_female") or cg.get("gender_label") == "female":
                    early_stopped = True
                    break
        result["crops"] = crop_results
        result["crop_count"] = len(crop_results)
        result["early_stopped"] = early_stopped

    return result


def _looks_like_svg(url, content_type, content):
    """Detects SVG payloads PIL cannot decode (extension, type, or magic)."""
    if "svg" in (content_type or "").lower():
        return True
    if (url or "").lower().split("?", 1)[0].split("#", 1)[0].endswith(".svg"):
        return True
    head = content[:200].lstrip().lower() if content else b""
    return b"<svg" in head


def _plain_image_detail(rating, debug=False, reason="modesty-not-assessed"):
    info = {"rating": rating, "nsfw_score": None, "female_score": None,
            "gender": "unknown", "faces": 0, "elapsed": 0.0}
    if rating is None:
        info["reason"] = reason
    if debug:
        info["debug"] = {"decided_by": reason, "nsfw_score": None,
                         "female_score": None, "thresholds": _thresholds_dict(),
                         "gender_on": "skipped", "face_boxes": [], "url_hits": []}
    return info


def _timed(info, start):
    """Stamps a details dict with seconds elapsed since start."""
    info["elapsed"] = time.perf_counter() - start
    return info


def _badwords_hit(target, debug=False):
    """Returns a scan-result dict when the host name is obviously adult.

    DNS/name-only pre-check: avoids loading the ONNX models or fetching the
    page at all. Rating 10 = strong term, 7 = suggestive only, None = clear.

    Local files are checked by their *file name* ("pornhub_clip.mp4"),
    not by the whole path.
    """
    text = str(target or "")
    if "://" not in text and (
            os.path.sep in text or
            (os.path.altsep and os.path.altsep in text)):
        text = os.path.basename(text)
    info = domain_badwords(text)
    if info["rating"] is None:
        return None
    return info


def analyze_image(url, nsfw_session, gender_session, timeout=DEFAULT_TIMEOUT,
                  details=False, face_session=None, debug=False, user_agent=None,
                  synthetic_gate=False):
    """Rates a single image URL; returns None when it could not be assessed."""
    if debug:
        details = True
    start = time.perf_counter()
    if not url or url.startswith("data:") or url.startswith("blob:"):
        return _timed(_plain_image_detail(None, debug), start) if details else None
    try:
        response = requests.get(url, timeout=timeout, headers=_headers(user_agent))
        if response.status_code != 200 or not response.content:
            return _timed(_plain_image_detail(None, debug), start) if details else None

        content_type = response.headers.get("Content-Type", "")
        if _looks_like_svg(url, content_type, response.content):
            # PIL cannot decode vector SVGs; labeled distinctly so this
            # blind spot doesn't hide inside a generic unassessed result.
            reason = "unsupported-svg"
            return _timed(_plain_image_detail(None, debug, reason), start) if details else None

        img = Image.open(io.BytesIO(response.content)).convert("RGB")
        result = analyze_pil_image(img, nsfw_session, gender_session,
                                   fallback_url=url, details=details,
                                   face_session=face_session, debug=debug,
                                   synthetic_gate=synthetic_gate,
                                   binary_session=get_binary_session())
        if details:
            # Total wall time including download (overwrites inference-only).
            result["elapsed"] = time.perf_counter() - start
        return result
    except Exception as exc:
        reason = "image scan failed: %s" % exc
        return _timed(_plain_image_detail(None, debug, reason), start) if details else None


def _tile_crops(img, mode="3x3"):
    """Sub-crops of a frame.

    A person in a wide shot is only a small part of the image, and the
    photo model dilutes the signal over the rest of the frame (0.03 for
    the whole frame vs 0.43 for the crop that holds the person).

    ``mode``: ``"3x3"`` (nine tiles - the default for media), ``"2x2"``
    (the four quadrants plus the centre half - cheaper) or ``"off"``.
    """
    width, height = img.size
    if mode == "off":
        return []
    if mode == "2x2":
        half_w, half_h = width // 2, height // 2
        return [img.crop((0, 0, half_w, half_h)),
                img.crop((half_w, 0, width, half_h)),
                img.crop((0, half_h, half_w, height)),
                img.crop((half_w, half_h, width, height)),
                img.crop((width // 4, height // 4, 3 * width // 4, 3 * height // 4))]
    cols = rows = 3
    tile_w, tile_h = width / cols, height / rows
    crops = []
    for row in range(rows):
        for col in range(cols):
            x0 = int(max(0, col * tile_w))
            x1 = int(min(width, x0 + tile_w))
            y0 = int(max(0, row * tile_h))
            y1 = int(min(height, y0 + tile_h))
            if x1 - x0 > 32 and y1 - y0 > 32:
                crops.append(img.crop((x0, y0, x1, y1)))
    return crops


def analyze_pil_image_tiles(img, nsfw_session, gender_session,
                            fallback_url=None, details=False,
                            face_session=None, debug=False,
                            synthetic_gate=False, tiles="off",
                            face_detector=None, binary_session=None,
                            plugins=None, high_accuracy=False):
    """Like :func:`analyze_pil_image`, but also scores sub-crops.

    The best crop decides: a person who fills a tenth of the frame would
    otherwise be rated as if the frame were empty.  The gender model is
    only run on the full frame (the crops are for the NSFW signal).

    OFF by default on purpose: a tight crop of a face or a hand reads as
    "porn 0.97" to both models, so the tiles manufacture explicit
    verdicts out of innocent close-ups (a smiling man scored 0.93 that
    way).  Keep it for the "hidden message" hunt, where a false positive
    costs nothing.
    """
    if debug:
        details = True
    full = analyze_pil_image(img, nsfw_session, gender_session,
                             fallback_url=fallback_url, details=True,
                             face_session=face_session, debug=debug,
                             synthetic_gate=synthetic_gate,
                             face_detector=face_detector,
                             binary_session=binary_session,
                             plugins=plugins,
                             high_accuracy=high_accuracy)
    if not tiles or tiles == "off" or not isinstance(full, dict):
        return full
    best = full
    best_score = full.get("nsfw_score") or 0.0
    for index, crop in enumerate(_tile_crops(img, tiles)):
        detail = analyze_pil_image(crop, nsfw_session, None, details=True)
        score = detail.get("nsfw_score") or 0.0
        if score > best_score:
            best_score = score
            best = detail
            best["crop"] = index
    if not details:
        return best.get("rating")
    if best is not full:
        # keep the full frame's context (gender, faces) but the crop's score
        merged = dict(full)
        merged.update({"rating": best.get("rating"),
                       "nsfw_score": best.get("nsfw_score"),
                       "crop": best.get("crop")})
        return merged
    return best


def _carry_synthetic(out, info):
    """Copies the synthetic verdict (and its stats) from an image result."""
    if not isinstance(info, dict):
        return out
    if "synthetic" in info:
        out["synthetic"] = info["synthetic"]
    if "synthetic_stats" in info:
        out["synthetic_stats"] = info["synthetic_stats"]
    if info.get("synthetic") and info.get("reason"):
        out["reason"] = info["reason"]
    if info.get("synthetic") and info.get("evidence"):
        out["evidence"] = info["evidence"]
    return out


def _image_reason(info):
    """The reason + evidence for a single-image verdict."""
    rating = info.get("rating")
    score = info.get("nsfw_score")
    female = info.get("female_score")
    classes = info.get("classes")
    if rating is None:
        return build_reason(None, "modesty-not-assessed",
                            reason=info.get("reason"))
    if classes and info.get("class") in NSFW_CLASS_RATINGS:
        binary = info.get("binary_score")
        return build_reason(rating, "class", score=score, binary=binary,
                            binary_agrees=(binary is not None
                                           and binary > NSFW_THRESHOLD_REVEALING),
                            **{"class": info.get("class"), "classes": classes})
    if score is not None and score > NSFW_THRESHOLD_EXPLICIT:
        return build_reason(rating, "image", score=score, count=1, total=1)
    if score is not None and score > NSFW_THRESHOLD_REVEALING:
        return build_reason(rating, "image", score=score, count=1, total=1)
    if female is not None and female > FEMALE_THRESHOLD:
        return build_reason(rating, "gender", score=female)
    return build_reason(rating, "modesty-no-signal", score=score)


def analyze_image_file(path, nsfw_session, gender_session, details=False,
                       face_session=None, debug=False, synthetic_gate=False,
                       binary_session=None, high_accuracy=False):
    """Rates a local image file; returns None when it could not be assessed."""
    if debug:
        details = True
    start = time.perf_counter()
    try:
        with open(path, "rb") as handle:
            content = handle.read()
        if _looks_like_svg(path, "", content):
            reason = "unsupported-svg"
            return _timed(_plain_image_detail(None, debug, reason), start) if details else None
        img = Image.open(io.BytesIO(content)).convert("RGB")
        result = analyze_pil_image(img, nsfw_session, gender_session,
                                   fallback_url=path, details=details,
                                   face_session=face_session, debug=debug,
                                   synthetic_gate=synthetic_gate,
                                   binary_session=binary_session
                                   if binary_session is not None
                                   else get_binary_session(),
                                   high_accuracy=high_accuracy)
        if details:
            result["elapsed"] = time.perf_counter() - start
        return result
    except Exception as exc:
        reason = "image scan failed: %s" % exc
        return _timed(_plain_image_detail(None, debug, reason), start) if details else None


def _rate_parsed_page(soup, page_url, nsfw_session, gender_session,
                      max_images=DEFAULT_MAX_IMAGES, timeout=DEFAULT_TIMEOUT,
                      details=False, face_session=None, debug=False,
                      user_agent=None):
    """Text/keyword rating plus sequential image scan over parsed HTML.

    Returns (rating, image_details, text_hits, facts); image_details is a
    per-image list of {"url", "rating", "nsfw_score", "female_score",
    "gender", "faces"} when details=True, else []. text_hits are the
    matched keyword surface forms.  ``facts`` describes the deciding
    stage (used to build the human reason).
    """
    if debug:
        details = True
    text_content = soup.get_text(separator=" ").lower()
    meta_tags = " ".join(
        meta.get("content", "").lower() for meta in soup.find_all("meta")
    )
    combined = text_content + " " + meta_tags

    text_hits = find_sensuality_hits(combined)
    if text_hits:
        # Explicit content found in text/meta
        return 10, [], text_hits, {"stage": "text-keywords", "hits": text_hits}

    # NOTE: non-adult keywords deliberately do NOT force a no-signal rating --
    # they appear on adult pages too. Rating falls through to image scan.

    # Sequential image scan (low-perf: one at a time, bounded count).
    highest_image_rating = 1
    image_details = []
    over_count = 0
    female_count = 0
    best_score = None
    best_female = None
    unassessed_count = 0
    for img_url in extract_image_urls(soup, page_url, max_images=max_images):
        if details:
            info = analyze_image(img_url, nsfw_session, gender_session,
                                 timeout=timeout, details=True,
                                 face_session=face_session, debug=debug,
                                 user_agent=user_agent)
            info["url"] = img_url
            image_details.append(info)
            image_rating = info["rating"]
            score = info.get("nsfw_score")
            if score is not None and (best_score is None or score > best_score):
                best_score = score
            female = info.get("female_score")
            if female is not None and (best_female is None or
                                       female > best_female):
                best_female = female
        else:
            image_rating = analyze_image(img_url, nsfw_session, gender_session,
                                         timeout=timeout, user_agent=user_agent)
        if image_rating is None:
            unassessed_count += 1
        elif image_rating >= 7:
            over_count += 1
        elif image_rating == 5:
            female_count += 1
        if image_rating is not None and image_rating > highest_image_rating:
            highest_image_rating = image_rating
            if highest_image_rating >= 10:
                break  # can't get higher; stop early to save resources

    if over_count:
        facts = {"stage": "image", "score": best_score, "count": over_count,
                 "total": len(image_details) or over_count}
    elif female_count:
        facts = {"stage": "gender", "score": best_female}
    elif unassessed_count and highest_image_rating == 1:
        highest_image_rating = None
        facts = {"stage": "modesty-not-assessed",
                 "reason": "%d image(s) had no usable model result" % unassessed_count}
    elif highest_image_rating > 1:
        facts = {"stage": "image", "score": best_score,
                 "count": 1, "total": len(image_details) or 1}
    else:
        facts = {"stage": "modesty-no-signal", "score": best_score}
    return highest_image_rating, image_details, text_hits, facts


def _rate_website_impl(url, max_images=DEFAULT_MAX_IMAGES, timeout=DEFAULT_TIMEOUT,
                       details=False, debug=False, user_agent=None,
                       use_badwords=True):
    """
    Rates a website 1..10 based on adult content (sequential, low-memory).

    Returns an int rating, or an "Error: ..." string if the page itself
    cannot be fetched. With details=True returns a dict:
    {"url", "rating" (None on error), "error", "text_hit", "images" [...]}.
    With debug=True (implies details) a "debug" dict is added with the
    matched keyword surface forms ("text_hits") and the deciding stage.
    """
    if debug:
        details = True
    def _err(message):
        if details:
            return {"url": url, "rating": None, "error": message,
                    "text_hit": False, "images": []}
        return message

    # Name-only pre-check: obvious hosts never pay for a fetch/model load.
    bw = _badwords_hit(url, debug) if use_badwords else None
    if bw is not None:
        if details:
            info = {"url": url, "rating": bw["rating"], "error": None,
                    "text_hit": False, "images": [], "badwords": bw}
            if debug:
                info["debug"] = {"decided_by": "dns-badwords",
                                 "badwords": bw, "text_hits": []}
            return info
        return bw["rating"]

    try:
        page_url = normalize_url(url)
        if not page_url:
            return _err("Error: Empty URL")

        nsfw_session, gender_session = get_sessions()

        try:
            response = requests.get(page_url, timeout=timeout, headers=_headers(user_agent))
        except requests.RequestException as exc:
            return _err(f"Error: Unable to access website ({exc})")
        if response.status_code != 200:
            return _err("Error: Unable to access website")

        soup = BeautifulSoup(response.text, "html.parser")
        rating, image_details, text_hits, facts = _rate_parsed_page(
            soup, page_url, nsfw_session, gender_session,
            max_images=max_images, timeout=timeout, details=details,
            face_session=get_face_session(), debug=debug, user_agent=user_agent)
        if details:
            reason, evidence = build_reason(rating, facts.pop("stage"), **facts)
            info = {"url": page_url, "rating": rating, "error": None,
                    "text_hit": rating == 10 and not image_details,
                    "images": image_details,
                    "reason": reason, "evidence": evidence}
            if debug:
                info["debug"] = {
                    "decided_by": "text-keywords" if text_hits else (
                        "modesty-not-assessed" if rating is None else (
                            "modesty-max-image-rating" if image_details else "modesty-no-signal")),
                    "text_hits": text_hits}
            return info
        return rating

    except Exception as e:
        return _err(f"Error: {str(e)}")


def rate_website(url, max_images=DEFAULT_MAX_IMAGES, timeout=DEFAULT_TIMEOUT,
                 details=False, debug=False, user_agent=None,
                 use_badwords=True):
    """Public wrapper: same as _rate_website_impl, plus "elapsed" seconds."""
    if debug:
        details = True
    start = time.perf_counter()
    result = _rate_website_impl(url, max_images=max_images, timeout=timeout,
                                details=details, debug=debug, user_agent=user_agent,
                                use_badwords=use_badwords)
    if details and isinstance(result, dict):
        result["elapsed"] = time.perf_counter() - start
    return result


def pick_scanner(target):
    """Picks the proper scanner for a target.

    Returns one of:

    * ``'image-file'``  - an existing local image
    * ``'image-url'``   - a URL with an image extension
    * ``'media-file'``  - an existing local video/audio file
    * ``'media-url'``   - a URL with a media extension
    * ``'website'``     - anything else

    The *header* wins for local files: a ``.jpg`` that is really a video
    is detected as media, not as an image.
    """
    text = (target or "").strip()
    path = text[7:] if text.lower().startswith("file://") else text
    if path and os.path.isfile(path):
        from . import media as _media
        kind = _media.detect_kind(path)
        if kind["media_type"] in ("video", "audio"):
            return "media-file"
        return "image-file"
    lowered = text.lower().split("?", 1)[0].split("#", 1)[0]
    if lowered.startswith(("http://", "https://")):
        if lowered.endswith(IMAGE_EXTENSIONS):
            return "image-url"
        from . import media as _media
        if lowered.endswith(_media.VIDEO_EXTENSIONS + _media.AUDIO_EXTENSIONS):
            return "media-url"
    return "website"


def _version():
    """The package version (without importing the package twice)."""
    try:
        from . import __version__
        return __version__
    except ImportError:
        return "?"


def _file_sha256(path, chunk_size=1048576):
    """Hex sha256 of a local file, streamed.  None when unreadable.

    It is what the user reports when they send a scan result back to the
    site ("do we know these bytes?").
    """
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


def _known_result(target, hit):
    """The result of a match against what we already know."""
    rating = hit.get("rating")
    result = {"target": target, "scanner": "known-db", "rating": rating,
              "error": None, "matched": True,
              "match_source": hit.get("source") or "known-db",
              "reason": hit.get("reason"), "elapsed": 0.0}
    if rating is not None:
        result["phrase"] = rating_phrase(rating)
    else:
        result["phrase"] = "known"
    if hit.get("media_type"):
        result["media_type"] = hit["media_type"]
    return result


def rate(target, max_images=DEFAULT_MAX_IMAGES, timeout=DEFAULT_TIMEOUT,
         details=False, debug=False, user_agent=None, use_badwords=True,
         media_options=None, known=None, use_known=True):
    """Rates any target by routing it to the right scanner.

    A local file never goes through the website path: images go to the
    image scanner, video/audio to the media scanner (which needs ffmpeg),
    and only URLs are treated as pages.  Media targets must be local
    files - download a URL first.

    ``known`` is a :class:`nsfw_scanner.known.KnownTargets`; when it
    matches, the target is not scanned again - the stored verdict is
    returned instead (``matched: True``).  ``use_known=False`` (or the
    CLI's ``--ignore-db``) scans it anyway.
    """
    if known is not None and use_known:
        hit = known.lookup(target)
        if hit is not None and hit.get("known"):
            return _known_result(target, hit)
    kind = pick_scanner(target)
    if kind in ("media-file", "media-url"):
        if kind == "media-url":
            raise ValueError(
                "media URLs must be downloaded first: pass a local file to "
                "rate() (the webservice does that for you)")
        from . import media as _media
        options = dict(media_options or {})
        options.setdefault("details", details)
        options.setdefault("debug", debug)
        return _media.scan_media(target, **options)
    return scan(target, max_images=max_images, timeout=timeout,
                details=details, debug=debug, user_agent=user_agent,
                use_badwords=use_badwords, known=known, use_known=use_known)


def _scan_impl(target, max_images=DEFAULT_MAX_IMAGES, timeout=DEFAULT_TIMEOUT,
               details=False, debug=False, user_agent=None,
               use_badwords=True, media_options=None):
    """Scans any target (local image file, image URL, website URL).

    Picks the proper scanner via pick_scanner(). Extensionless image URLs
    are detected through the response Content-Type. Returns an int rating,
    or an "Error: ..." string if the target cannot be read/fetched.
    With details=True returns a dict with per-image gender info:
    {"target", "scanner", "rating" (None on error), "error", "text_hit",
     "nsfw_score", "female_score", "gender", "images" [...]}.
    With debug=True (implies details) a "debug" dict explains the verdict:
    keyword hits and the deciding stage (models can't caption content).
    user_agent overrides the global UA for this scan only.
    """
    target = (target or "").strip()
    if debug:
        details = True
    if not target:
        if details:
            return {"target": target, "scanner": "unknown", "rating": None,
                    "error": "Error: Empty target", "text_hit": False,
                    "nsfw_score": None, "female_score": None,
                    "gender": "unknown", "images": []}
        return "Error: Empty target"

    kind = pick_scanner(target)

    # Media files (video/audio) go to the media scanner, never through the
    # website path.  It needs ffmpeg and fails loudly when it is missing.
    if kind in ("media-file", "media-url"):
        if kind == "media-url":
            message = ("Error: media URLs must be downloaded first "
                       "(pass a local file)")
            if details:
                return {"target": target, "scanner": "media", "rating": None,
                        "error": message, "text_hit": False,
                        "nsfw_score": None, "female_score": None,
                        "gender": "unknown", "images": []}
            return message
        # A media file with an obvious hostname still needs no hardware.
        if use_badwords:
            bw = _badwords_hit(target, debug)
            if bw is not None:
                if details:
                    reason, evidence = build_reason(
                        bw["rating"], "dns-badwords",
                        matches=bw.get("matches") or bw.get("strong") or [])
                    info = {"target": target, "scanner": "dns-badwords",
                            "rating": bw["rating"], "error": None,
                            "text_hit": False, "nsfw_score": None,
                            "female_score": None, "gender": "unknown",
                            "faces": 0, "images": [], "badwords": bw,
                            "reason": reason, "evidence": evidence}
                    if debug:
                        info["debug"] = {"decided_by": "dns-badwords",
                                         "badwords": bw, "text_hits": []}
                    return info
                return bw["rating"]
        from . import media as _media
        try:
            options = dict(media_options or {})
            options.setdefault("details", details)
            options.setdefault("debug", debug)
            result = _media.scan_media(target, **options)
        except _media.MediaDependencyError as exc:
            if details:
                return {"target": target, "scanner": "media", "rating": None,
                        "error": "Error: %s" % exc, "text_hit": False,
                        "nsfw_score": None, "female_score": None,
                        "gender": "unknown", "images": []}
            return "Error: %s" % exc
        if details and isinstance(result, dict) and not result.get("sha256"):
            result["sha256"] = _file_sha256(target)
        return result

    # Name-only pre-check for URL-based targets: no fetch, no ONNX sessions.
    bw = _badwords_hit(target, debug) if (use_badwords and kind != "image-file") else None
    if bw is not None:
        if details:
            reason, evidence = build_reason(
                bw["rating"], "dns-badwords",
                matches=bw.get("matches") or bw.get("strong") or [])
            info = {"target": target, "scanner": "dns-badwords",
                    "rating": bw["rating"], "error": None,
                    "text_hit": False, "nsfw_score": None,
                    "female_score": None, "gender": "unknown", "faces": 0,
                    "images": [], "badwords": bw,
                    "reason": reason, "evidence": evidence}
            if debug:
                info["debug"] = {"decided_by": "dns-badwords",
                                 "badwords": bw, "text_hits": []}
            return info
        return bw["rating"]

    nsfw_session, gender_session = get_sessions()
    face_session = get_face_session()

    if kind == "image-file":
        path = target[7:] if target.lower().startswith("file://") else target
        info = analyze_image_file(path, nsfw_session, gender_session, details=True,
                                  face_session=face_session, debug=debug,
                                  synthetic_gate=True)
        if details:
            reason, evidence = _image_reason(info)
            out = {"target": target, "scanner": kind, "rating": info["rating"],
                   "error": None, "text_hit": False,
                   "nsfw_score": info["nsfw_score"],
                   "female_score": info["female_score"], "gender": info["gender"],
                   "faces": info["faces"], "debug": info.get("debug"),
                   "images": [dict(info, url=path)],
                   "reason": reason, "evidence": evidence,
                   "sha256": _file_sha256(path)}
            _carry_synthetic(out, info)
            return out
        return info["rating"]
        return info["rating"]

    if kind == "image-url":
        info = analyze_image(target, nsfw_session, gender_session,
                             timeout=timeout, details=True,
                             face_session=face_session, debug=debug,
                             user_agent=user_agent, synthetic_gate=True)
        if details:
            reason, evidence = _image_reason(info)
            return {"target": target, "scanner": kind, "rating": info["rating"],
                    "error": None, "text_hit": False,
                    "nsfw_score": info["nsfw_score"],
                    "female_score": info["female_score"], "gender": info["gender"],
                    "faces": info["faces"], "debug": info.get("debug"),
                    "images": [dict(info, url=target)],
                    "reason": reason, "evidence": evidence}
        return info["rating"]

    page_url = normalize_url(target)
    try:
        response = requests.get(page_url, timeout=timeout, headers=_headers(user_agent))
    except requests.RequestException as exc:
        message = f"Error: Unable to access website ({exc})"
        if details:
            return {"target": target, "scanner": kind, "rating": None,
                    "error": message, "text_hit": False, "nsfw_score": None,
                    "female_score": None, "gender": "unknown", "images": []}
        return message
    if response.status_code != 200:
        if details:
            return {"target": target, "scanner": kind, "rating": None,
                    "error": "Error: Unable to access website", "text_hit": False,
                    "nsfw_score": None, "female_score": None,
                    "gender": "unknown", "images": []}
        return "Error: Unable to access website"

    content_type = response.headers.get("Content-Type", "").split(";")[0].strip().lower()
    if content_type.startswith("image/"):
        try:
            img = Image.open(io.BytesIO(response.content)).convert("RGB")
        except Exception:
            reason = ("unsupported-svg" if _looks_like_svg(
                page_url, content_type, response.content)
                else "image response could not be decoded")
            if details:
                img0 = {"url": page_url, "rating": None, "nsfw_score": None,
                        "female_score": None, "gender": "unknown", "faces": 0,
                        "elapsed": 0.0, "reason": reason}
                top = {"target": target, "scanner": "image-url",
                       "rating": None, "error": None, "text_hit": False,
                       "nsfw_score": None, "female_score": None,
                       "gender": "unknown", "images": [img0],
                       "reason": "not assessed: %s" % reason}
                if debug:
                    dbg = {"decided_by": reason, "nsfw_score": None,
                           "female_score": None, "thresholds": _thresholds_dict(),
                           "gender_on": "skipped", "face_boxes": [], "url_hits": []}
                    top["debug"] = dbg
                    img0["debug"] = dbg
                return top
            return None
        info = analyze_pil_image(img, nsfw_session, gender_session,
                                 fallback_url=page_url, details=True,
                                 face_session=face_session, debug=debug,
                                 synthetic_gate=True,
                                 binary_session=get_binary_session())
        if details:
            reason, evidence = _image_reason(info)
            out = {"target": target, "scanner": "image-url",
                   "rating": info["rating"], "error": None, "text_hit": False,
                   "nsfw_score": info["nsfw_score"],
                   "female_score": info["female_score"], "gender": info["gender"],
                   "faces": info["faces"], "debug": info.get("debug"),
                   "images": [dict(info, url=page_url)],
                   "reason": reason, "evidence": evidence}
            _carry_synthetic(out, info)
            return out
        return info["rating"]

    soup = BeautifulSoup(response.text, "html.parser")
    rating, image_details, text_hits, facts = _rate_parsed_page(
        soup, page_url, nsfw_session, gender_session,
        max_images=max_images, timeout=timeout, details=details,
        face_session=face_session, debug=debug, user_agent=user_agent)
    if details:
        reason, evidence = build_reason(rating, facts.pop("stage"), **facts)
        info = {"target": target, "scanner": kind, "rating": rating,
                "error": None,
                "text_hit": rating == 10 and not image_details,
                "nsfw_score": None, "female_score": None, "gender": "unknown",
                "images": image_details, "reason": reason, "evidence": evidence}
        if debug:
            info["debug"] = {
                "decided_by": "text-keywords" if text_hits else (
                    "modesty-not-assessed" if rating is None else (
                        "modesty-max-image-rating" if image_details else "modesty-no-signal")),
                "text_hits": text_hits}
        return info
    return rating


def scan(target, max_images=DEFAULT_MAX_IMAGES, timeout=DEFAULT_TIMEOUT,
         details=False, debug=False, user_agent=None, use_badwords=True,
         media_options=None, known=None, use_known=True):
    """Public wrapper: same as _scan_impl, plus "elapsed" seconds.

    ``use_badwords`` (default True) enables the DNS/name-only pre-check:
    obvious adult hostnames return rating 10/7 without a fetch or model load.

    ``media_options`` is passed to the media scanner for video/audio files
    (fps, every_frame, max_frames, ...).

    ``known`` is a :class:`nsfw_scanner.known.KnownTargets`: when it
    matches, the stored verdict is returned instead of scanning again.
    """
    if known is not None and use_known:
        hit = known.lookup(target)
        if hit is not None and hit.get("known"):
            return _known_result(target, hit)
    if debug:
        details = True
    start = time.perf_counter()
    result = _scan_impl(target, max_images=max_images, timeout=timeout,
                        details=details, debug=debug, user_agent=user_agent,
                        use_badwords=use_badwords, media_options=media_options)
    if details and isinstance(result, dict):
        result["elapsed"] = time.perf_counter() - start
    return result


def _format_size(size_bytes):
    """A file size as a short string ("12.3 MB", "840 kB", None)."""
    try:
        size = float(size_bytes)
    except (TypeError, ValueError):
        return None
    if size <= 0:
        return None
    if size < 1024:
        return "%d B" % int(size)
    if size < 1024 * 1024:
        return "%.0f kB" % (size / 1024.0)
    if size < 1024 * 1024 * 1024:
        return "%.1f MB" % (size / 1048576.0)
    return "%.2f GB" % (size / 1073741824.0)


def progress_printer(enabled=True, stream=None, label=None, style="full"):
    """A progress callback that writes to stderr (stdout stays pipeable).

    Shows the percentage, the position in the file, the elapsed time, the
    ETA and the speed - both in frames/s and as a multiple of the
    file's own runtime, which is what makes the sampling rate legible.

    ``label`` is shown in the header - the benchmark uses it to say which
    model and which decoder the run is using ("joint · cpu").
    ``style`` is "full" (the CLI) or "compact" (one line every couple of
    seconds - the queue worker).
    """
    import sys

    if not enabled:
        return None
    out = stream or sys.stderr
    state = {"last": 0.0, "last_percent": -1.0, "tty": False}
    try:
        state["tty"] = out.isatty()
    except Exception:
        state["tty"] = False

    def report(info):
        stage = info.get("stage")
        now = time.time()

        def clear():
            """Wipes the redrawn progress line (a terminal only)."""
            if state["tty"]:
                out.write("\r" + " " * 120 + "\r")

        if stage == "probe":
            duration = info.get("duration")
            fps = info.get("fps")
            total = info.get("total")
            native = info.get("native_fps")
            size = _format_size(info.get("size"))
            if duration:
                clear()
                out.write("scanning %s%s — %s long%s, %s frames%s\n" % (
                    os.path.basename(str(info.get("target") or "")),
                    " [%s]" % label if label else "",
                    format_timecode(duration),
                    ", %s" % size if size else "",
                    total if total else "?",
                    " sampled of %s (%s of %s fps)" % (native, fps, native)
                    if native else ""))
                out.flush()
            return
        if stage == "done":
            clear()
            out.write("done: rating %s · %s problem area(s) · %.1fs\n" % (
                info.get("rating"), info.get("problems"),
                info.get("elapsed") or 0.0))
            out.write("\n")
            out.flush()
            return
        if stage != "frames":
            return
        percent = info.get("percent")
        done = info.get("done") or 0
        elapsed = info.get("elapsed") or 0.0
        if style == "compact":
            # one line every couple of seconds - the worker log
            if now - state["last"] < 2.0 and percent != 100:
                return
            state["last"] = now
            out.write("    %5.1f%%  %s/%s frames  @ %s  %s frames/s\n" % (
                percent or 0, done, info.get("total") or "?",
                info.get("timecode") or 0,
                round(done / elapsed, 1) if elapsed else 0))
            out.flush()
            return
        # a redrawn line on a terminal, a line every 5% elsewhere
        if state["tty"]:
            if now - state["last"] < 0.2 and percent != 100:
                return
        elif percent is not None and percent < state["last_percent"] + 5:
            return
        state["last"] = now
        state["last_percent"] = percent or 0
        rate = done / elapsed if elapsed else 0.0
        fps = info.get("fps") or 0
        realtime = (done / fps / elapsed) if (fps and elapsed) else 0.0
        clear()
        out.write("  frames %6s/%-6s %5s%%  @ %s  %6.1fs elapsed  ETA %s  "
                  "%.1f frames/s  %.1fx realtime%s" % (
                      done, info.get("total") or "?", percent or 0,
                      format_timecode(info.get("timecode")),
                      elapsed, ("%5.1fs" % info["eta"]) if info.get("eta") else "  -  ",
                      rate, realtime,
                      "  (every frame)" if info.get("every_frame") else ""))
        out.write("\r" if state["tty"] else "\n")
        out.flush()

    return report


def format_benchmark(result):
    """A human benchmark block for a media scan (the CLI's --benchmark).

    Shows where the time went, so a user can compare a CPU run with a
    GPU run (or the joint model with the binary one) on their own
    machine::

        nsfw-scan clip.mp4 --hwaccel none --benchmark
        nsfw-scan clip.mp4 --hwaccel cuda --benchmark
    """
    import sys

    if not isinstance(result, dict):
        return ""
    timings = result.get("timings") or {}
    if not timings:
        return ""
    frames = result.get("frame_count") or 0
    lines = []
    lines.append("benchmark - %s" % os.path.basename(str(result.get("target") or "")))
    lines.append("  decoder        : %s" % (result.get("hwaccel") or "none"))
    lines.append("  model          : %s" % (result.get("model") or "none"))
    lines.append("  frames         : %d @ %s fps%s" % (
        frames, result.get("fps") or ("every frame" if result.get("every_frame")
                                      else "?"),
        "  (%s long)" % format_timecode(result.get("duration_sec"))
        if result.get("duration_sec") else ""))
    lines.append("  total          : %.1f s" % (timings.get("total_sec") or 0.0))
    lines.append("  frame loop     : %.1f s   (decode + PNG encode + pipe)" % (
        timings.get("frame_loop_sec") or 0.0))
    lines.append("  model inference: %.1f s   (%s ms/frame)" % (
        timings.get("inference_sec") or 0.0,
        timings.get("ms_per_frame_inference")
        if timings.get("ms_per_frame_inference") is not None else "n/a"))
    if timings.get("subtitles_sec"):
        lines.append("  subtitles      : %.1f s" % timings["subtitles_sec"])
    lines.append("  throughput     : %s frames/s   %sx realtime" % (
        timings.get("frames_per_sec"), timings.get("realtime_x")))
    lines.append("  verdict        : rating %s   %s" % (
        result.get("rating"), result.get("reason") or ""))
    lines.append("  compare        : re-run with another --hwaccel / --model "
                 "and diff these numbers")
    return "\n".join(lines) + "\n"


def _benchmark_run(target, media_options, model, hwaccel, known, args):
    """One benchmark run: a physical scan (never the known-target cache)."""
    previous = set_model_mode(model)
    try:
        options = dict(media_options or {})
        options["hwaccel"] = hwaccel
        options["progress"] = progress_printer(
            not args.no_progress,
            label="%s · %s" % (model, "cpu" if hwaccel == "none"
                               else "gpu(%s)" % hwaccel))
        result = rate(target, max_images=args.max_images,
                      timeout=args.timeout, details=False, debug=False,
                      user_agent=args.user_agent,
                      use_badwords=not args.no_badwords,
                      media_options=options,
                      known=known, use_known=False)
    finally:
        set_model_mode(previous)
    return result


def format_benchmark_comparison(runs, title="benchmark"):
    """The comparison table for the benchmark runs (CLI --benchmark).

    ``runs`` is a list of {"decoder", "model", "result"} dicts.
    """
    lines = []
    first = (runs[0]["result"] if runs else {}) or {}
    name = os.path.basename(str(first.get("target") or ""))
    duration = first.get("duration_sec")
    try:
        size = _format_size(os.path.getsize(str(first.get("target"))))
    except OSError:
        size = None
    lines.append("%s - %s%s" % (
        title, name,
        " — %s long%s" % (format_timecode(duration),
                           ", %s" % size if size else "")
        if duration else ""))
    lines.append("  %d run(s): %s" % (
        len(runs), ", ".join("%s/%s" % (run["decoder"], run["model"])
                              for run in runs)))
    lines.append("  " + "-" * 86)
    lines.append("  %-18s %8s %8s %10s %10s %8s %9s %6s %5s" % (
        "run", "total", "loop", "inference", "ms/frame", "frames/s",
        "x realtime", "rating", "flags"))
    for run in runs:
        result = run["result"] or {}
        timings = result.get("timings") or {}
        actual = result.get("hwaccel") or run["decoder"]
        label = "%s / %s" % (run["decoder"], run["model"])
        if actual and actual != run["decoder"]:
            # "auto" may have fallen back to the software decoder - say so
            label = "%s / %s (%s)" % (run["decoder"], run["model"], actual)
        lines.append("  %-18s %7.1fs %7.1fs %9.1fs %10s %8s %9s %6s %5s" % (
            label,
            timings.get("total_sec") or 0.0,
            timings.get("frame_loop_sec") or 0.0,
            timings.get("inference_sec") or 0.0,
            timings.get("ms_per_frame_inference")
            if timings.get("ms_per_frame_inference") is not None else "n/a",
            timings.get("frames_per_sec") or "n/a",
            timings.get("realtime_x") or "n/a",
            result.get("rating"),
            result.get("problem_count")))
    lines.append("  " + "-" * 78)
    timed = [run for run in runs if (run["result"] or {}).get("timings")]
    if timed:
        fastest = min(timed, key=lambda run: (
            run["result"]["timings"].get("total_sec") or 0.0))
        lines.append("  fastest: %s / %s (%.1fs)" % (
            fastest["decoder"], fastest["model"],
            fastest["result"]["timings"].get("total_sec") or 0.0))
    ratings = {}
    for run in runs:
        ratings.setdefault((run["result"] or {}).get("rating"), []).append(
            "%s/%s" % (run["decoder"], run["model"]))
    if len(ratings) > 1:
        lines.append("  the verdicts differ:")
        for rating, labels in sorted(ratings.items(), key=lambda kv: str(kv[0])):
            lines.append("    rating %-4s %s" % (rating, ", ".join(labels)))
        lines.append("    (the binary model has no romance grade and its own "
                     "thresholds - a different verdict is expected)")
    else:
        lines.append("  every run agreed on the verdict (rating %s)" % (
            list(ratings)[0] if ratings else "?"))
    lines.append("  note: the hardware frames are not bit-identical to the "
                 "software ones,")
    lines.append("        so a borderline frame can flip its class.")
    return "\n".join(lines) + "\n"


def main(argv=None):
    """CLI entry point: python NSFW_url_scan_low_perf.py <target> [--max-images N]."""
    import argparse

    parser = argparse.ArgumentParser(description="Rate a website, image URL, or local file 1-10 for adult content.")
    parser.add_argument("--version", action="version",
                        version="nsfw-scan %s" % _version())
    parser.add_argument("target", nargs="?", help="Website URL, image URL, or local image file to scan (not needed with --list-plugins)")
    parser.add_argument("--max-images", type=int, default=DEFAULT_MAX_IMAGES,
                        help="Max images to analyze (default: %(default)s)")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                        help="Per-request timeout in seconds (default: %(default)s)")
    parser.add_argument("--details", action="store_true",
                        help="Print rating plus nsfw/female scores and gender")
    parser.add_argument("--debug", action="store_true",
                        help="Explain the verdict (keyword hits, deciding stage)")
    parser.add_argument("--user-agent",
                        help="'browser', 'scanner', or a custom User-Agent string")
    parser.add_argument("--no-badwords", action="store_true",
                        help="Skip the DNS/name-only badword pre-check")
    parser.add_argument("--fps", type=float, default=1.0,
                        help="Media: frames per second to sample "
                             "(default: %(default)s)")
    parser.add_argument("--every-frame", action="store_true",
                        help="Media: decode every frame (slow; for hunting "
                             "hidden messages)")
    parser.add_argument("--max-frames", type=int, default=2000,
                        help="Media: stop after this many frames "
                             "(default: %(default)s)")
    parser.add_argument("--db", default=None,
                        help="local product DB (nsfw.sqlite) to match "
                             "against before scanning")
    parser.add_argument("--hash-list", default=None,
                        help="text file with sha256 hashes (one per line) "
                             "to match against before scanning")
    parser.add_argument("--api-url", default=None,
                        help="hosted API to match against (paid; needs "
                             "--api-key)")
    parser.add_argument("--api-key", default=None,
                        help="API key for --api-url")
    parser.add_argument("--imdb-id", metavar="TTL", default=None,
                        help="tag the scan and the optional webservice "
                             "submission with an IMDb ID (e.g. tt0111161)")
    parser.add_argument("--submit-to-webservice", metavar="URL", default=None,
                        help="opt in to upload this local media file to "
                             "URL/v1/sample for an independent hosted scan")
    parser.add_argument("--ignore-db", action="store_true",
                        help="ignore every known-target source (local DB, "
                             "hash list, API) and scan the target anyway")
    parser.add_argument("--model", choices=("auto", "joint", "binary"),
                        default="auto",
                        help="which NSFW model to use: 'joint' (5 classes, "
                             "the romance grade, slower) or 'binary' (2 "
                             "classes, ~4x faster, no romance grade - the "
                             "path for old phones); 'auto' picks the joint "
                             "one when it is there")
    parser.add_argument("--hwaccel", choices=("none", "auto", "cuda", "qsv",
                                               "d3d11va", "vaapi"),
                        default="none",
                        help="media: hardware-accelerated decoding "
                             "('auto' tries cuda/qsv/d3d11va/vaapi and falls "
                             "back to the software decoder). The hardware "
                             "frames are not bit-identical to the software "
                             "ones (a few levels of rounding), so a "
                             "borderline frame can flip")
    parser.add_argument("--benchmark", action="store_true",
                        help="media: benchmark the CPU and the GPU decoder "
                             "on this file and print the comparison (two "
                             "scans; the known-target cache is bypassed so "
                             "the file is really scanned)")
    parser.add_argument("--benchmark-model", action="store_true",
                        help="media: benchmark everything - both decoders "
                             "and both models (joint and binary), four "
                             "scans, one comparison table")
    parser.add_argument("--frames", choices=("raw", "png"), default="raw",
                        help="media: how the frames come out of ffmpeg - "
                             "'raw' (raw RGB, no PNG encode/decode, ~6 "
                             "ms/frame cheaper and bit-identical) or 'png' "
                             "(the PNG pipe)")
    parser.add_argument("--decoder", choices=("auto", "av", "ffmpeg"),
                        default="auto",
                        help="media: where the frames come from - 'av' "
                             "(PyAV, in-process: no pipe, ~2.7x faster "
                             "decode, but it brings its own FFmpeg and the "
                             "pixels are not bit-identical), 'ffmpeg' (the "
                             "external binary, the system build) or 'auto'")
    parser.add_argument("--face-crop", action="store_true",
                        help="media: run the gender model on the detected "
                             "face crop instead of the whole frame (the "
                             "finer '5' grade, but the face detector costs "
                             "14-36 ms per frame - it is deprecated as a "
                             "default)")
    parser.add_argument("--output", default=None, metavar="FILE",
                        help="write the result to this file (JSON) instead of "
                             "stdout - the benchmark needs it to keep the "
                             "per-run results")
    parser.add_argument("--no-progress", action="store_true",
                        help="do not print the scan progress to stderr")
    parser.add_argument("--tiles", choices=("3x3", "2x2", "off"), default="off",
                        help="media: also score sub-crops of each frame, so a "
                             "person in a wide shot is not diluted (off by "
                             "default: a tight crop of a face reads as "
                             "'porn' to both models, which invents explicit "
                             "verdicts; use it to hunt hidden messages)")
    parser.add_argument("--list-plugins", action="store_true",
                        help="list built-in classifiers and configured plugins, then exit")
    parser.add_argument("--skip-file", nargs="?", const="auto", default=None, metavar="FILE",
                        help="write a .skp skip file (sidecar) after video scan; "
                             "no path -> auto next to source; explicit path overrides")
    parser.add_argument("--skip-threshold", type=int, default=6, metavar="N",
                        help="minimum rating to include in .skp (default: 6)")
    parser.add_argument("--skip-actions", default=None, metavar="MAP",
                        help="rating->action mapping, e.g. '10=blackout+mute,7=blur,subtitle=mute'")
    args = parser.parse_args(argv)

    if args.list_plugins:
        from .plugins import load_plugins
        print("Built-in classifiers:")
        for name, filename, description in BUILTIN_CLASSIFIERS:
            available = os.path.isfile(_default_model_path(filename))
            state = "available" if available else "not installed"
            print(f"  {name}: {state} ({description})")
        print("Configured plugins:")
        plugins = load_plugins()
        if not plugins:
            print("  none loaded (set NSFW_PLUGINS to a JSON config file)")
        else:
            for p in plugins:
                labels = ", ".join(
                    f"{k}→{v}" for k, v in sorted(
                        p.labels.items(), key=lambda item: str(item[0]))
                )
                kinds = ", ".join(p.media_kinds)
                print(f"{p.name}: action={p.action}, size={p.size[0]}x{p.size[1]}, "
                      f"kinds=[{kinds}], labels=[{labels}]")
        return

    if args.submit_to_webservice and (
            not args.target or not os.path.isfile(args.target)):
        parser.error("--submit-to-webservice requires an existing local media file")
    if args.submit_to_webservice and (args.benchmark or args.benchmark_model):
        parser.error("--submit-to-webservice cannot be combined with benchmark modes")

    set_model_mode(args.model)

    known = None
    if not args.ignore_db:
        from .known import KnownTargets
        db_path = args.db or os.environ.get("NSFW_DB")
        if not db_path:
            # the repo's own product DB, when it is there
            here = os.path.dirname(os.path.abspath(__file__))
            candidate = os.path.normpath(os.path.join(
                here, "..", "..", "..", "nsfw-db", "data", "nsfw.sqlite"))
            db_path = candidate if os.path.isfile(candidate) else None
        hash_list = args.hash_list or os.environ.get("NSFW_HASH_LIST")
        if db_path or hash_list or (args.api_url and args.api_key):
            known = KnownTargets(db_path=db_path, hash_list=hash_list,
                                 api_url=args.api_url, api_key=args.api_key)

    media_options = {"fps": args.fps, "every_frame": args.every_frame,
                     "max_frames": args.max_frames, "tiles": args.tiles,
                     "hwaccel": args.hwaccel, "frames_format": args.frames,
                     "backend": args.decoder, "face_crop": args.face_crop}
    media_options["progress"] = progress_printer(not args.no_progress)
    if args.benchmark or args.benchmark_model:
        import sys
        # The benchmark replaces the single scan: every run is physical
        # (the known-target cache is bypassed, otherwise there would be
        # nothing to time).
        decoders = ["none", args.hwaccel if args.hwaccel != "none" else "auto"]
        models = (["joint", "binary"] if args.benchmark_model
                  else [args.model if args.model != "auto" else "joint"])
        runs = []
        for model in models:
            for decoder in decoders:
                runs.append({
                    "decoder": decoder,
                    "model": model,
                    "result": _benchmark_run(args.target, media_options,
                                             model, decoder, known, args),
                })
        result = runs[0]["result"]
        sys.stderr.write(format_benchmark_comparison(
            runs, "benchmark (backend: %s, models: %s)"
            % (args.decoder, ", ".join(models))))
        sys.stderr.write("\n")
        sys.stderr.flush()
    else:
        if args.skip_file is not None:
            media_options["include_frame_ratings"] = True
        result = rate(args.target, max_images=args.max_images,
                      timeout=args.timeout,
                      details=args.details, debug=args.debug,
                      user_agent=args.user_agent,
                      use_badwords=not args.no_badwords,
                      media_options=media_options,
                      known=known,
                      use_known=not args.ignore_db and args.skip_file is None)

    if args.imdb_id and isinstance(result, dict):
        result["imdb_id"] = args.imdb_id

    submission_error = None
    if args.submit_to_webservice:
        from .webservice import WebserviceError, submit_media_file
        try:
            hosted_result = submit_media_file(args.submit_to_webservice,
                                              args.target,
                                              imdb_id=args.imdb_id)
            if not isinstance(result, dict):
                result = {"target": args.target, "rating": result}
            result["webservice_submission"] = hosted_result
        except WebserviceError as error:
            submission_error = str(error)
            if not isinstance(result, dict):
                result = {"target": args.target, "rating": result}
            result["webservice_submission"] = {"error": submission_error}
            import sys
            sys.stderr.write(f"webservice submission failed: {submission_error}\n")

    # Write .skp skip file if requested (media results only)
    if args.skip_file is not None and not (args.benchmark or args.benchmark_model):
        try:
            from .skiplist import (
                entries_from_result,
                write_skiplist,
                sidecar_skipfile,
                parse_actions_string,
            )
            actions_map = DEFAULT_ACTIONS
            if args.skip_actions:
                actions_map = parse_actions_string(args.skip_actions)
            # Also allow env override
            env_actions = os.environ.get("NSFW_SKIP_ACTIONS")
            if env_actions and not args.skip_actions:
                actions_map = parse_actions_string(env_actions)
            entries = entries_from_result(
                result,
                threshold=args.skip_threshold,
                actions_map=actions_map,
            )
            if entries:
                if args.skip_file == "auto":
                    skp_path = sidecar_skipfile(args.target)
                else:
                    skp_path = args.skip_file
                # header info
                sha256 = result.get("sha256") or result.get("file_sha256")
                duration = (result.get("duration") or
                            result.get("media_duration") or
                            result.get("duration_sec"))
                header = {
                    "source": os.path.basename(args.target),
                    "sha256": sha256,
                    "duration": duration,
                }
                write_skiplist(skp_path, entries, header=header)
                import sys
                sys.stderr.write(f"wrote skip file: {skp_path}\n")
        except Exception as e:
            import sys
            sys.stderr.write(f"warning: failed to write .skp: {e}\n")

    if args.output:
        # The result goes to the file (JSON); stdout stays clean.
        import json
        import sys
        payload = result
        if args.benchmark or args.benchmark_model:
            payload = {"benchmark": [
                {"decoder": run["decoder"], "model": run["model"],
                 "result": run["result"]} for run in runs]}
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=str)
        sys.stderr.write("wrote %s\n" % args.output)
        return 1 if submission_error else None
    if args.benchmark or args.benchmark_model:
        # The benchmark's table is on stderr; the results themselves are
        # not dumped (they are huge) - --output keeps them.
        import sys
        sys.stderr.write(
            "the per-run results were not printed - use --output FILE to "
            "keep them\n")
        sys.stderr.flush()
        return 1 if submission_error else None
    print(result)
    return 1 if submission_error else None


if __name__ == "__main__":
    main()
