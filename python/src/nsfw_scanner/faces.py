"""Face detection - an optional extra, on its way out.

The UltraFace-320 detector finds the faces in a frame; the gender model
then runs on the *face crop* (it was trained on face crops), which is
what makes the "5 - women/girls present" grade work.

It is also one of the two signals of the **synthetic gate** (a render, a
drawing or a UI screenshot has no real face in it) — that signal needs
the detector.

**Status: deprecated as a default.**  Measured on real video, the
detector costs 14-36 ms per frame (a third of the per-frame budget) and
the *file ratings do not change without it* - only the fine-grained "5"
grades get coarser (the gender model then runs on the whole frame).  So:

* the **media scanner** does not use it per frame any more (it still
  uses it for the synthetic gate, which samples only a couple of dozen
  frames over the whole file - negligible);
* the **single-image** path still uses it (one image, the cost does not
  matter, and the "5" grade stays accurate);
* the code lives here, apart from the core, so it can be imported (or
  dropped) on its own.

Import it directly if you want the face-crop path::

    from nsfw_scanner.faces import detect_faces, get_face_session
"""

import os

import numpy as np
from PIL import Image

try:
    import onnxruntime as ort
except ImportError:  # optional: text-only mode still works
    ort = None

__all__ = [
    "FACE_INPUT_SIZE",
    "FACE_MIN_SIZE",
    "FACE_THRESHOLD",
    "detect_faces",
    "get_face_session",
    "load_face_model",
    "preprocess_face",
]


# UltraFace-320 (version-RFB-320); the box decode is ported from
# onnx/models, MIT.
FACE_INPUT_SIZE = (320, 240)  # width x height expected by version-RFB-320
FACE_THRESHOLD = 0.9  # min detector confidence; small detectors hallucinate
FACE_MIN_SIZE = 24  # min face-box width/height in original-image pixels

_face_cache = {}


def _default_model_path(filename):
    """Resolves a model file: NSFW_MODEL_DIR env > bundled models > cwd."""
    here = os.path.dirname(os.path.abspath(__file__))
    env_dir = os.environ.get("NSFW_MODEL_DIR")
    if env_dir:
        candidate = os.path.join(env_dir, filename)
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    candidate = os.path.join(here, "models", filename)
    if os.path.exists(candidate):
        return os.path.abspath(candidate)
    if os.path.exists(filename):
        return os.path.abspath(filename)
    return os.path.abspath(candidate)  # canonical spot; the loader returns None


def load_face_model(model_path=None):
    """Loads the UltraFace-320 face detector. Returns None if unavailable."""
    if ort is None:
        return None
    model_path = model_path or _default_model_path("face_model.onnx")
    if not os.path.exists(model_path):
        return None
    options = ort.SessionOptions()
    options.log_severity_level = 3
    return ort.InferenceSession(model_path, sess_options=options,
                                providers=["CPUExecutionProvider"])


def get_face_session(model_path=None):
    """Returns the cached face-detection session (None if unavailable)."""
    path = os.path.abspath(model_path or _default_model_path("face_model.onnx"))
    if path not in _face_cache:
        _face_cache[path] = load_face_model(path)
    return _face_cache[path]


def _box_area(left_top, right_bottom):
    hw = np.clip(right_bottom - left_top, 0.0, None)
    return hw[..., 0] * hw[..., 1]


def _box_iou(boxes0, boxes1, eps=1e-5):
    overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])
    overlap_area = _box_area(overlap_left_top, overlap_right_bottom)
    area0 = _box_area(boxes0, boxes0)
    area1 = _box_area(boxes1, boxes1)
    union = area0 + area1 - overlap_area
    return overlap_area / (union + eps)


def _hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    picked = []
    indexes = np.argsort(scores)[-candidate_size:]
    while len(indexes) > 0:
        current = indexes[-1]
        picked.append(current)
        if 0 < top_k == len(picked) or len(indexes) == 1:
            break
        current_box = boxes[current, :]
        indexes = indexes[:-1]
        ious = _box_iou(boxes[indexes, :], np.expand_dims(current_box, axis=0))
        indexes = indexes[ious <= iou_threshold]
    return box_scores[picked, :]


def _pick_face_boxes(width, height, confidences, boxes, prob_threshold,
                     iou_threshold=0.5):
    """Decodes raw UltraFace outputs to pixel boxes. Returns (boxes, probs)."""
    boxes = boxes[0]
    confidences = confidences[0]
    picked, probs = [], []
    for class_index in range(1, confidences.shape[1]):
        class_probs = confidences[:, class_index]
        mask = class_probs > prob_threshold
        class_probs = class_probs[mask]
        if class_probs.shape[0] == 0:
            continue
        box_probs = np.concatenate([boxes[mask, :], class_probs.reshape(-1, 1)],
                                   axis=1)
        box_probs = _hard_nms(box_probs, iou_threshold=iou_threshold)
        picked.append(box_probs)
    if not picked:
        return np.zeros((0, 4), dtype=np.int32), np.zeros((0,))
    picked = np.concatenate(picked)
    picked[:, 0] *= width
    picked[:, 1] *= height
    picked[:, 2] *= width
    picked[:, 3] *= height
    return picked[:, :4].astype(np.int32), picked[:, 4]


def preprocess_face(img):
    """UltraFace-320 pipeline: 320x240 RGB, (x-127)/128, NCHW batch."""
    img = img.convert("RGB").resize(FACE_INPUT_SIZE, resample=Image.BILINEAR)
    arr = (np.asarray(img, dtype=np.float32) - 127.0) / 128.0
    arr = np.transpose(arr, (2, 0, 1))[np.newaxis, ...]
    return np.ascontiguousarray(arr, dtype=np.float32)


def _square_box(box, width, height):
    """Expands a face box to a square (reference pipeline crops squares)."""
    x1, y1, x2, y2, prob = box
    side = max(x2 - x1, y2 - y1)
    dx = (side - (x2 - x1)) // 2
    dy = (side - (y2 - y1)) // 2
    return (max(0, x1 - dx), max(0, y1 - dy),
            min(width, x2 + dx), min(height, y2 + dy), prob)


def detect_faces(img, face_session, threshold=FACE_THRESHOLD):
    """Finds faces, returning [(x1, y1, x2, y2, prob)] in original pixels."""
    try:
        if face_session is None:
            return []
        img = img.convert("RGB")
        width, height = img.size
        input_name = face_session.get_inputs()[0].name
        output_names = [o.name for o in face_session.get_outputs()]
        confidences, boxes = face_session.run(
            output_names, {input_name: preprocess_face(img)})
        kept, probs = _pick_face_boxes(width, height, confidences, boxes,
                                       threshold)
        faces = []
        for (x1, y1, x2, y2), prob in zip(kept.tolist(), probs.tolist()):
            x1 = max(0, min(width - 1, x1))
            y1 = max(0, min(height - 1, y1))
            x2 = max(0, min(width, x2))
            y2 = max(0, min(height, y2))
            if x2 - x1 < FACE_MIN_SIZE or y2 - y1 < FACE_MIN_SIZE:
                continue  # logo-size hallucinations, not faces
            if x2 > x1 and y2 > y1:
                faces.append((x1, y1, x2, y2, float(prob)))
        return faces
    except Exception:
        return []
