"""Is this a photograph, or synthetic content (3D render / UI / drawing)?

The NSFW model (Yahoo ``open_nsfw``) is trained on **real photographs**.
On rendered or drawn content it still produces a score, but the score no
longer tracks nudity - a fully clothed game character scored 0.59 and a
game-engine map editor 0.52 in our tests.  Those are false positives we
cannot fix by moving the threshold, so we detect synthetic content and
stop trusting the picture model for it.

The signals are cheap (numpy only, no extra model, no extra dependency):

``noise``
    The 25th percentile of the absolute difference between neighbouring
    pixels.  Camera footage has sensor noise everywhere, so this is > 0;
    a render or a UI screenshot is noise-free, so it is 0.
``flat``
    The share of neighbouring pixel pairs that are *exactly* equal.
    High for UI screenshots and flat shading, low for a busy photo.

The noise measure is defeated by compression (a JPEG of a render gets
noise from the codec), which is why the media scanner decides on the
**file level**: it samples frames spread over the whole duration and
looks at the ratio.  A single image only gets the weaker per-frame test.
"""

import numpy as np

__all__ = ["frame_stats", "is_synthetic_frame"]


def frame_stats(image):
    """Noise/flatness statistics of one image (PIL image or numpy array).

    Returns ``{"noise": float, "flat": float, "pixels": int}``.
    """
    if hasattr(image, "convert"):                 # a PIL image
        image = image.convert("RGB")
        array = np.asarray(image)
    else:
        array = np.asarray(image)
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    if array.shape[0] < 2 or array.shape[1] < 2:
        return {"noise": 0.0, "flat": 0.0, "pixels": int(array.size)}
    array = array.astype(np.int16)
    dx = np.abs(np.diff(array, axis=1))
    noise = float(np.percentile(dx, 25))
    flat = float((dx == 0).all(axis=2).mean())
    return {"noise": noise, "flat": flat, "pixels": int(array.shape[0] * array.shape[1])}


def is_synthetic_frame(image, face_session=None, noise_max=None, flat_min=None):
    """True when the frame is confidently synthetic (a render / a UI / a drawing).

    Two signals have to agree:

    * **no real face** - the face detector is trained on photographed
      faces, so a drawn or rendered face is not found (a music video's
      singer is found in every frame);
    * **noise-free** - camera footage has sensor noise, a render or a
      screenshot does not.  (Compression removes it, which is why the
      media scanner also samples many frames and the face signal does
      the heavy lifting.)

    A photograph without a visible face (a body, a landscape) is caught
    by the second signal - the price of not having a photo-vs-drawing
    classifier, and the reason such a file is reported as "not rated"
    rather than as a false positive.
    """
    from .scanner import (NSFW_SYNTHETIC_FLAT_MIN, NSFW_SYNTHETIC_NOISE_MAX,
                          detect_faces)

    if noise_max is None:
        noise_max = NSFW_SYNTHETIC_NOISE_MAX
    if flat_min is None:
        flat_min = NSFW_SYNTHETIC_FLAT_MIN
    stats = frame_stats(image)
    noise_free = stats["noise"] <= noise_max
    faces = 0
    if face_session is not None:
        try:
            faces = len(detect_faces(image, face_session))
        except Exception:
            faces = 0
    stats["faces"] = faces
    stats["synthetic"] = bool(noise_free and (faces == 0
                                             or stats["flat"] >= flat_min))
    return stats
