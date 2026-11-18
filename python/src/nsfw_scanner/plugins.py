"""Classifier plugins: bring your own model, and say what it may do.

A plugin is a model that answers one question about a picture.  The
**config file** says, per plugin: which model, how to run it, which of
its labels mean what, and — the important part — **what it is allowed to
do**:

``rate``
    it decides the rating (it replaces the built-in one);
``censor``
    it may only **raise** the rating, never lower it;
``skip``
    it is ignored for that media kind;
``report``
    its verdict is stored as the evidence, it changes nothing.

The built-in scale stays ours (1 no unsafe signal · 2 synthetic · 5 people ·
6 romance · 7 revealing · 10 explicit) and the worst-frame rule stays in
the core, so two deployments with different plugins still produce
comparable verdicts.

The config is JSON (the standard library, no dependency)::

    {
      "plugins": [
        {"name": "drawing-tagger",
         "kind": "onnx",
         "model": "drawings.onnx",
         "labels": {"hentai": 10, "porn": 10},
         "action": "censor",
         "media_kinds": ["image"]},

        {"name": "my-own-model",
         "kind": "onnx",
         "model": "mine.onnx",
         "size": [224, 224],
         "labels": {"porn": 10, "sexy": 6},
         "action": "rate",
         "media_kinds": ["image", "video"]}
      ]
    }

Point the library at it with the ``NSFW_PLUGINS`` environment variable
(or pass ``plugins=`` to the scanner).
"""

import json
import logging
import os

__all__ = ["Plugin", "load_plugins", "merge_verdicts", "ACTIONS",
           "MEDIA_KINDS"]

_LOGGER = logging.getLogger(__name__)
ACTIONS = ("rate", "censor", "skip", "report")
MEDIA_KINDS = ("image", "video", "audio", "text")


class Plugin:
    """One configured classifier."""

    def __init__(self, name, session, labels, action="censor",
                 media_kinds=None, size=(224, 224), mean=None, std=None,
                 class_labels=None):
        self.name = name
        self.session = session
        self.labels = dict(labels or {})
        self.action = action if action in ACTIONS else "censor"
        self.media_kinds = tuple(media_kinds or MEDIA_KINDS)
        self.size = tuple(size or (224, 224))
        self.mean = mean
        self.std = std
        self.class_labels = (
            tuple(str(label) for label in class_labels)
            if class_labels is not None else None
        )

    def handles(self, media_kind):
        return media_kind in self.media_kinds

    def classify(self, img):
        """The plugin's labels for an image: {label: probability}."""
        if self.session is None:
            return {}
        try:
            import numpy as np
            from PIL import Image

            width, height = self.size
            arr = np.asarray(img.convert("RGB").resize((width, height)),
                             dtype=np.float32) / 255.0
            if self.mean is not None and self.std is not None:
                arr = (arr - self.mean) / self.std
            name = self.session.get_inputs()[0].name
            outputs = self.session.run(None, {name: np.ascontiguousarray(
                arr[np.newaxis, ...], dtype=np.float32)})
            flat = np.asarray(outputs[0]).reshape(-1)
        except Exception as error:
            _LOGGER.warning("plugin classifier %s failed: %s", self.name, error)
            return {}
        total = float(flat.sum()) if flat.size else 0.0
        out = {}
        for index, value in enumerate(flat):
            label = (
                self.class_labels[index]
                if self.class_labels is not None and index < len(self.class_labels)
                else index
            )
            if self.class_labels is None and label not in self.labels:
                continue
            out[label] = float(value) if total <= 1.0 else float(value) / total
        return out

    def rating_for(self, img):
        """The rating this plugin proposes (None when it has nothing)."""
        labels = self.classify(img)
        if not labels:
            return None
        best_label = max(labels, key=labels.get)
        rating = self.labels.get(best_label)
        if (rating is None and self.class_labels is not None and
                isinstance(best_label, str)):
            index = self.class_labels.index(best_label)
            rating = self.labels.get(index, self.labels.get(str(index)))
        elif rating is None and isinstance(best_label, int):
            rating = self.labels.get(str(best_label))
        return int(rating) if rating is not None else None


def _session_for(model_path):
    """An ONNX session for a plugin model (None when it cannot be loaded)."""
    try:
        import onnxruntime as ort
    except ImportError:
        return None
    if not os.path.exists(model_path):
        return None
    options = ort.SessionOptions()
    options.log_severity_level = 3
    try:
        return ort.InferenceSession(model_path, sess_options=options,
                                    providers=["CPUExecutionProvider"])
    except Exception:
        return None


def load_plugins(path=None):
    """Reads the plugin config.  Returns a list of :class:`Plugin`.

    A broken plugin is skipped (never a crash, never a false no-signal verdict).
    """
    path = path or os.environ.get("NSFW_PLUGINS")
    if not path:
        return []
    if not os.path.exists(path):
        _LOGGER.warning("plugin config does not exist: %s", path)
        return []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            config = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        _LOGGER.warning("could not read plugin config %s: %s", path, error)
        return []
    if not isinstance(config, dict) or not isinstance(config.get("plugins", []), list):
        _LOGGER.warning("plugin config %s must contain a 'plugins' array", path)
        return []
    here = os.path.dirname(os.path.abspath(path))
    plugins = []
    for entry in (config.get("plugins") or []):
        try:
            if not isinstance(entry, dict):
                raise ValueError("plugin entries must be objects")
            model = entry.get("model")
            if not model:
                raise ValueError("missing model path")
            model_path = model if os.path.isabs(model or "") else os.path.join(
                here, model or "")
            session = _session_for(model_path)
            if session is None:
                _LOGGER.warning("plugin %s model could not be loaded: %s",
                                entry.get("name", model), model_path)
                continue
            labels = entry.get("labels") or {}
            if not isinstance(labels, dict):
                raise ValueError("'labels' must be an object")
            class_labels = entry.get("class_labels")
            if class_labels is not None and (
                not isinstance(class_labels, list) or
                not all(isinstance(label, str) for label in class_labels)
            ):
                raise ValueError("'class_labels' must be an array of strings")
            rating_map = {}
            for key, value in labels.items():
                try:
                    normalized_key = int(key)
                except (TypeError, ValueError):
                    normalized_key = str(key)
                try:
                    rating_map[normalized_key] = int(value)
                except (TypeError, ValueError):
                    raise ValueError("label ratings must be integers") from None
            plugin = Plugin(entry.get("name") or os.path.basename(model_path),
                            session, rating_map,
                            action=entry.get("action") or "censor",
                            media_kinds=entry.get("media_kinds"),
                            size=entry.get("size") or (224, 224),
                            class_labels=class_labels)
            plugins.append(plugin)
        except (TypeError, ValueError) as error:
            _LOGGER.warning("invalid plugin in %s: %s", path, error)
            continue
    return plugins


def merge_verdicts(rating, plugins, img, media_kind="image"):
    """Folds the plugins' verdicts into the rating.

    Returns ``(rating, evidence)`` — the evidence lists what every plugin
    said, so a decision is always explainable.
    """
    evidence = []
    for plugin in plugins or []:
        if not plugin.handles(media_kind) or plugin.action == "skip":
            continue
        proposed = plugin.rating_for(img)
        evidence.append({"plugin": plugin.name, "action": plugin.action,
                         "rating": proposed})
        if proposed is None:
            continue
        if plugin.action == "rate":
            rating = proposed
        elif plugin.action == "censor" and rating is not None:
            rating = max(rating, proposed)
    return rating, evidence
