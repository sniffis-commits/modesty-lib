"""Compare locally stored image classifiers on labeled images and video frames.

The evaluator never downloads models or sends images anywhere. Model paths
must be supplied in a local JSON manifest.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from PIL import Image, ImageOps


RATING_SUFFIX = re.compile(r"_(10|[1-9])\.png$", re.IGNORECASE)
NUDENET_LABELS = (
    "FEMALE_GENITALIA_COVERED",
    "FACE_FEMALE",
    "BUTTOCKS_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_BREAST_EXPOSED",
    "ANUS_EXPOSED",
    "FEET_EXPOSED",
    "BELLY_COVERED",
    "FEET_COVERED",
    "ARMPITS_COVERED",
    "ARMPITS_EXPOSED",
    "FACE_MALE",
    "BELLY_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "ANUS_COVERED",
    "FEMALE_BREAST_COVERED",
    "BUTTOCKS_COVERED",
)


class Classifier(Protocol):
    name: str
    load_ms: float

    def predict(self, image: Image.Image) -> tuple[dict[str, float], float, float]:
        """Return scores, preprocessing milliseconds, and inference milliseconds."""


def softmax(values: np.ndarray) -> np.ndarray:
    """Convert a logits vector to stable probabilities."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("classifier returned an empty output")
    values -= values.max()
    exp = np.exp(values)
    return exp / exp.sum()


def expected_rating(path: Path) -> int | None:
    match = RATING_SUFFIX.search(path.name)
    return int(match.group(1)) if match else None


def nudenet_scores(output: np.ndarray) -> dict[str, float]:
    """Extract the highest proposal score per NudeNet class from YOLO output."""
    rows = np.asarray(output).squeeze()
    if rows.ndim != 2:
        raise ValueError("NudeNet output must reduce to a 2D array")
    class_count = len(NUDENET_LABELS)
    if rows.shape[0] == class_count + 4:
        rows = rows.T
    elif rows.shape[1] != class_count + 4:
        raise ValueError(
            "unexpected NudeNet output shape %r; expected 4 box values and %d classes"
            % (rows.shape, class_count)
        )
    if rows.shape[1] != class_count + 4:
        raise ValueError("unexpected NudeNet class dimension %r" % (rows.shape,))

    scores = {label: 0.0 for label in NUDENET_LABELS}
    for row in rows:
        class_values = row[4:]
        class_id = int(np.argmax(class_values))
        label = NUDENET_LABELS[class_id]
        scores[label] = max(scores[label], float(class_values[class_id]))
    return scores


def _resampling(value: int) -> Image.Resampling:
    return {
        0: Image.Resampling.NEAREST,
        1: Image.Resampling.LANCZOS,
        2: Image.Resampling.BILINEAR,
        3: Image.Resampling.BICUBIC,
        4: Image.Resampling.BOX,
        5: Image.Resampling.HAMMING,
    }.get(value, Image.Resampling.BILINEAR)


class HuggingFaceOnnxClassifier:
    def __init__(self, spec: dict[str, Any], manifest_dir: Path):
        import onnxruntime as ort

        started = time.perf_counter()
        model_path = _manifest_path(spec["model"], manifest_dir)
        config_path = _manifest_path(spec["config"], manifest_dir)
        processor_path = _manifest_path(spec["processor"], manifest_dir)
        self.name = spec["name"]
        config = json.loads(config_path.read_text(encoding="utf-8-sig"))
        processor = json.loads(processor_path.read_text(encoding="utf-8-sig"))
        self.labels = {
            int(index): label
            for index, label in config.get("id2label", {}).items()
        }
        size = processor.get("size", config.get("image_size", 224))
        if isinstance(size, dict):
            self.height = int(size.get("height", size.get("shortest_edge", 224)))
            self.width = int(size.get("width", size.get("shortest_edge", self.height)))
        else:
            self.height = self.width = int(size)
        self.mean = np.asarray(processor.get("image_mean", [0.0, 0.0, 0.0]), dtype=np.float32)
        self.std = np.asarray(processor.get("image_std", [1.0, 1.0, 1.0]), dtype=np.float32)
        self.resample = _resampling(int(processor.get("resample", 2)))
        self.input_layout = spec.get("input_layout", "nchw")
        self.input_name = spec.get("input_name")
        self.output_name = spec.get("output_name")

        options = ort.SessionOptions()
        options.intra_op_num_threads = int(spec.get("intra_op_threads", 0))
        options.log_severity_level = 3
        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.load_ms = (time.perf_counter() - started) * 1000.0
        if self.input_name is None:
            self.input_name = self.session.get_inputs()[0].name
        if self.output_name is None:
            self.output_name = self.session.get_outputs()[0].name

    def predict(self, image: Image.Image) -> tuple[dict[str, float], float, float]:
        start = time.perf_counter()
        arr = np.asarray(
            image.convert("RGB").resize((self.width, self.height), self.resample),
            dtype=np.float32,
        ) / 255.0
        arr = (arr - self.mean) / self.std
        if self.input_layout == "nchw":
            arr = np.transpose(arr, (2, 0, 1))
        elif self.input_layout != "nhwc":
            raise ValueError("input_layout must be 'nchw' or 'nhwc'")
        input_tensor = np.ascontiguousarray(arr[np.newaxis, ...], dtype=np.float32)
        preprocess_ms = (time.perf_counter() - start) * 1000.0

        start = time.perf_counter()
        output = self.session.run([self.output_name], {self.input_name: input_tensor})[0]
        inference_ms = (time.perf_counter() - start) * 1000.0
        probabilities = softmax(np.asarray(output))
        return {
            self.labels.get(index, str(index)): float(probability)
            for index, probability in enumerate(probabilities)
        }, preprocess_ms, inference_ms


class NudeNetOnnxClassifier:
    def __init__(self, spec: dict[str, Any], manifest_dir: Path):
        import onnxruntime as ort

        self.name = spec["name"]
        self.input_size = int(spec.get("input_size", 320))
        options = ort.SessionOptions()
        options.intra_op_num_threads = int(spec.get("intra_op_threads", 0))
        options.log_severity_level = 3
        started = time.perf_counter()
        self.session = ort.InferenceSession(
            str(_manifest_path(spec["model"], manifest_dir)),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.load_ms = (time.perf_counter() - started) * 1000.0
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict(self, image: Image.Image) -> tuple[dict[str, float], float, float]:
        start = time.perf_counter()
        rgb = image.convert("RGB")
        width, height = rgb.size
        side = max(width, height)
        padded = ImageOps.expand(
            rgb,
            (0, 0, side - width, side - height),
            fill=(0, 0, 0),
        )
        prepared = np.asarray(
            padded.resize(
                (self.input_size, self.input_size),
                Image.Resampling.BILINEAR,
            ),
            dtype=np.float32,
        ) / 255.0
        prepared = np.ascontiguousarray(
            prepared.transpose(2, 0, 1)[np.newaxis, ...],
            dtype=np.float32,
        )
        preprocess_ms = (time.perf_counter() - start) * 1000.0

        start = time.perf_counter()
        output = self.session.run(
            [self.output_name],
            {self.input_name: prepared},
        )[0]
        inference_ms = (time.perf_counter() - start) * 1000.0
        return nudenet_scores(output), preprocess_ms, inference_ms


class HuggingFaceTransformersClassifier:
    def __init__(self, spec: dict[str, Any], manifest_dir: Path):
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForImageClassification
        except ImportError as error:
            raise RuntimeError(
                "The transformers model adapter requires local torch and transformers installs"
            ) from error

        self.name = spec["name"]
        model_dir = _manifest_path(spec["model_dir"], manifest_dir)
        started = time.perf_counter()
        self.processor = AutoImageProcessor.from_pretrained(
            str(model_dir),
            local_files_only=True,
        )
        self.model = AutoModelForImageClassification.from_pretrained(
            str(model_dir),
            local_files_only=True,
        ).to("cpu").eval()
        self.torch = torch
        self.load_ms = (time.perf_counter() - started) * 1000.0

    def predict(self, image: Image.Image) -> tuple[dict[str, float], float, float]:
        start = time.perf_counter()
        inputs = self.processor(images=image.convert("RGB"), return_tensors="pt")
        inputs = {key: value.to("cpu") for key, value in inputs.items()}
        preprocess_ms = (time.perf_counter() - start) * 1000.0

        start = time.perf_counter()
        with self.torch.inference_mode():
            logits = self.model(**inputs).logits[0]
            probabilities = self.torch.softmax(logits, dim=-1).cpu().numpy()
        inference_ms = (time.perf_counter() - start) * 1000.0
        return {
            self.model.config.id2label.get(index, str(index)): float(probability)
            for index, probability in enumerate(probabilities)
        }, preprocess_ms, inference_ms


class LlamaGuardVisionClassifier:
    """Run local Llama Guard 4 image classification without downloading weights."""

    def __init__(self, spec: dict[str, Any], manifest_dir: Path):
        try:
            import torch
            from transformers import AutoProcessor, Llama4ForConditionalGeneration
        except ImportError as error:
            raise RuntimeError(
                "The llama-guard-vision adapter requires local torch and transformers installs"
            ) from error

        self.name = spec["name"]
        self.device = spec.get("device", "cpu")
        dtype_name = spec.get("torch_dtype", "auto")
        if dtype_name == "auto":
            torch_dtype: Any = "auto"
        else:
            torch_dtype = getattr(torch, dtype_name, None)
            if torch_dtype is None:
                raise ValueError(
                    "unsupported torch_dtype %r; use auto, float32, float16, or bfloat16"
                    % dtype_name
                )

        model_dir = _manifest_path(spec["model_dir"], manifest_dir)
        started = time.perf_counter()
        self.processor = AutoProcessor.from_pretrained(
            str(model_dir),
            local_files_only=True,
        )
        self.model = Llama4ForConditionalGeneration.from_pretrained(
            str(model_dir),
            local_files_only=True,
            torch_dtype=torch_dtype,
        ).to(self.device).eval()
        self.torch = torch
        self.max_new_tokens = int(spec.get("max_new_tokens", 16))
        self.load_ms = (time.perf_counter() - started) * 1000.0

    @staticmethod
    def parse_response(response: str) -> dict[str, float]:
        """Store the model's categorical verdict and any MLCommons category IDs."""
        normalized = response.strip().casefold()
        if normalized.startswith("unsafe"):
            unsafe = 1.0
        elif normalized.startswith("safe"):
            unsafe = 0.0
        else:
            raise ValueError("Llama Guard returned an unrecognized verdict: %r" % response)

        categories = set(re.findall(r"\bS\d{1,2}\b", response.upper()))
        scores = {"verdict_unsafe": unsafe}
        scores.update({category: 1.0 for category in sorted(categories)})
        return scores

    def predict(self, image: Image.Image) -> tuple[dict[str, float], float, float]:
        start = time.perf_counter()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image.convert("RGB")},
                    {"type": "text", "text": "Classify the safety of this image."},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)
        preprocess_ms = (time.perf_counter() - start) * 1000.0

        start = time.perf_counter()
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        response = self.processor.batch_decode(
            generated[:, inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
        )[0]
        inference_ms = (time.perf_counter() - start) * 1000.0
        return self.parse_response(response), preprocess_ms, inference_ms


class TensorFlowKerasClassifier:
    def __init__(self, spec: dict[str, Any], manifest_dir: Path):
        try:
            import tensorflow as tf
            import tensorflow_hub as hub
        except ImportError as error:
            raise RuntimeError(
                "The tensorflow-keras adapter requires local tensorflow and tensorflow-hub installs"
            ) from error

        self.name = spec["name"]
        self.input_size = int(spec.get("input_size", 224))
        self.labels = spec.get(
            "labels",
            ["drawings", "hentai", "neutral", "porn", "sexy"],
        )
        if not isinstance(self.labels, list) or not self.labels:
            raise ValueError("tensorflow-keras models require a non-empty labels list")

        started = time.perf_counter()
        self.model = tf.keras.models.load_model(
            str(_manifest_path(spec["model"], manifest_dir)),
            custom_objects={"KerasLayer": hub.KerasLayer},
            compile=False,
        )
        self.load_ms = (time.perf_counter() - started) * 1000.0

    def predict(self, image: Image.Image) -> tuple[dict[str, float], float, float]:
        start = time.perf_counter()
        arr = np.asarray(
            image.convert("RGB").resize(
                (self.input_size, self.input_size),
                Image.Resampling.NEAREST,
            ),
            dtype=np.float32,
        ) / 255.0
        batch = arr[np.newaxis, ...]
        preprocess_ms = (time.perf_counter() - start) * 1000.0

        start = time.perf_counter()
        output = self.model.predict(batch, batch_size=1, verbose=0)
        inference_ms = (time.perf_counter() - start) * 1000.0
        probabilities = np.asarray(output, dtype=np.float32)
        if probabilities.ndim != 2 or probabilities.shape != (1, len(self.labels)):
            raise ValueError(
                "TensorFlow classifier returned shape %r for %d labels"
                % (probabilities.shape, len(self.labels))
            )
        return {
            label: float(probability)
            for label, probability in zip(self.labels, probabilities[0])
        }, preprocess_ms, inference_ms


def _manifest_path(value: str, manifest_dir: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (manifest_dir / path).resolve()


def load_classifiers(manifest_path: Path) -> list[Classifier]:
    config = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    specs = config.get("models")
    if not isinstance(specs, list) or not specs:
        raise ValueError("manifest must contain a non-empty 'models' list")
    constructors = {
        "hf-onnx": HuggingFaceOnnxClassifier,
        "huggingface": HuggingFaceTransformersClassifier,
        "llama-guard-vision": LlamaGuardVisionClassifier,
        "nudenet-onnx": NudeNetOnnxClassifier,
        "tensorflow-keras": TensorFlowKerasClassifier,
    }
    classifiers = []
    names = set()
    for spec in specs:
        if not isinstance(spec, dict) or not spec.get("name"):
            raise ValueError("every model entry must have a name")
        if spec["name"] in names:
            raise ValueError("model names must be unique: %s" % spec["name"])
        names.add(spec["name"])
        kind = spec.get("kind")
        if kind not in constructors:
            raise ValueError("unsupported model kind %r" % kind)
        classifiers.append(constructors[kind](spec, manifest_path.parent))
    return classifiers


def collect_labeled_images(directory: Path) -> list[tuple[Path, int | None]]:
    return [
        (path, expected_rating(path))
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.suffix.lower() == ".png"
        and expected_rating(path) is not None
    ]


def extract_video_frames(
    video: Path,
    fps: float,
    ffmpeg: str,
) -> tuple[tempfile.TemporaryDirectory[str], list[tuple[Path, float]]]:
    temporary = tempfile.TemporaryDirectory(prefix="nsfw-model-eval-")
    frame_pattern = str(Path(temporary.name) / "frame_%08d.png")
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video),
        "-vf",
        "fps=%s" % fps,
        "-vsync",
        "0",
        "-start_number",
        "1",
        frame_pattern,
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=3600,
        )
        if result.returncode:
            raise RuntimeError(
                "ffmpeg failed for %s: %s"
                % (video, result.stderr.strip() or "exit code %d" % result.returncode)
            )
        frames = []
        for path in sorted(Path(temporary.name).glob("frame_*.png")):
            index = int(path.stem.removeprefix("frame_"))
            frames.append((path, (index - 1) / fps))
        return temporary, frames
    except BaseException:
        temporary.cleanup()
        raise


def percentile(values: list[float], value: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), value))


def _format_ms(value: float | None) -> str:
    return "n/a" if value is None else "%.2f" % value


def binary_threshold_grid(
    rows: list[dict[str, Any]],
) -> dict[str, list[tuple[float, int, int, int, int]]]:
    """Compare local binary NSFW scores with the labeled 6-vs-7 boundary."""
    grouped: dict[str, list[tuple[bool, float]]] = {}
    for row in rows:
        expected = row["expected_rating"]
        if expected in (None, ""):
            continue
        scores = json.loads(row["scores"])
        normalized_scores = {
            label.casefold(): float(score) for label, score in scores.items()
        }
        positive_score = normalized_scores.get("nsfw")
        if positive_score is None and {
            "porn",
            "hentai",
        }.issubset(normalized_scores):
            positive_score = (
                normalized_scores["porn"] + normalized_scores["hentai"]
            )
        if positive_score is not None:
            grouped.setdefault(row["model"], []).append(
                (int(expected) >= 7, positive_score)
            )

    grid = {}
    for name, labeled in grouped.items():
        results = []
        for threshold in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
            true_positive = sum(
                label and score >= threshold for label, score in labeled
            )
            false_positive = sum(
                not label and score >= threshold for label, score in labeled
            )
            true_negative = sum(
                not label and score < threshold for label, score in labeled
            )
            false_negative = sum(
                label and score < threshold for label, score in labeled
            )
            results.append(
                (threshold, true_positive, false_positive, true_negative, false_negative)
            )
        grid[name] = results
    return grid


def evaluate(
    classifiers: list[Classifier],
    sources: list[tuple[Path, str, float | None, int | None]],
    output_path: Path,
) -> float:
    if not sources:
        raise ValueError("no image or video frames were selected")

    evaluation_start = time.perf_counter()
    first_path = sources[0][0]
    with Image.open(first_path) as first_image:
        warmup_image = first_image.convert("RGB")
        first_calls = {
            classifier.name: classifier.predict(warmup_image)
            for classifier in classifiers
        }

    rows: list[dict[str, Any]] = []
    timings: dict[str, dict[str, list[float]]] = {
        classifier.name: {
            "decode": [],
            "preprocess": [],
            "inference": [],
            "frame_total": [],
        }
        for classifier in classifiers
    }
    for path, source_kind, timestamp, rating in sources:
        decode_start = time.perf_counter()
        with Image.open(path) as opened:
            image = opened.convert("RGB")
        decode_ms = (time.perf_counter() - decode_start) * 1000.0
        for classifier in classifiers:
            scores, preprocess_ms, inference_ms = classifier.predict(image)
            total_ms = decode_ms + preprocess_ms + inference_ms
            timings[classifier.name]["decode"].append(decode_ms)
            timings[classifier.name]["preprocess"].append(preprocess_ms)
            timings[classifier.name]["inference"].append(inference_ms)
            timings[classifier.name]["frame_total"].append(total_ms)
            rows.append(
                {
                    "model": classifier.name,
                    "source_kind": source_kind,
                    "source": path.name,
                    "timestamp_seconds": timestamp,
                    "expected_rating": rating,
                    "scores": json.dumps(scores, sort_keys=True),
                    "decode_ms": decode_ms,
                    "preprocess_ms": preprocess_ms,
                    "inference_ms": inference_ms,
                    "frame_total_ms": total_ms,
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "model",
        "source_kind",
        "source",
        "timestamp_seconds",
        "expected_rating",
        "scores",
        "decode_ms",
        "preprocess_ms",
        "inference_ms",
        "frame_total_ms",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    print("model | load ms | first-call ms | frames | inference p50/p95 ms | frame p95 ms")
    for classifier in classifiers:
        model_times = timings[classifier.name]
        first_call = first_calls[classifier.name]
        first_total = first_call[1] + first_call[2]
        print(
            "%s | %s | %s | %d | %s/%s | %s"
            % (
                classifier.name,
                _format_ms(classifier.load_ms),
                _format_ms(first_total),
                len(model_times["inference"]),
                _format_ms(percentile(model_times["inference"], 50)),
                _format_ms(percentile(model_times["inference"], 95)),
                _format_ms(percentile(model_times["frame_total"], 95)),
            )
        )
    threshold_results = binary_threshold_grid(rows)
    if threshold_results:
        print("offline binary threshold grid (positive=expected rating >=7; TP/FP/TN/FN):")
        for name, results in threshold_results.items():
            print(
                "%s | labeled=%d | %s"
                % (
                    name,
                    sum(
                        row["model"] == name and row["expected_rating"] is not None
                        for row in rows
                    ),
                    " ".join(
                        "%.1f:%d/%d/%d/%d" % result for result in results
                    ),
                )
            )
    print("wrote %d side-by-side rows to %s" % (len(rows), output_path))
    return (time.perf_counter() - evaluation_start) * 1000.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run local FOSS classifiers side-by-side; no model downloads or uploads."
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--samples-dir", type=Path)
    parser.add_argument("--image", action="append", type=Path, default=[])
    parser.add_argument("--video", action="append", type=Path, default=[])
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.fps <= 0 or not math.isfinite(args.fps):
        parser.error("--fps must be a positive finite number")
    if not args.samples_dir and not args.image and not args.video:
        parser.error("provide at least one of --samples-dir, --image, or --video")
    for path in args.image + args.video:
        if not path.is_file():
            parser.error("input file does not exist: %s" % path)
    if args.samples_dir and not args.samples_dir.is_dir():
        parser.error("sample directory does not exist: %s" % args.samples_dir)

    selected: list[tuple[Path, str, float | None, int | None]] = []
    if args.samples_dir:
        selected.extend(
            (path, "labeled-sample", None, rating)
            for path, rating in collect_labeled_images(args.samples_dir)
        )
    for image_path in args.image:
        selected.append((image_path, "image", None, expected_rating(image_path)))

    if not selected and not args.video:
        parser.error("no labeled PNGs were found in --samples-dir")

    temporaries: list[tempfile.TemporaryDirectory[str]] = []
    model_startup_ms = 0.0
    extraction_ms = 0.0
    evaluation_ms = 0.0
    wall_start = time.perf_counter()
    try:
        classifiers = load_classifiers(args.manifest.resolve())
        model_startup_ms = sum(classifier.load_ms for classifier in classifiers)
        for video in args.video:
            extraction_start = time.perf_counter()
            temporary, frames = extract_video_frames(video, args.fps, args.ffmpeg)
            extraction_ms += (time.perf_counter() - extraction_start) * 1000.0
            temporaries.append(temporary)
            selected.extend(
                (frame, "video-frame:%s" % video.name, timestamp, None)
                for frame, timestamp in frames
            )
        evaluation_ms = evaluate(classifiers, selected, args.output.resolve())
    finally:
        for temporary in temporaries:
            temporary.cleanup()
    wall_ms = (time.perf_counter() - wall_start) * 1000.0
    print(
        "wall time | model loading %.2f ms | video extraction %.2f ms | "
        "model evaluation %.2f ms | total %.2f ms"
        % (model_startup_ms, extraction_ms, evaluation_ms, wall_ms)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
