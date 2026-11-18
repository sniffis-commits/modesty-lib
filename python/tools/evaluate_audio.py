"""Compare local audio-scanning options on one labeled corpus.

Audio paths and transcripts remain local. Model downloads require the
explicit --allow-downloads switch; transcripts are never written to disk.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from nsfw_scanner import audio as audio_scanner
from nsfw_scanner.audio import scan_audio_transcript


THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


@dataclass(frozen=True)
class AudioSample:
    path: Path
    expected_rating: int | None
    expected_vocalization: bool | None
    expected_events: frozenset[str] | None
    reference_transcript: str | None
    duration_sec: float | None


@dataclass
class Prediction:
    status: str = "completed"
    rating: int | None = None
    transcript: str | None = None
    event_scores: dict[str, float] | None = None
    vocalization_score: float | None = None
    error: str | None = None


class Runner(Protocol):
    def predict(self, sample: AudioSample) -> Prediction:
        """Return one model's result for a labeled audio sample."""


@dataclass
class ModelOption:
    name: str
    kind: str
    tier: str
    runner: Runner | None
    load_ms: float
    load_error: str | None = None
    threshold: float = 0.5


class FasterWhisperRunner:
    def __init__(self, spec: dict[str, Any], manifest_dir: Path, allow_downloads: bool):
        try:
            from faster_whisper import WhisperModel
        except ImportError as error:
            raise RuntimeError(
                "faster-whisper is not installed; install the audio extra"
            ) from error

        model_name = str(spec["model"])
        model_path = Path(model_name).expanduser()
        if not model_path.is_absolute():
            model_path = (manifest_dir / model_path).resolve()
        if model_path.exists():
            model_source = str(model_path)
        elif allow_downloads:
            model_source = model_name
        else:
            raise FileNotFoundError(
                "model %r is not a local path; pass --allow-downloads to use "
                "a faster-whisper model ID" % model_name
            )

        download_root = spec.get("download_root")
        if download_root:
            download_root = str(_manifest_path(download_root, manifest_dir))

        self.beam_size = int(spec.get("beam_size", 5))
        if self.beam_size < 1:
            raise ValueError("beam_size must be greater than zero")
        self.language = spec.get("language")
        self.vad_filter = bool(spec.get("vad_filter", False))
        self.model = WhisperModel(
            model_source,
            device=str(spec.get("device", "cpu")),
            compute_type=str(spec.get("compute_type", "int8")),
            download_root=download_root,
            cpu_threads=int(spec.get("cpu_threads", 0)),
            num_workers=int(spec.get("num_workers", 1)),
        )

    def predict(self, sample: AudioSample) -> Prediction:
        options: dict[str, Any] = {
            "beam_size": self.beam_size,
            "vad_filter": self.vad_filter,
        }
        if self.language:
            options["language"] = str(self.language)
        segments, _info = self.model.transcribe(str(sample.path), **options)
        transcript = " ".join(segment.text.strip() for segment in segments).strip()
        return Prediction(
            rating=scan_audio_transcript(transcript),
            transcript=transcript,
        )


class AudioEventsRunner:
    def __init__(self, spec: dict[str, Any]):
        self.threshold = _probability(spec.get("threshold", 0.5), "threshold")

    def predict(self, sample: AudioSample) -> Prediction:
        events = audio_scanner.detect_audio_events(str(sample.path))
        if events is None:
            return Prediction(status="not_implemented")
        scores = {
            str(name).strip().casefold(): _probability(score, "event score")
            for name, score in events
        }
        return Prediction(event_scores=scores)


class VocalizationRunner:
    def __init__(self, spec: dict[str, Any]):
        self.threshold = _probability(spec.get("threshold", 0.5), "threshold")

    def predict(self, sample: AudioSample) -> Prediction:
        score = audio_scanner.detect_sexual_vocalizations(str(sample.path))
        if score is None:
            return Prediction(status="not_implemented")
        return Prediction(vocalization_score=_probability(score, "vocalization score"))


def _manifest_path(value: str, manifest_dir: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (manifest_dir / path).resolve()


def _probability(value: Any, label: str) -> float:
    try:
        probability = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("%s must be a number between 0 and 1" % label) from error
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("%s must be between 0 and 1" % label)
    return probability


def _optional_rating(value: str, row_number: int) -> int | None:
    if not value.strip():
        return None
    try:
        rating = int(value)
    except ValueError as error:
        raise ValueError("row %d: expected_rating must be an integer" % row_number) from error
    if not 1 <= rating <= 10:
        raise ValueError("row %d: expected_rating must be from 1 to 10" % row_number)
    return rating


def _optional_bool(value: str, column: str, row_number: int) -> bool | None:
    normalized = value.strip().casefold()
    if not normalized:
        return None
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    raise ValueError("row %d: %s must be true or false" % (row_number, column))


def _optional_duration(value: str, row_number: int) -> float | None:
    if not value.strip():
        return None
    try:
        duration = float(value)
    except ValueError as error:
        raise ValueError("row %d: duration_sec must be numeric" % row_number) from error
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("row %d: duration_sec must be positive" % row_number)
    return duration


def load_samples(labels_path: Path) -> list[AudioSample]:
    """Load audio paths and ground truth from a CSV beside the audio corpus."""
    samples = []
    with labels_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "path" not in reader.fieldnames:
            raise ValueError("labels CSV must have a 'path' column")

        for row_number, row in enumerate(reader, start=2):
            path_value = (row.get("path") or "").strip()
            if not path_value:
                raise ValueError("row %d: path is required" % row_number)
            path = Path(path_value).expanduser()
            if not path.is_absolute():
                path = (labels_path.parent / path).resolve()
            if not path.is_file():
                raise ValueError("row %d: audio file does not exist: %s" % (row_number, path))

            expected_events_value = (row.get("expected_events") or "").strip()
            if not expected_events_value:
                expected_events = None
            elif expected_events_value.casefold() in {"none", "-"}:
                expected_events = frozenset()
            else:
                expected_events = frozenset(
                    event.strip().casefold()
                    for event in expected_events_value.split(";")
                    if event.strip()
                )

            reference = (row.get("reference_transcript") or "").strip()
            samples.append(
                AudioSample(
                    path=path,
                    expected_rating=_optional_rating(
                        row.get("expected_rating") or "", row_number
                    ),
                    expected_vocalization=_optional_bool(
                        row.get("expected_vocalization") or "",
                        "expected_vocalization",
                        row_number,
                    ),
                    expected_events=expected_events,
                    reference_transcript=reference or None,
                    duration_sec=_optional_duration(
                        row.get("duration_sec") or "", row_number
                    ),
                )
            )

    if not samples:
        raise ValueError("labels CSV contains no audio samples")
    return samples


def load_models(manifest_path: Path, allow_downloads: bool = False) -> list[ModelOption]:
    config = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if not isinstance(config, dict):
        raise ValueError("model manifest must contain a JSON object")
    specs = config.get("models")
    if not isinstance(specs, list) or not specs:
        raise ValueError("model manifest must contain a non-empty 'models' list")

    options = []
    names = set()
    for spec in specs:
        if not isinstance(spec, dict) or not spec.get("name"):
            raise ValueError("every model entry must have a name")
        name = str(spec["name"])
        if name in names:
            raise ValueError("model names must be unique: %s" % name)
        names.add(name)

        kind = spec.get("kind")
        if kind == "faster-whisper":
            if not spec.get("model"):
                raise ValueError("%s: faster-whisper entries require a model" % name)
            tier = "tier1_stt_lexicon"
            constructor = lambda: FasterWhisperRunner(
                spec, manifest_path.parent, allow_downloads
            )
        elif kind in {"audio-events", "panns"}:
            tier = "tier2_audio_events"
            constructor = lambda: AudioEventsRunner(spec)
        elif kind == "sexual-vocalization":
            tier = "tier3_vocalization"
            constructor = lambda: VocalizationRunner(spec)
        else:
            raise ValueError("unsupported audio model kind %r" % kind)

        started = time.perf_counter()
        try:
            runner = constructor()
            load_error = None
        except Exception as error:
            runner = None
            load_error = "%s: %s" % (type(error).__name__, error)
        load_ms = (time.perf_counter() - started) * 1000.0
        threshold = _probability(spec.get("threshold", 0.5), "threshold")
        options.append(
            ModelOption(
                name=name,
                kind=str(kind),
                tier=tier,
                runner=runner,
                load_ms=load_ms,
                load_error=load_error,
                threshold=threshold,
            )
        )
    return options


def word_error_counts(reference: str, hypothesis: str) -> tuple[int, int]:
    """Return word-level edit errors and reference word count."""
    tokenize = lambda text: re.findall(r"\b[\w']+\b", text.casefold())
    reference_words = tokenize(reference)
    hypothesis_words = tokenize(hypothesis)
    if not reference_words:
        return len(hypothesis_words), 0

    previous = list(range(len(reference_words) + 1))
    for row_index, hypothesis_word in enumerate(hypothesis_words, start=1):
        current = [row_index]
        for column_index, reference_word in enumerate(reference_words, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column_index] + 1,
                    previous[column_index - 1]
                    + (hypothesis_word != reference_word),
                )
            )
        previous = current
    return previous[-1], len(reference_words)


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _binary_metrics(pairs: list[tuple[bool, bool]]) -> dict[str, Any]:
    true_positive = sum(expected and predicted for expected, predicted in pairs)
    false_positive = sum(not expected and predicted for expected, predicted in pairs)
    true_negative = sum(not expected and not predicted for expected, predicted in pairs)
    false_negative = sum(expected and not predicted for expected, predicted in pairs)
    precision = _ratio(true_positive, true_positive + false_positive)
    recall = _ratio(true_positive, true_positive + false_negative)
    specificity = _ratio(true_negative, true_negative + false_positive)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
        "accuracy": _ratio(
            true_positive + true_negative,
            true_positive + false_positive + true_negative + false_negative,
        ),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "balanced_accuracy": (
            (recall + specificity) / 2
            if recall is not None and specificity is not None
            else None
        ),
    }


def _event_metrics(
    rows: list[dict[str, Any]], threshold: float
) -> dict[str, Any] | None:
    labeled = [
        row
        for row in rows
        if row["status"] == "completed" and row["expected_events"] is not None
        and row["event_scores"] is not None
    ]
    if not labeled:
        return None

    classes = set()
    sample_sets = []
    false_positives = 0
    for row in labeled:
        expected = set(row["expected_events"])
        predicted = {
            label
            for label, score in row["event_scores"].items()
            if score >= threshold
        }
        classes.update(expected)
        classes.update(predicted)
        sample_sets.append((expected, predicted))
        false_positives += len(predicted - expected)

    per_class = {}
    for event in sorted(classes):
        metrics = _binary_metrics(
            [(event in expected, event in predicted) for expected, predicted in sample_sets]
        )
        per_class[event] = metrics

    f1_values = [
        metrics["f1"] for metrics in per_class.values() if metrics["f1"] is not None
    ]
    return {
        "threshold": threshold,
        "labeled_samples": len(labeled),
        "macro_f1": statistics.mean(f1_values) if f1_values else None,
        "false_positives": false_positives,
        "false_positives_per_hour": (
            false_positives
            / (sum(row["duration_sec"] for row in labeled) / 3600.0)
            if all(row["duration_sec"] is not None for row in labeled)
            and sum(row["duration_sec"] for row in labeled) > 0
            else None
        ),
        "per_class": per_class,
    }


def _average_precision(rows: list[dict[str, Any]]) -> float | None:
    labeled = [
        (bool(row["expected_vocalization"]), float(row["vocalization_score"]))
        for row in rows
        if row["expected_vocalization"] is not None
        and row["vocalization_score"] is not None
    ]
    positive_count = sum(expected for expected, _score in labeled)
    if not positive_count:
        return None

    ranked = sorted(labeled, key=lambda item: item[1], reverse=True)
    true_positive = 0
    false_positive = 0
    previous_recall = 0.0
    area = 0.0
    index = 0
    while index < len(ranked):
        score = ranked[index][1]
        tied = []
        while index < len(ranked) and ranked[index][1] == score:
            tied.append(ranked[index])
            index += 1
        true_positive += sum(expected for expected, _score in tied)
        false_positive += sum(not expected for expected, _score in tied)
        recall = true_positive / positive_count
        precision = true_positive / (true_positive + false_positive)
        area += (recall - previous_recall) * precision
        previous_recall = recall
    return area


def _threshold_rows(
    options: list[ModelOption], rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    results = []
    for option in options:
        model_rows = [row for row in rows if row["model"] == option.name]
        if option.tier == "tier3_vocalization":
            labeled = [
                row
                for row in model_rows
                if row["expected_vocalization"] is not None
                and row["vocalization_score"] is not None
            ]
            if not labeled:
                continue
            for threshold in THRESHOLDS:
                metrics = _binary_metrics(
                    [
                        (
                            bool(row["expected_vocalization"]),
                            row["vocalization_score"] >= threshold,
                        )
                        for row in labeled
                    ]
                )
                results.append(
                    {
                        "model": option.name,
                        "tier": option.tier,
                        "threshold": threshold,
                        **metrics,
                    }
                )
    return results


def _summarize(
    option: ModelOption,
    rows: list[dict[str, Any]],
    unsafe_threshold: int,
) -> dict[str, Any]:
    model_rows = [row for row in rows if row["model"] == option.name]
    completed = [row for row in model_rows if row["status"] == "completed"]
    latencies = [
        row["inference_ms"]
        for row in completed
        if row["inference_ms"] is not None
    ]
    first_latency = next(
        (row["inference_ms"] for row in model_rows if row["inference_ms"] is not None),
        None,
    )
    labeled_ratings = [
        row for row in model_rows if row["expected_rating"] is not None
    ]
    rated = [
        row
        for row in labeled_ratings
        if row["status"] == "completed" and row["predicted_rating"] is not None
    ]
    unsafe_metrics = _binary_metrics(
        [
            (
                row["expected_rating"] >= unsafe_threshold,
                row["predicted_rating"] >= unsafe_threshold,
            )
            for row in rated
        ]
    )
    rating_errors = [
        abs(row["predicted_rating"] - row["expected_rating"]) for row in rated
    ]
    exact_ratings = sum(
        row["predicted_rating"] == row["expected_rating"] for row in rated
    )

    transcript_rows = [
        row
        for row in completed
        if row["reference_transcript"] is not None and row["transcript"] is not None
    ]
    word_errors = 0
    reference_words = 0
    for row in transcript_rows:
        errors, words = word_error_counts(
            row["reference_transcript"], row["transcript"]
        )
        word_errors += errors
        reference_words += words

    event_metrics = _event_metrics(model_rows, option.threshold)
    vocalization_rows = [
        row
        for row in model_rows
        if row["status"] == "completed"
        and row["expected_vocalization"] is not None
        and row["vocalization_score"] is not None
    ]
    vocalization_metrics = _binary_metrics(
        [
            (
                bool(row["expected_vocalization"]),
                row["vocalization_score"] >= option.threshold,
            )
            for row in vocalization_rows
        ]
    )
    duration_known = bool(completed) and all(
        row["duration_sec"] is not None for row in completed
    )
    duration_sec = (
        sum(row["duration_sec"] for row in completed) if duration_known else 0.0
    )
    inference_sec = sum(latencies) / 1000.0
    failed_count = sum(row["status"] == "failed" for row in model_rows)
    not_implemented_count = sum(
        row["status"] == "not_implemented" for row in model_rows
    )
    if not_implemented_count == len(model_rows):
        status = "not_implemented"
    elif failed_count and completed:
        status = "partial_failure"
    elif failed_count:
        status = "failed"
    else:
        status = "completed"

    return {
        "model": option.name,
        "kind": option.kind,
        "tier": option.tier,
        "status": status,
        "load_ms": round(option.load_ms, 2),
        "load_error": option.load_error,
        "samples_total": len(model_rows),
        "samples_completed": len(completed),
        "samples_failed": failed_count,
        "samples_not_implemented": not_implemented_count,
        "first_sample_latency_ms": (
            round(first_latency, 2) if first_latency is not None else None
        ),
        "mean_latency_ms": round(statistics.mean(latencies), 2) if latencies else None,
        "p50_latency_ms": round(_percentile(latencies, 50), 2) if latencies else None,
        "p95_latency_ms": round(_percentile(latencies, 95), 2) if latencies else None,
        "audio_duration_sec": round(duration_sec, 3) if duration_known else None,
        "real_time_factor": (
            round(inference_sec / duration_sec, 4)
            if duration_known and duration_sec
            else None
        ),
        "samples_per_second": (
            round(len(completed) / inference_sec, 3) if inference_sec else None
        ),
        "rating_labeled": len(labeled_ratings),
        "rating_scored": len(rated),
        "rating_coverage": (
            round(len(rated) / len(labeled_ratings), 4) if labeled_ratings else None
        ),
        "unsafe_threshold": unsafe_threshold,
        "unsafe_metrics": unsafe_metrics,
        "rating_mean_absolute_error": (
            round(statistics.mean(rating_errors), 3) if rating_errors else None
        ),
        "rating_exact_accuracy": (
            round(exact_ratings / len(rated), 4) if rated else None
        ),
        "transcripts_with_reference": len(transcript_rows),
        "word_errors": word_errors,
        "reference_words": reference_words,
        "word_error_rate": (
            round(word_errors / reference_words, 4) if reference_words else None
        ),
        "event_metrics": event_metrics,
        "vocalization_labeled": len(vocalization_rows),
        "vocalization_threshold": option.threshold,
        "vocalization_metrics": (
            vocalization_metrics if vocalization_rows else None
        ),
        "vocalization_pr_auc": _average_precision(model_rows),
    }


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (
        position - lower
    )


def evaluate(
    options: list[ModelOption],
    samples: list[AudioSample],
    output_dir: Path,
    unsafe_threshold: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if not samples:
        raise ValueError("no labeled audio samples were selected")
    if not 1 <= unsafe_threshold <= 10:
        raise ValueError("unsafe rating threshold must be from 1 to 10")

    rows = []
    for option in options:
        for sample in samples:
            prediction = Prediction(status="failed", error=option.load_error)
            inference_ms = None
            if option.runner is not None:
                started = time.perf_counter()
                try:
                    prediction = option.runner.predict(sample)
                except Exception as error:
                    prediction = Prediction(
                        status="failed",
                        error="%s: %s" % (type(error).__name__, error),
                    )
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                if prediction.status == "completed":
                    inference_ms = elapsed_ms

            word_error_rate = None
            if sample.reference_transcript is not None and prediction.transcript is not None:
                errors, reference_words = word_error_counts(
                    sample.reference_transcript, prediction.transcript
                )
                word_error_rate = (
                    errors / reference_words if reference_words else None
                )

            event_scores = prediction.event_scores
            detected_events = (
                sorted(
                    event
                    for event, score in event_scores.items()
                    if score >= option.threshold
                )
                if event_scores is not None
                else None
            )
            rows.append(
                {
                    "model": option.name,
                    "kind": option.kind,
                    "tier": option.tier,
                    "sample": str(sample.path),
                    "status": prediction.status,
                    "expected_rating": sample.expected_rating,
                    "predicted_rating": prediction.rating,
                    "expected_unsafe": (
                        sample.expected_rating >= unsafe_threshold
                        if sample.expected_rating is not None
                        else None
                    ),
                    "predicted_unsafe": (
                        prediction.rating >= unsafe_threshold
                        if prediction.rating is not None
                        else None
                    ),
                    "expected_vocalization": sample.expected_vocalization,
                    "vocalization_score": prediction.vocalization_score,
                    "expected_events": (
                        sorted(sample.expected_events)
                        if sample.expected_events is not None
                        else None
                    ),
                    "event_scores": event_scores,
                    "detected_events": detected_events,
                    "reference_transcript": sample.reference_transcript,
                    "word_error_rate": word_error_rate,
                    "transcript": prediction.transcript,
                    "duration_sec": sample.duration_sec,
                    "load_ms": round(option.load_ms, 2),
                    "inference_ms": (
                        round(inference_ms, 2) if inference_ms is not None else None
                    ),
                    "error": prediction.error,
                }
            )

    summaries = [
        _summarize(option, rows, unsafe_threshold) for option in options
    ]
    threshold_results = _threshold_rows(options, rows)
    _write_results(rows, summaries, threshold_results, output_dir, unsafe_threshold)
    return rows, summaries, threshold_results


def _write_results(
    rows: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    threshold_results: list[dict[str, Any]],
    output_dir: Path,
    unsafe_threshold: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    per_sample_columns = [
        "model",
        "kind",
        "tier",
        "sample",
        "status",
        "expected_rating",
        "predicted_rating",
        "expected_unsafe",
        "predicted_unsafe",
        "expected_vocalization",
        "vocalization_score",
        "expected_events",
        "event_scores",
        "detected_events",
        "word_error_rate",
        "duration_sec",
        "load_ms",
        "inference_ms",
        "error",
    ]
    with (output_dir / "per_sample.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=per_sample_columns)
        writer.writeheader()
        for row in rows:
            output = {key: row.get(key) for key in per_sample_columns}
            for key in ("expected_events", "event_scores", "detected_events"):
                if output[key] is not None:
                    output[key] = json.dumps(output[key], sort_keys=True)
            writer.writerow(output)

    summary_columns = [
        "model",
        "kind",
        "tier",
        "status",
        "load_ms",
        "samples_total",
        "samples_completed",
        "samples_failed",
        "samples_not_implemented",
        "first_sample_latency_ms",
        "mean_latency_ms",
        "p50_latency_ms",
        "p95_latency_ms",
        "audio_duration_sec",
        "real_time_factor",
        "samples_per_second",
        "rating_labeled",
        "rating_scored",
        "rating_coverage",
        "unsafe_threshold",
        "unsafe_accuracy",
        "unsafe_precision",
        "unsafe_recall",
        "unsafe_f1",
        "unsafe_false_positives",
        "unsafe_false_negatives",
        "rating_mean_absolute_error",
        "rating_exact_accuracy",
        "transcripts_with_reference",
        "word_error_rate",
        "event_macro_f1",
        "event_false_positives_per_hour",
        "vocalization_labeled",
        "vocalization_threshold",
        "vocalization_accuracy",
        "vocalization_precision",
        "vocalization_recall",
        "vocalization_f1",
        "vocalization_pr_auc",
        "load_error",
    ]
    with (output_dir / "summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_columns)
        writer.writeheader()
        for summary in summaries:
            unsafe = summary["unsafe_metrics"]
            vocalization = summary["vocalization_metrics"] or {}
            events = summary["event_metrics"] or {}
            writer.writerow(
                {
                    **{key: summary.get(key) for key in summary_columns},
                    "unsafe_accuracy": unsafe["accuracy"],
                    "unsafe_precision": unsafe["precision"],
                    "unsafe_recall": unsafe["recall"],
                    "unsafe_f1": unsafe["f1"],
                    "unsafe_false_positives": unsafe["false_positive"],
                    "unsafe_false_negatives": unsafe["false_negative"],
                    "vocalization_accuracy": vocalization.get("accuracy"),
                    "vocalization_precision": vocalization.get("precision"),
                    "vocalization_recall": vocalization.get("recall"),
                    "vocalization_f1": vocalization.get("f1"),
                    "event_macro_f1": events.get("macro_f1"),
                    "event_false_positives_per_hour": events.get(
                        "false_positives_per_hour"
                    ),
                }
            )

    with (output_dir / "thresholds.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        columns = [
            "model",
            "tier",
            "threshold",
            "true_positive",
            "false_positive",
            "true_negative",
            "false_negative",
            "accuracy",
            "precision",
            "recall",
            "specificity",
            "f1",
            "balanced_accuracy",
        ]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(threshold_results)

    report = {
        "unsafe_rating_threshold": unsafe_threshold,
        "sample_count": len({row["sample"] for row in rows}),
        "model_count": len(summaries),
        "transcripts_written": False,
        "results": summaries,
        "thresholds": threshold_results,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare configured audio tiers on the same local labeled corpus; "
            "transcripts are never saved."
        )
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--unsafe-rating-threshold",
        type=int,
        default=7,
        help="ratings at or above this value count as unsafe (default: 7)",
    )
    parser.add_argument(
        "--allow-downloads",
        action="store_true",
        help="allow faster-whisper model IDs to download weights",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 1 <= args.unsafe_rating_threshold <= 10:
        parser.error("--unsafe-rating-threshold must be from 1 to 10")

    try:
        samples = load_samples(args.labels.resolve())
        options = load_models(args.manifest.resolve(), args.allow_downloads)
        _rows, summaries, _threshold_results = evaluate(
            options,
            samples,
            args.output_dir.resolve(),
            args.unsafe_rating_threshold,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))

    print("model | tier | status | load/first ms | p50/p95 ms | RTF | unsafe F1 | WER")
    for summary in summaries:
        metrics = summary["unsafe_metrics"]
        f1 = metrics["f1"]
        print(
            "%s | %s | %s | %s/%s | %s/%s | %s | %s | %s"
            % (
                summary["model"],
                summary["tier"],
                summary["status"],
                _format(summary["load_ms"]),
                _format(summary["first_sample_latency_ms"]),
                _format(summary["p50_latency_ms"]),
                _format(summary["p95_latency_ms"]),
                _format(summary["real_time_factor"]),
                _format(f1),
                _format(summary["word_error_rate"]),
            )
        )
    print("Wrote per-sample, summary, threshold, and JSON results to %s" % args.output_dir)
    has_unimplemented = any(
        summary["status"] == "not_implemented" for summary in summaries
    )
    has_failures = any(
        summary["status"] in {"failed", "partial_failure"}
        for summary in summaries
    )
    if has_unimplemented:
        print("Some tiers are not implemented yet; inspect report.json for their status.")
    if has_failures:
        print("Model failures occurred; inspect summary.csv and report.json.")
        return 1
    return 0


def _format(value: float | None) -> str:
    return "n/a" if value is None else "%.3f" % value


if __name__ == "__main__":
    raise SystemExit(main())
