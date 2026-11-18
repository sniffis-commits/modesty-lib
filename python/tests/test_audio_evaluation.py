import csv
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

from evaluate_audio import (  # noqa: E402
    evaluate,
    load_models,
    load_samples,
    main,
    word_error_counts,
)


class AudioEvaluationTests(unittest.TestCase):
    def test_word_error_counts_use_word_edits(self):
        self.assertEqual(word_error_counts("the cat sat", "the dog sat"), (1, 3))
        self.assertEqual(word_error_counts("", "extra words"), (2, 0))

    def test_unimplemented_scaffold_tiers_do_not_fail_the_cli(self):
        summary = {
            "model": "PANNs",
            "tier": "tier2_audio_events",
            "status": "not_implemented",
            "load_ms": 0,
            "first_sample_latency_ms": None,
            "p50_latency_ms": None,
            "p95_latency_ms": None,
            "real_time_factor": None,
            "unsafe_metrics": {"f1": None},
            "word_error_rate": None,
        }
        with (
            patch("evaluate_audio.load_samples", return_value=[object()]),
            patch("evaluate_audio.load_models", return_value=[]),
            patch("evaluate_audio.evaluate", return_value=([], [summary], [])),
            redirect_stdout(io.StringIO()),
        ):
            result = main(
                [
                    "--manifest",
                    "models.json",
                    "--labels",
                    "labels.csv",
                    "--output-dir",
                    "results",
                ]
            )
        self.assertEqual(result, 0)

    def test_load_samples_resolves_paths_and_parses_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "clip.wav"
            audio.touch()
            labels = root / "labels.csv"
            with labels.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "path",
                        "expected_rating",
                        "expected_vocalization",
                        "expected_events",
                        "reference_transcript",
                        "duration_sec",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "path": "clip.wav",
                        "expected_rating": "10",
                        "expected_vocalization": "true",
                        "expected_events": "Moan; speech",
                        "reference_transcript": "explicit words",
                        "duration_sec": "2.5",
                    }
                )

            sample = load_samples(labels)[0]

            self.assertEqual(sample.path, audio)
            self.assertEqual(sample.expected_rating, 10)
            self.assertTrue(sample.expected_vocalization)
            self.assertEqual(sample.expected_events, frozenset({"moan", "speech"}))
            self.assertEqual(sample.reference_transcript, "explicit words")
            self.assertEqual(sample.duration_sec, 2.5)

    def test_evaluation_compares_whisper_and_marks_scaffold_tiers_unimplemented(self):
        class FakeWhisperModel:
            def __init__(self, model, **kwargs):
                self.model = model
                self.options = kwargs

            def transcribe(self, audio_path, **kwargs):
                return iter([SimpleNamespace(text="explicit porn")]), None

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "unsafe.wav"
            audio.touch()
            labels = root / "labels.csv"
            with labels.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "path",
                        "expected_rating",
                        "reference_transcript",
                        "expected_events",
                        "expected_vocalization",
                        "duration_sec",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "path": "unsafe.wav",
                        "expected_rating": "10",
                        "reference_transcript": "explicit porn",
                        "expected_events": "moan",
                        "expected_vocalization": "true",
                        "duration_sec": "3",
                    }
                )
            manifest = root / "models.json"
            manifest.write_text(
                json.dumps(
                    {
                        "models": [
                            {
                                "name": "Whisper tiny CPU",
                                "kind": "faster-whisper",
                                "model": "tiny",
                            },
                            {"name": "PANNs", "kind": "audio-events"},
                            {
                                "name": "Vocalization head",
                                "kind": "sexual-vocalization",
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )

            with patch.dict(
                sys.modules,
                {"faster_whisper": SimpleNamespace(WhisperModel=FakeWhisperModel)},
            ):
                options = load_models(manifest, allow_downloads=True)
                rows, summaries, threshold_rows = evaluate(
                    options, load_samples(labels), root / "results", 7
                )

            self.assertEqual(rows[0]["status"], "completed", rows[0]["error"])
            self.assertEqual(rows[0]["predicted_rating"], 10)
            self.assertEqual(rows[0]["word_error_rate"], 0)
            self.assertEqual(summaries[0]["unsafe_metrics"]["recall"], 1)
            self.assertEqual(summaries[0]["word_error_rate"], 0)
            self.assertIsNotNone(summaries[0]["first_sample_latency_ms"])
            self.assertEqual(summaries[1]["status"], "not_implemented")
            self.assertEqual(summaries[2]["status"], "not_implemented")
            self.assertEqual(threshold_rows, [])
            report = json.loads((root / "results" / "report.json").read_text())
            self.assertFalse(report["transcripts_written"])
            sample_results = root / "results" / "per_sample.csv"
            with sample_results.open(newline="", encoding="utf-8") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 3)
            for result_file in (root / "results").iterdir():
                if result_file.is_file():
                    contents = result_file.read_text(encoding="utf-8")
                    self.assertNotIn("explicit porn", contents)


if __name__ == "__main__":
    unittest.main()
