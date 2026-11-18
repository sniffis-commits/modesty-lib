import sys
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

from evaluate_models import (  # noqa: E402
    NUDENET_LABELS,
    LlamaGuardVisionClassifier,
    TensorFlowKerasClassifier,
    collect_labeled_images,
    binary_threshold_grid,
    expected_rating,
    nudenet_scores,
    softmax,
)


class ModelEvaluationTests(unittest.TestCase):
    def test_softmax_is_stable_and_normalized(self):
        probabilities = softmax(np.array([1000.0, 1001.0]))
        self.assertAlmostEqual(float(probabilities.sum()), 1.0)
        self.assertGreater(float(probabilities[1]), float(probabilities[0]))

    def test_expected_rating_is_read_only_from_rating_suffix(self):
        self.assertEqual(expected_rating(Path("wedding_frame54_48_6.png")), 6)
        self.assertIsNone(expected_rating(Path("frame_00000008.png")))
        self.assertEqual(expected_rating(Path("explicit_10.png")), 10)
        self.assertIsNone(expected_rating(Path("frame_10.jpg")))

    def test_labeled_sample_collection_skips_unlabeled_and_blurred_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "expected_7.png").touch()
            (root / "plain.png").touch()
            (root / "preview_7-blurred.png").touch()
            (root / "nested").mkdir()
            (root / "nested" / "explicit_10.png").touch()
            self.assertEqual(
                collect_labeled_images(root),
                [
                    (root / "expected_7.png", 7),
                    (root / "nested" / "explicit_10.png", 10),
                ],
            )

    def test_nudenet_output_maps_each_detected_class_to_its_maximum(self):
        output = np.zeros((1, len(NUDENET_LABELS) + 4, 3), dtype=np.float32)
        output[0, 4, 0] = 0.4
        output[0, 4, 1] = 0.8
        output[0, 4 + 3, 2] = 0.9
        scores = nudenet_scores(output)
        self.assertAlmostEqual(scores["FEMALE_GENITALIA_COVERED"], 0.8)
        self.assertAlmostEqual(scores["FEMALE_BREAST_EXPOSED"], 0.9)
        self.assertEqual(scores["ANUS_EXPOSED"], 0.0)

    def test_binary_threshold_grid_uses_labeled_6_vs_7_boundary(self):
        rows = [
            {"model": "local-vit", "expected_rating": 6, "scores": '{"nsfw": 0.8}'},
            {"model": "local-vit", "expected_rating": 7, "scores": '{"nsfw": 0.7}'},
            {"model": "local-vit", "expected_rating": 10, "scores": '{"nsfw": 0.1}'},
            {"model": "local-vit", "expected_rating": None, "scores": '{"nsfw": 0.99}'},
            {"model": "local-vit", "expected_rating": "", "scores": '{"nsfw": 0.99}'},
        ]
        result = binary_threshold_grid(rows)["local-vit"]
        self.assertEqual(result[0], (0.1, 2, 1, 0, 0))
        self.assertEqual(result[-1], (0.9, 0, 0, 1, 2))

    def test_binary_threshold_grid_uses_explicit_class_sum(self):
        rows = [
            {
                "model": "five-class",
                "expected_rating": 7,
                "scores": '{"porn": 0.3, "hentai": 0.2, "sexy": 0.4}',
            },
            {
                "model": "five-class",
                "expected_rating": 6,
                "scores": '{"porn": 0.1, "hentai": 0.1, "sexy": 0.8}',
            },
        ]
        result = binary_threshold_grid(rows)["five-class"]
        self.assertEqual(result[2], (0.3, 1, 0, 1, 0))

    def test_tensorflow_classifier_uses_keras_image_preprocessing_and_probabilities(self):
        class FakeModel:
            def predict(self, batch, batch_size, verbose):
                self.batch = batch
                self.batch_size = batch_size
                self.verbose = verbose
                return np.array([[0.1, 0.2]], dtype=np.float32)

        classifier = TensorFlowKerasClassifier.__new__(TensorFlowKerasClassifier)
        classifier.name = "fake-keras"
        classifier.input_size = 1
        classifier.labels = ["safe", "nsfw"]
        classifier.model = FakeModel()
        image = Image.new("RGB", (2, 1))
        image.putpixel((0, 0), (0, 0, 0))
        image.putpixel((1, 0), (255, 0, 0))

        scores, _, _ = classifier.predict(image)

        self.assertAlmostEqual(scores["safe"], 0.1)
        self.assertAlmostEqual(scores["nsfw"], 0.2)
        self.assertEqual(classifier.model.batch.shape, (1, 1, 1, 3))
        self.assertEqual(classifier.model.batch_size, 1)
        self.assertEqual(classifier.model.verbose, 0)
        np.testing.assert_array_equal(
            classifier.model.batch[0, 0, 0],
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
        )

    def test_llama_guard_response_records_categorical_verdict_and_hazards(self):
        classifier = LlamaGuardVisionClassifier.__new__(LlamaGuardVisionClassifier)
        classifier.name = "llama-guard-test"
        classifier.device = "cpu"
        classifier.max_new_tokens = 16
        classifier.torch = SimpleNamespace(inference_mode=nullcontext)

        class FakeBatch(dict):
            def to(self, device):
                self.device = device
                return self

        class FakeProcessor:
            def apply_chat_template(self, messages, **kwargs):
                self.messages = messages
                self.kwargs = kwargs
                return FakeBatch(input_ids=np.array([[1, 2, 3]], dtype=np.int64))

            def batch_decode(self, generated, skip_special_tokens):
                self.generated = generated
                self.skip_special_tokens = skip_special_tokens
                return ["unsafe\nS12"]

        class FakeModel:
            def generate(self, **kwargs):
                self.kwargs = kwargs
                return np.array([[1, 2, 3, 4, 5]], dtype=np.int64)

        classifier.processor = FakeProcessor()
        classifier.model = FakeModel()

        scores, _, _ = classifier.predict(Image.new("RGB", (4, 4)))

        self.assertEqual(scores, {"verdict_unsafe": 1.0, "S12": 1.0})
        self.assertEqual(classifier.processor.messages[0]["content"][0]["type"], "image")
        self.assertEqual(classifier.model.kwargs["max_new_tokens"], 16)
        self.assertEqual(
            LlamaGuardVisionClassifier.parse_response("safe"),
            {"verdict_unsafe": 0.0},
        )
        with self.assertRaises(ValueError):
            LlamaGuardVisionClassifier.parse_response("not a guard verdict")


if __name__ == "__main__":
    unittest.main()
