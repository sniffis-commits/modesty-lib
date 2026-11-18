import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from nsfw_scanner import scanner


class GenderClassificationTests(unittest.TestCase):
    def test_any_female_crop_wins_even_when_another_crop_has_a_higher_male_score(self):
        joint_session = SimpleNamespace(_nsfw_kind="joint")
        crop_scores = iter([
            (0.65, 0.35),
            (0.10, 0.90),
            (0.20, 0.80),
            (0.25, 0.75),
        ])
        with patch.object(
            scanner,
            "classify_pil_image",
            return_value={
                "drawings": 0.01,
                "hentai": 0.01,
                "neutral": 0.95,
                "porn": 0.02,
                "sexy": 0.01,
            },
        ), patch.object(scanner, "_run_session", return_value=0.01), patch.object(
            scanner, "_run_gender_both", side_effect=lambda *_: next(crop_scores)
        ):
            result = scanner.analyze_pil_image(
                Image.new("RGB", (32, 32)),
                joint_session,
                gender_session=object(),
                binary_session=object(),
                details=True,
            )

        self.assertEqual(result["rating"], 5)
        self.assertEqual(result["gender"], "female")
        self.assertEqual(result["female_score"], 0.65)
        self.assertEqual(result["male_score"], 0.35)

    @unittest.skipUnless(
        scanner.ort is not None
        and (Path(__file__).resolve().parents[1]
             / "src" / "nsfw_scanner" / "models" / "fairface_int8.onnx").exists(),
        "FairFace ONNX model is unavailable",
    )
    def test_fairface_edge_samples_vote_by_class_order_per_crop(self):
        root = Path(__file__).resolve().parents[3]
        samples = {
            "SAMPLES/real_live_data/111.png": False,
            "SAMPLES/real_live_data/222.png": True,
            "SAMPLES/_ANDROID_SAMPLE/nastypeople.jpeg": True,
        }
        with patch.dict(os.environ, {"NSFW_GENDER_MODEL": ""}):
            session = scanner.load_gender_model()

        self.assertEqual(session._nsfw_gender_output_order, ("female", "male"))
        for relative_path, expected_female in samples.items():
            with self.subTest(sample=relative_path):
                image = Image.open(root / relative_path).convert("RGB")
                votes = [
                    female > male
                    for crop in scanner.crop_halves(image)
                    for female, male in [
                        scanner._run_gender_both(
                            session, scanner.preprocess_gender(crop)
                        )
                    ]
                ]
                self.assertEqual(any(votes), expected_female)


if __name__ == "__main__":
    unittest.main()
