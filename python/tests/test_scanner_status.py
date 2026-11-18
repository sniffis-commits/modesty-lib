import os
import tempfile
import unittest
from unittest.mock import patch

from bs4 import BeautifulSoup
from PIL import Image

from nsfw_scanner import scanner as scanner_module
from nsfw_scanner import media as media_module
from nsfw_scanner.media import _media_reason, scan_video
from nsfw_scanner.scanner import (
    _rate_parsed_page,
    analyze_image_file,
    analyze_image,
    analyze_pil_image,
    rating_phrase,
    verdict_for,
)


class ScanStatusTests(unittest.TestCase):
    def test_rating_one_means_no_unsafe_signal_not_safe(self):
        self.assertEqual(rating_phrase(1), "no unsafe signal detected")
        self.assertEqual(verdict_for(1), "modesty-no-signal")

    def test_missing_image_models_are_not_assessed(self):
        result = analyze_pil_image(
            Image.new("RGB", (16, 16), "gray"),
            nsfw_session=None,
            gender_session=None,
            details=True,
            debug=True,
        )

        self.assertIsNone(result["rating"])
        self.assertEqual(result["debug"]["decided_by"], "modesty-not-assessed")

    def test_successful_model_scan_with_no_unsafe_signal_rates_one(self):
        class JointSession:
            _nsfw_kind = "joint"

        classes = {
            "drawings": 0.01,
            "hentai": 0.01,
            "neutral": 0.95,
            "porn": 0.02,
            "sexy": 0.01,
        }
        with patch("nsfw_scanner.scanner.classify_pil_image",
                   return_value=classes), patch(
                       "nsfw_scanner.scanner._run_session", return_value=0.01
                   ):
            result = analyze_pil_image(
                Image.new("RGB", (16, 16), "gray"),
                JointSession(),
                gender_session=None,
                binary_session=object(),
                details=True,
                debug=True,
            )

        self.assertEqual(result["rating"], 1)
        self.assertEqual(result["debug"]["decided_by"], "modesty-no-signal")

    def test_unreadable_image_is_not_assessed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "missing.png")
            result = analyze_image_file(path, None, None, details=True, debug=True)

        self.assertIsNone(result["rating"])
        self.assertIn("image scan failed", result["reason"])

    def test_failed_image_download_is_not_assessed(self):
        with patch("nsfw_scanner.scanner.requests.get",
                   side_effect=OSError("offline")):
            result = analyze_image(
                "https://example.test/frame.jpg",
                None,
                None,
                details=True,
                debug=True,
            )

        self.assertIsNone(result["rating"])
        self.assertIn("image scan failed", result["reason"])

    def test_webpage_with_unassessed_images_is_not_given_rating_one(self):
        soup = BeautifulSoup('<img src="frame.jpg">', "html.parser")
        with patch("nsfw_scanner.scanner.extract_image_urls",
                   return_value=["https://example.test/frame.jpg"]), patch(
                       "nsfw_scanner.scanner.analyze_image", return_value=None
                   ):
            rating, _, _, facts = _rate_parsed_page(
                soup, "https://example.test/", None, None
            )

        self.assertIsNone(rating)
        self.assertEqual(facts["stage"], "modesty-not-assessed")

    def test_unassessed_video_reason_is_explicit(self):
        reason, evidence = _media_reason(None, [])

        self.assertIn("not assessed", reason)
        self.assertEqual(evidence["stage"], "modesty-not-assessed")

    def test_video_with_no_usable_model_output_is_not_assessed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "sample.mp4")
            with open(path, "wb") as handle:
                handle.write(b"\x00\x00\x00\x18ftypisom")

            probe_info = {
                "format": {"duration": "1"},
                "streams": [{
                    "codec_type": "video",
                    "index": 0,
                    "width": 1,
                    "height": 1,
                    "r_frame_rate": "1/1",
                }],
            }
            patches = (
                patch.object(media_module, "require_ffmpeg",
                             return_value={"ffmpeg": "ffmpeg", "ffprobe": "ffprobe"}),
                patch.object(media_module, "probe", return_value=probe_info),
                patch.object(media_module, "_sample_synthetic",
                             return_value={"synthetic": False}),
                patch.object(media_module, "iter_frames", return_value=iter([
                    (0, "00:00:00.0", Image.new("RGB", (1, 1), "gray")),
                ])),
                patch.object(scanner_module, "get_sessions", return_value=(None, None)),
                patch.object(scanner_module, "get_binary_session", return_value=None),
                patch.object(scanner_module, "get_face_session", return_value=None),
                patch.object(scanner_module, "analyze_pil_image_tiles",
                             return_value={"rating": None, "nsfw_score": None,
                                           "class": None}),
            )
            with patches[0], patches[1], patches[2], patches[3], patches[4], \
                    patches[5], patches[6], patches[7]:
                result = scan_video(
                    path,
                    subtitles=False,
                    save_samples=False,
                    include_frame_ratings=True,
                )

        self.assertIsNone(result["rating"])
        self.assertEqual(result["evidence"]["stage"], "modesty-not-assessed")
        self.assertEqual(len(result["frame_ratings"]), 1)
        self.assertIsNone(result["frame_ratings"][0]["rating"])


if __name__ == "__main__":
    unittest.main()
