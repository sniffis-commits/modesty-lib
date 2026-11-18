import json
import os
import tempfile
import unittest

from PIL import Image, ImageChops

from nsfw_scanner.media import _save_hit_frame


class SaveHitFrameTests(unittest.TestCase):
    def test_saves_original_and_blurred_preview_with_collision_safe_names(self):
        with tempfile.TemporaryDirectory() as directory:
            video_path = os.path.join(directory, "clip.mp4")
            image = Image.new("RGB", (80, 80), "white")
            for x in range(40):
                for y in range(80):
                    image.putpixel((x, y), (0, 0, 0))

            first = _save_hit_frame(image, video_path, 1.25, 4, rating=7)
            self.assertTrue(first["original"].endswith(
                os.path.join("SAMPLES", "clip.frame.00.00.01.250.png")))
            self.assertTrue(first["blurred"].endswith(
                "clip.frame.00.00.01.250-blurred.png"))
            self.assertTrue(first["metadata"].endswith(
                os.path.join("SAMPLES", "clip.frame.00.00.01.250.json")))
            with open(first["metadata"], encoding="utf-8") as metadata_file:
                self.assertEqual(json.load(metadata_file)["rating"], 7)
            self.assertEqual(first["rating"], 7)

            original = Image.open(first["original"]).convert("RGB")
            blurred = Image.open(first["blurred"]).convert("RGB")
            self.assertIsNone(ImageChops.difference(image, original).getbbox())
            self.assertIsNotNone(ImageChops.difference(image, blurred).getbbox())

            second = _save_hit_frame(image, video_path, 1.25, 4, rating=6)
            self.assertTrue(first["original"] != second["original"])
            self.assertTrue(first["blurred"] != second["blurred"])
            self.assertTrue(first["metadata"] != second["metadata"])


if __name__ == "__main__":
    unittest.main()
