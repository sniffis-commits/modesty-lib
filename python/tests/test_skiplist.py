import os
import tempfile
import unittest

from nsfw_scanner.skiplist import (
    entries_from_result,
    generated_skipfile,
    parse_line,
    read_skiplist,
    write_result_skiplist,
    write_skiplist,
)


class SkiplistDurationTests(unittest.TestCase):
    def test_write_result_skiplist_persists_entries_from_scan_timeline(self):
        result = {
            "duration_sec": 4.0,
            "fps": 1.0,
            "sha256": "abc123",
            "frame_ratings": [
                {"timecode": 0.0, "rating": 1, "class": "neutral"},
                {"timecode": 1.0, "rating": 7, "class": "sexy"},
                {"timecode": 2.0, "rating": 1, "class": "neutral"},
            ],
        }

        with tempfile.TemporaryDirectory() as directory:
            video_path = os.path.join(directory, "video.mp4")
            user_skip_path = os.path.join(directory, "video.skp")
            with open(user_skip_path, "w", encoding="utf-8") as skip_file:
                skip_file.write("user skip data")

            saved = write_result_skiplist(video_path, result)

            self.assertEqual(saved["path"], generated_skipfile(video_path))
            self.assertEqual(saved["entry_count"], 1)
            self.assertEqual(read_skiplist(saved["path"])[0]["time"], "00:00:01.000")
            with open(user_skip_path, encoding="utf-8") as skip_file:
                self.assertEqual(skip_file.read(), "user skip data")

    def test_write_result_skiplist_persists_empty_scan_result(self):
        with tempfile.TemporaryDirectory() as directory:
            video_path = os.path.join(directory, "video.mp4")

            saved = write_result_skiplist(
                video_path,
                {"duration_sec": 2.0, "fps": 1.0, "frame_ratings": []},
            )

            self.assertTrue(os.path.isfile(saved["path"]))
            self.assertEqual(saved["entry_count"], 0)
            self.assertEqual(read_skiplist(saved["path"]), [])

    def test_video_intervals_end_at_the_next_sampled_rating_one_frame(self):
        result = {
            "scanner": "media-video",
            "duration_sec": 10.0,
            "fps": 2.0,
            "frame_ratings": [
                {"timecode": 0.0, "rating": 1, "class": "neutral"},
                {"timecode": 1.0, "rating": 7, "class": "sexy"},
                {"timecode": 1.5, "rating": 5, "class": "neutral"},
                {"timecode": 2.0, "rating": 2, "class": "synthetic"},
                {"timecode": 2.25, "rating": None, "class": None},
                {"timecode": 2.5, "rating": 1, "class": "neutral"},
                {"timecode": 3.0, "rating": 8, "class": "sexy"},
                {"timecode": 3.5, "rating": 5, "class": "neutral"},
                {"timecode": 4.0, "rating": 1, "class": "neutral"},
                {"timecode": 5.0, "rating": 6, "class": "sexy"},
                {"timecode": 6.0, "rating": 5, "class": "neutral"},
            ],
        }

        entries = entries_from_result(result)

        self.assertEqual(len(entries), 3)
        self.assertEqual(entries[0]["time"], "00:00:01.000")
        self.assertAlmostEqual(entries[0]["duration"], 1.5)
        self.assertEqual(entries[1]["time"], "00:00:03.000")
        self.assertAlmostEqual(entries[1]["duration"], 1.0)
        self.assertEqual(entries[2]["time"], "00:00:05.000")
        self.assertAlmostEqual(entries[2]["duration"], 5.0)

    def test_safe_sample_splits_same_action_intervals(self):
        result = {
            "scanner": "media-video",
            "duration_sec": 6.0,
            "fps": 1.0,
            "frame_ratings": [
                {"timecode": 1.0, "rating": 7, "class": "sexy"},
                {"timecode": 2.0, "rating": 1, "class": "neutral"},
                {"timecode": 3.0, "rating": 7, "class": "sexy"},
                {"timecode": 4.0, "rating": 1, "class": "neutral"},
            ],
        }

        entries = entries_from_result(result)

        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0]["time"], "00:00:01.000")
        self.assertAlmostEqual(entries[0]["duration"], 1.0)
        self.assertEqual(entries[1]["time"], "00:00:03.000")
        self.assertAlmostEqual(entries[1]["duration"], 1.0)

    def test_frame_timeline_hits_do_not_require_problem_list_entries(self):
        result = {
            "scanner": "media-video",
            "duration_sec": 4.0,
            "fps": 1.0,
            "problems": [],
            "frame_ratings": [
                {"timecode": 1.0, "rating": 7, "class": "sexy"},
                {"timecode": 2.0, "rating": 1, "class": "neutral"},
            ],
        }

        entries = entries_from_result(result)

        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["time"], "00:00:01.000")
        self.assertAlmostEqual(entries[0]["duration"], 1.0)

    def test_malformed_timeline_rows_are_ignored(self):
        result = {
            "scanner": "media-video",
            "duration_sec": 4.0,
            "fps": 1.0,
            "problems": [],
            "frame_ratings": [
                None,
                {"timecode": "bad", "rating": 1},
                {"timecode": 1.0, "rating": 7, "class": "sexy"},
                {"timecode": float("nan"), "rating": 1},
                {"timecode": 2.0, "rating": "unknown"},
                {"timecode": 3.0, "rating": 1},
            ],
        }

        entries = entries_from_result(result)

        self.assertEqual(len(entries), 1)
        self.assertAlmostEqual(entries[0]["duration"], 2.0)

    def test_same_action_intervals_merge_only_when_they_overlap(self):
        result = {
            "scanner": "media-video",
            "duration_sec": 3605.0,
            "fps": 1.0,
            "frame_ratings": [
                {"timecode": 3599.0, "rating": 7, "class": "sexy"},
                {"timecode": 3600.0, "rating": 7, "class": "sexy"},
                {"timecode": 3601.0, "rating": 1, "class": "neutral"},
                {"timecode": 3602.0, "rating": 7, "class": "sexy"},
                {"timecode": 3603.0, "rating": 1, "class": "neutral"},
            ],
        }

        entries = entries_from_result(result)

        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0]["time"], "00:59:59.000")
        self.assertAlmostEqual(entries[0]["duration"], 2.0)
        self.assertEqual(entries[1]["time"], "01:00:02.000")
        self.assertAlmostEqual(entries[1]["duration"], 1.0)

    def test_every_frame_uses_native_frame_rate(self):
        result = {
            "scanner": "media-video",
            "duration_sec": 3.0,
            "every_frame": True,
            "native_fps": 25.0,
            "problems": [
                {"kind": "frame", "timecode": 1.0, "rating": 10, "class": "porn"},
            ],
        }

        entries = entries_from_result(result)

        self.assertEqual(len(entries), 1)
        self.assertAlmostEqual(entries[0]["duration"], 0.04)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "video.skp")
            write_skiplist(path, entries)
            with open(path, encoding="utf-8") as skip_file:
                written = next(line for line in skip_file if "[duration=" in line)
        self.assertAlmostEqual(parse_line(written)["duration"], 0.04)

    def test_audio_subtitle_hit_writes_a_timed_mute_entry(self):
        result = {
            "scanner": "media-audio",
            "duration_sec": 60.0,
            "problems": [
                {
                    "kind": "subtitle",
                    "timecode": 4.25,
                    "timecode_end": 8.75,
                    "rating": 10,
                    "keyword": "hit",
                },
            ],
        }
        entries = entries_from_result(result)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["actions"], ["mute"])
        self.assertAlmostEqual(entries[0]["duration"], 4.5)

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "audio.skp")
            write_skiplist(path, entries, header={"duration": result["duration_sec"]})
            with open(path, encoding="utf-8") as skip_file:
                content = skip_file.read()

        self.assertIn("duration: 60.0", content)
        parsed_line = next(line for line in reversed(content.splitlines()) if line)
        parsed = parse_line(parsed_line)
        self.assertEqual(parsed["time"], "00:00:04.250")
        self.assertEqual(parsed["actions"], ["mute"])
        self.assertAlmostEqual(parsed["duration"], 4.5)

    def test_legacy_result_without_frame_timeline_uses_sampling_interval(self):
        result = {
            "scanner": "media-video",
            "problems": [
                {"kind": "frame", "timecode": 5.0, "rating": 6, "class": "sexy"},
            ],
        }

        entries = entries_from_result(result)

        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["duration"], 1.0)


if __name__ == "__main__":
    unittest.main()
