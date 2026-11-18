import unittest
from unittest.mock import patch

from nsfw_scanner.audio import scan_audio, scan_audio_transcript


class AudioScaffoldTests(unittest.TestCase):
    def test_unavailable_audio_models_remain_unassessed(self):
        with patch("nsfw_scanner.audio._WHISPER_AVAILABLE", False):
            result = scan_audio("missing-audio.wav")

        self.assertIsNone(result["rating"])
        self.assertEqual(result["tiers"]["stt_lexicon"]["status"], "unavailable")
        self.assertEqual(result["tiers"]["audio_events"]["status"], "not_implemented")
        self.assertEqual(
            result["tiers"]["panns_vocalization_head"]["status"],
            "not_implemented",
        )

    def test_completed_transcription_with_no_hits_is_rating_one(self):
        with patch("nsfw_scanner.audio._WHISPER_AVAILABLE", True), patch(
            "nsfw_scanner.audio.transcribe_audio", return_value=""
        ):
            result = scan_audio("sample.wav")

        self.assertEqual(result["rating"], 1)
        self.assertEqual(result["tiers"]["stt_lexicon"]["status"], "completed")

    def test_transcript_scoring_uses_shared_lexicon_matching(self):
        self.assertEqual(scan_audio_transcript("explicit porn"), 10)
        self.assertEqual(scan_audio_transcript("analysis"), 1)
        self.assertEqual(
            scan_audio_transcript(
                "romance",
                {"terms": ["romance"], "substring_terms": [], "safe_mask": []},
            ),
            6,
        )


if __name__ == "__main__":
    unittest.main()
