# Audio scanning for the NSFW scanner.
#
# Three tiers, in order of cost:
#   tier 1 - speech-to-text (faster-whisper): transcript -> lexicon scan
#   tier 2 - audio event model (PANNs): gunshots, screams, music, etc.
#   tier 3 - sexual-vocalization detector (direct classifier)
#
# The audio path is a non-APK feature first: its main use is scanning
# offline media (a library of files, a recorder's folder), not the live
# screen.
#
# Online subtitle provider: hash-based lookup (OpenSubtitles-style
# size+md5 scheme) to fetch subtitles for a file that has none.

import os
import hashlib
import logging
import time
from typing import List, Optional, Tuple

_LOGGER = logging.getLogger(__name__)

# Tier 1 is optional; the other two tiers remain explicit scaffolds.
try:
    from faster_whisper import WhisperModel
    _WHISPER_AVAILABLE = True
except ImportError:
    WhisperModel = None
    _WHISPER_AVAILABLE = False


def transcribe_audio(audio_path: str, model_size: str = "base") -> Optional[str]:
    """Tier 1: transcribe audio to text using faster-whisper.

    Returns the transcript text, or None if unavailable.
    """
    if not _WHISPER_AVAILABLE:
        return None
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, _info = model.transcribe(audio_path)
    return " ".join(segment.text.strip() for segment in segments).strip()


def scan_audio_transcript(transcript: str, lexicons: Optional[dict] = None) -> int:
    """Scan a transcript for sensuality keywords.

    Returns a rating (1-10) based on keyword hits.
    """
    if not transcript:
        return 1
    from .scanner import find_sensuality_hits

    hits = find_sensuality_hits(transcript, lexicons)
    if not hits:
        return 1
    romantic_terms = {"love", "romance", "sexy"}
    return 6 if all(hit.casefold() in romantic_terms for hit in hits) else 10


# Tier 2: audio event model (PANNs)
def detect_audio_events(audio_path: str) -> Optional[List[Tuple[str, float]]]:
    """Tier 2 scaffold: return None until file decoding and PANNs are wired."""
    return None


# Tier 3: sexual-vocalization detector (direct classifier)
def detect_sexual_vocalizations(audio_path: str) -> Optional[float]:
    """Tier 3: detect sexual vocalizations.

    Returns a probability (0.0-1.0), or None until the trained head exists.
    """
    return None


# Online subtitle provider (OpenSubtitles-style)
def compute_subtitle_hash(file_path: str) -> str:
    """Compute the OpenSubtitles hash for a file.

    The hash is based on the file size and the first/last 64KB.
    """
    size = os.path.getsize(file_path)
    hash_val = size
    with open(file_path, 'rb') as f:
        # Read first 64KB
        head = f.read(65536)
        for i in range(0, len(head) - 7, 8):
            hash_val += int.from_bytes(head[i:i+8], 'little')
        # Read last 64KB
        if size > 65536:
            f.seek(max(0, size - 65536))
            tail = f.read(65536)
            for i in range(0, len(tail) - 7, 8):
                hash_val += int.from_bytes(tail[i:i+8], 'little')
    return hex(hash_val & 0xFFFFFFFFFFFFFFFF)[2:]


def fetch_subtitles_online(file_path: str, api_key: str = None) -> Optional[str]:
    """Fetch subtitles from an online provider using the file hash.

    Provider, API key and terms are an open decision - the shipped lib
    must not do this implicitly.
    """
    if not api_key:
        return None
    # Placeholder - needs provider integration
    return None


def scan_audio(audio_path: str, lexicons: Optional[dict] = None) -> dict:
    """Run available audio tiers and report independent status and timing.

    Rating 1 is returned only when transcription completed and found no
    unsafe-signal terms. Missing or unimplemented tiers leave the result
    unassessed (rating None).
    """
    result = {
        'transcript': None,
        'events': None,
        'vocalization_score': None,
        'rating': None,
        'tiers': {},
    }

    started = time.perf_counter()
    if not _WHISPER_AVAILABLE:
        result['tiers']['stt_lexicon'] = {
            'status': 'unavailable',
            'elapsed_ms': round((time.perf_counter() - started) * 1000, 2),
            'rating': None,
        }
    else:
        try:
            transcript = transcribe_audio(audio_path)
        except Exception as error:
            _LOGGER.warning("audio transcription failed for %s: %s",
                            audio_path, error)
            result['tiers']['stt_lexicon'] = {
                'status': 'failed',
                'elapsed_ms': round((time.perf_counter() - started) * 1000, 2),
                'rating': None,
                'error': str(error),
            }
        else:
            rating = scan_audio_transcript(transcript or "", lexicons)
            result['transcript'] = transcript
            result['rating'] = rating
            result['tiers']['stt_lexicon'] = {
                'status': 'completed',
                'elapsed_ms': round((time.perf_counter() - started) * 1000, 2),
                'rating': rating,
            }

    started = time.perf_counter()
    result['tiers']['audio_events'] = {
        'status': 'not_implemented',
        'elapsed_ms': round((time.perf_counter() - started) * 1000, 2),
        'rating': None,
    }

    started = time.perf_counter()
    result['tiers']['panns_vocalization_head'] = {
        'status': 'not_implemented',
        'elapsed_ms': round((time.perf_counter() - started) * 1000, 2),
        'rating': None,
    }

    return result
