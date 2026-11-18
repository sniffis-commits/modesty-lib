# Modesty - nsfw protection toolkit (Python library)

Shippable library only: no CLI, no web code, no frontend. Rates websites,
image URLs, image files **and media files** (video/audio) 1–10 for adult
content.

```python
from nsfw_scanner import scan, rating_phrase

result = scan("https://example.com", details=True, debug=True)
print(result["rating"], rating_phrase(result["rating"]))
```

## Layout

- `src/nsfw_scanner/scanner.py` — the web + image logic (one module, dependency-light).
- `src/nsfw_scanner/badwords.py` — DNS badwords: rates a hostname from its
  name alone (10 strong / 7 suggestive / None) before any fetch or model
  load; masks innocent look-alikes (`essex`, `analysis`, `peacock`, …).
- `src/nsfw_scanner/lexicons.py` — the configurable sensuality lexicon.
- `src/nsfw_scanner/media.py` — the media scanner (video frames, audio
  profile, file-type detection). **Needs ffmpeg + ffprobe.**
- `src/nsfw_scanner/subtitles.py` — SRT / VTT / ASS parsing and lookup.
- `src/nsfw_scanner/data/lexicons/*.txt` — the lexicon files (editable).
- `src/nsfw_scanner/models/` — bundled ONNX weights (nsfw joint + binary,
  gender, face).
- `src/nsfw_scanner/__init__.py` — public API surface.

## The two NSFW models

| model | file | size | speed | grades |
|---|---|---|---|---|
| **joint** (5 classes) | `nsfw_joint_model.onnx` | 85.2 MB | ~32 ms/frame | 1 · 2 · 5 · **6** · 7 · 10 |
| **binary** (2 classes) | `nsfw_model.onnx` | 22.5 MB | ~7 ms/frame | 1 · 2 · 5 · 7 · 10 |

The joint model decides by its strongest class (`nsfw_scanner.NSFW_CLASSES`):
`porn`/`hentai` → 10, `sexy` → **6 (romance)**, `drawings`/`neutral` →
nothing on their own (the gender model may still add 5). The class vector is
part of the evidence, so the reason reads “the model called the frame 204 at
03:24 ‘porn’ (0.88)”.

The binary model is the **fast path** — old phones, small servers, and the
Android app until the distilled student exists (see the ROADMAP). It has no
romance grade: it rates 10 / 7 / 5 / 1 by threshold.

```bash
nsfw-scan clip.mp4 --model joint    # the accurate path (the default)
nsfw-scan clip.mp4 --model binary   # ~4x faster, no romance grade
NSFW_MODEL_MODE=binary nsfw-scan clip.mp4
```

```python
from nsfw_scanner import set_model_mode, model_mode
set_model_mode("binary")   # returns the previous mode
```

## The rating scale

| rating | meaning |
|---|---|
| `1` | no unsafe signal detected in the material that was successfully scanned; not a safety guarantee |
| `2` | synthetic content (a 3D render / a screenshot / a drawing) — the picture model is trained on photographs, so it is **not rated** |
| `5` | women/girls present; no stronger unsafe signal detected |
| `6` | romance — clothed, affectionate (the model calls it `sexy`) |
| `7` | revealing, not fully explicit |
| `10` | explicit adult content |

The picture model decides by its **strongest class** (`nsfw_scanner.NSFW_CLASSES`):
`porn`/`hentai` → explicit, `sexy` → 6, `drawings`/`neutral` → nothing on their own
(the gender model may still add 5). The class vector is part of the evidence,
so the reason reads “the model called the frame 204 at 03:24 ‘porn’ (0.88)”.

**A rating of 10 needs a second opinion.** The joint model's `porn` class is
a *skin* detector, not an “explicit” verdict — measured on real frames it
fires at 0.48 on a man singing with his neck exposed, at 0.92 on a *wedding*
and at 0.77 on a rabbi's white beard. The **binary** model (the old 2-class
one, ~7 ms/frame) says 0.07, 0.01 and 0.03 on exactly those frames. So:

| rating | what it takes |
|---|---|
| `10` | the binary model agrees (`nsfw > 0.5`), or the class is `hentai` (an explicit drawing) |
| `7` | a strong skin hypothesis (`NSFW_JOINT_STRONG`, 0.8) **and** the binary model sees something too (`> 0.2`) |
| `6` | the class is `sexy` (romance) **or** a strong skin hypothesis the binary model does not confirm — a wedding or a dancer is not an act |
| `5` / `1` | a woman is present (the gender model) / nothing at all |

If the binary model is missing, the conservative old rule applies — “we did
not look” must never mean “safe”.

**One model instead of two** — the distillation
(**tools-internal-staging/distill/**) trains a small MobileNetV2 with **two
heads** (the five classes + the binary score), so one file replaces both:
`distill.sh` runs the whole pipeline (fetch → label → train → export →
validate) and the library picks the student up with
`NSFW_MODEL=nsfw_student.onnx`. The rating rule does not change.

**The sample frames** (`SAMPLES/` at the repo root) are the frames the fast
tests scan — cut out of the real test videos, the name carrying the
expected rating (`testJewish_frame54_48_6.png` = the wedding, expected 6).
They are the good cases *and* the known bad ones (the exposed neck, the
wedding, the rabbi’s beard), so the suite runs in seconds and the model’s
behaviour is pinned.

## Install / build

```bash
pip install ./packages/nsfw-lib        # from the repo root
python -m build packages/nsfw-lib     # wheel includes the models/
```

The wheel distribution is `modesty_nsfw_scanner`; the Python import remains
`nsfw_scanner` for compatibility. PyPI names are reserved by publishing the
distribution, not by defining a Python import package.

Model lookup order: `NSFW_MODEL_DIR` env var → bundled `models/` → cwd.
Without the primary image model, suspicious text/URL keywords can still
produce a positive rating, but image-only targets are not assessed (rating
`None`), rather than being reported as rating 1. Rating 1 means a successful
model scan found no unsafe signal; it is not a safety guarantee.

## High-accuracy image scanning (`high_accuracy=True`)

The fast scanner uses global image classification: it detects that skin is
present, but not *where* or *how much* is exposed. This causes
**middle-tier over-protection** — revealing/risque/lingerie images (expected
rating 6 or 7) get bumped to explicit (10) because the joint model's "porn"
class fires strongly on skin regardless of exposure level.

Enable `high_accuracy=True` to run YOLOv8 (`yolo_best.pt`) after the fast
scanner. YOLO detects localized body parts with covered vs exposed state:

| YOLO detection | Rating effect |
|---|---|
| Exposed genitalia (female/male) | → 10 (explicit) |
| Exposed breasts or buttocks | → at least 7 (risque) |
| Only covered body parts | → capped at 7 (never explicit) |
| No body parts found | → scanner rating unchanged |

**Performance:** ~570ms per image vs ~112ms fast path (~5x slower).
Use high-accuracy for middle-tier disambiguation (6 vs 7 vs 10); use the
fast path for bulk scanning where over-protection to 10 is acceptable.

```python
from nsfw_scanner import analyze_pil_image, get_sessions, get_binary_session
from PIL import Image

nsfw_sess, gender_sess = get_sessions()
binary_sess = get_binary_session()
img = Image.open("photo.jpg")

# Fast: ~112ms, may over-protect middle tiers to 10
result = analyze_pil_image(img, nsfw_sess, gender_sess,
                           binary_session=binary_sess, details=True)

# High accuracy: ~570ms, YOLO refines 6 vs 7 vs 10
result = analyze_pil_image(img, nsfw_sess, gender_sess,
                           binary_session=binary_sess,
                           details=True, high_accuracy=True)
# result["rating"]        -> YOLO-refined rating
# result["high_accuracy"] -> True
# result["yolo_refined"]  -> YOLO-adjusted rating
```

Requires `ultralytics` (`pip install ultralytics`) and the bundled
`___TESTS/yolo_best.pt` model. Without either, `high_accuracy=True`
silently falls back to the fast path.

## Badwords (cheap first pass)

```python
from nsfw_scanner import domain_badwords, badword_rating, is_bad_domain

badword_rating("pornhub.com")   # 10
badword_rating("example.com")   # None
is_bad_domain("xxx-videos.net")  # True
```

`scan()` runs this check first by default (`use_badwords=True`); pass
`use_badwords=False` (CLI: `--no-badwords`) to always fetch and classify.
For local files the *file name* is checked (`pornhub_clip.mp4` → 10).

## Match before scanning

The CLI asks what we already know before it scans anything:

```bash
nsfw-scan clip.mp4 --db packages/nsfw-db/data/nsfw.sqlite   # local DB
nsfw-scan clip.mp4 --hash-list hashes.txt                    # offline hash list
nsfw-scan example.com --api-url https://api.example --api-key nsfw_live_…
nsfw-scan example.com --ignore-db                           # ignore everything
```

| source | flag | cost |
|---|---|---|
| local product DB (domain → `known_domains`, file → sha256 → `media_hashes`) | `--db` (auto-found) | free, offline |
| hash list file (one sha256 per line, optional reason) | `--hash-list` | free, offline |
| hosted API (`?p=api&action=lookup` / `action=hash`) | `--api-url` + `--api-key` | **paid / higher tier**, opt-in |

A match returns the stored verdict (`matched: true`, `match_source`,
`reason`) instead of scanning again. `--ignore-db` skips every source and
scans the target physically.

```python
from nsfw_scanner import KnownTargets

known = KnownTargets(db_path="nsfw.sqlite", hash_list="hashes.txt")
hit = known.lookup("https://example.com")   # -> {"rating": 10, "reason": ...}
```

## The sensuality lexicon (configurable)

The terms that decide a text or subtitle verdict live in plain text files:

```
packages/nsfw-lib/src/nsfw_scanner/data/lexicons/sensuality.txt
packages/nsfw-lib/src/nsfw_scanner/data/lexicons/sensuality_safe.txt
```

| line | meaning |
|---|---|
| `porn` | word-boundary match (default) — never fires on "Capricorn" |
| `~pornhub` | substring match — only for long, unambiguous terms |
| `-cams` | drop a built-in term |
| `# …` | comment |

`sensuality_safe.txt` lists look-alike words that are masked before the
substring matching ("analysis" contains "anal", "essex" contains "sex").

Point **`NSFW_LEXICON_DIR`** at your own directory to use your own files
(the built-in terms stay the base). `nsfw_scanner.sensuality_lexicon()`
shows what is in effect; the files are re-read when they change.

```python
from nsfw_scanner import sensuality_lexicon, find_sensuality_hits

sensuality_lexicon()["terms"][:5]
find_sensuality_hits("visit pornhub.com")   # ['pornhub']
find_sensuality_hits("essex county")        # []
```

## Local model evaluation

`tools/evaluate_models.py` runs local ONNX, Hugging Face image classifiers,
optional TensorFlow/Keras models, and Llama Guard 4 Vision side-by-side on
labeled sample PNGs, individual images, and video frames.
It reports each model's load time, first-call time, per-frame p50/p95
inference time, and writes raw per-frame scores to CSV. It does not download
models, make network requests, or combine model outputs into a verdict.
When a classifier exposes an `nsfw` score, or both `porn` and `hentai`
classes, and labeled images are included, it also prints a threshold grid
for the coarse expected-rating `>=7` boundary (ratings 1–6 are negative; 7
and 10 are positive). The five-class path uses `porn + hentai` for this
comparison. This is an offline comparison only; it does not change scanner
thresholds or production ratings.

Create a local manifest based on
`tools/model-evaluation.example.json`, point its model paths at weights that
you have already obtained and license-checked, then run from the repository
root:

For `tensorflow-keras` entries, install TensorFlow and TensorFlow Hub in the
evaluation environment; these are optional and are not runtime dependencies
of `nsfw-scanner`. Keras model output values are recorded as returned without
applying another softmax.

Llama Guard 4 Vision uses the separate
`tools/model-evaluation.llama-guard.example.json` manifest. Its locally
licensed weights must already be present; the adapter uses
`local_files_only=True` and sends no images over the network. The model emits
a categorical `safe`/`unsafe` verdict and hazard IDs, not a calibrated NSFW
probability, so those results are recorded as categorical labels and are not
included in the numeric threshold sweep. Llama Guard is a large multimodal
moderation model, not a lightweight client candidate.

```powershell
python packages/nsfw-lib/tools/evaluate_models.py `
  --manifest C:\path\to\local-models.json `
  --samples-dir SAMPLES `
  --video SAMPLES\Shakira.webm `
  --video SAMPLES\indian.webm `
  --video SAMPLES\indian2.webm `
  --fps 1 `
  --output (Join-Path $env:TEMP 'nsfw-model-evaluation.csv')
```

Only labeled sample filenames ending in `_1.png` through `_10.png` are read
from `--samples-dir`; video frames are extracted to temporary storage and
removed after evaluation. Hugging Face models must be present locally; the
Transformers adapter uses `local_files_only=True`.

## Audio model evaluation (in progress)

`tools/evaluate_audio.py` runs every configured option against the same
locally labeled audio corpus. The example manifest lists the standard
faster-whisper size/language variants plus the PANNs event and vocalization
tiers. Whisper entries can also compare CPU/GPU devices and compute types.
Model IDs download weights only
when `--allow-downloads` is explicitly supplied; local model directories need
no network access.

Install the optional transcription dependency and prepare a CSV beside the
audio files:

```powershell
python -m pip install "packages/nsfw-lib[audio]"
```

The CSV requires `path` and accepts `expected_rating` (1–10),
`reference_transcript`, `expected_events` (semicolon-separated; use `none`
when the clip was checked and has no target event), `expected_vocalization`
(`true`/`false`), and `duration_sec`. Leave a label blank when it is unknown;
do not treat an unknown label as a negative example. Ratings 7–10 count as
unsafe by default.

```powershell
python packages/nsfw-lib/tools/evaluate_audio.py `
  --manifest packages/nsfw-lib/tools/audio-model-evaluation.example.json `
  --labels C:\audio-eval\labels.csv `
  --output-dir C:\audio-eval\results `
  --allow-downloads
```

The output includes per-sample timing, a summary with load and first-sample
times, p50/p95 latency, real-time factor, unsafe precision/recall/F1, rating
error, transcript word-error rate, event macro-F1/false positives per hour,
and vocalization precision-recall results. Audio transcripts and reference
text are not written to the reports. Timestamp error is deferred until the
transcriber preserves segment times. The event and vocalization scanners are
still scaffolds, so their rows are explicitly `not_implemented` and do not
receive fabricated accuracy scores. A rights-cleared labeled corpus is
required before reporting real accuracy.

## Media scanner (video + audio)

Video and audio files are **never** routed through the website path:
`rate()` dispatches by the file's *header* (the extension is only a
claim, and a mismatch is flagged).

```bash
# a local video: frames sampled at 1 fps + the subtitles
nsfw-scan "D:\Videos\My Clip.mp4" --details

# every frame — for hunting hidden messages
nsfw-scan clip.mp4 --every-frame --max-frames 5000

# denser sampling, or a local image
nsfw-scan clip.mp4 --fps 0.5
nsfw-scan photo.jpg --details
```

```python
from nsfw_scanner import rate, scan_media, detect_kind

detect_kind("clip.mp4")
# {'kind': 'video', 'mime': 'video/mp4', 'mime_sniffed': 'video/mp4',
#  'mime_claimed': None, 'ext': '.mp4', 'container': 'mp4',
#  'type_mismatch': False}

result = rate("clip.mp4", details=True)
result["rating"]        # 1..10
result["problems"]      # where it stood out
```

The answer looks like this:

```python
{'scanner': 'media-video', 'rating': 2,
 'phrase': 'synthetic - not rated, scan it yourself',
 'media_type': 'video', 'mime': 'video/mp4', 'duration_sec': 797.6,
 'frame_count': 798, 'problem_count': 6,
 'problems': [{'kind': 'frame', 'timecode': 676.0, 'frame_index': 676, 'score': 0.3029}, …]}
```

* **Frames** are sampled (`fps=1` by default, one frame per second) and
  rated with the same models the image scanner uses. `--every-frame`
  decodes every frame (for hunting subliminal content); `--fps 0.5`
  scans denser. Every frame above `min_report_score` is reported with
  its timestamp and score.
* **Detected frames are saved** beside the video: the original is written
  under `SAMPLES/<video>.frame.<time>.png`, and its blurred preview as
  `<video>.frame.<time>-blurred.png`; a matching JSON sidecar in `SAMPLES/`
  stores the rating, frame index, and timestamp. The Python API can disable
  this with `scan_video(..., save_samples=False)`.
* **The frames come out of ffmpeg as raw RGB** (`--frames raw`, the
  default) — no PNG encode, no PNG decode: the extraction is 13 ms per
  frame instead of 19 ms, and the pixels are **bit-identical** to the
  PNG path (so the verdict does not change). `--frames png` uses the
  old path, which is also the automatic fallback.
* **Two frame sources**: `--decoder ffmpeg` (the external binary, the
  default) or `--decoder av` (PyAV, the optional `av` package — libav
  in-process, no pipe at all). PyAV decodes 2.7× faster but the scan
  ends up ~7 % slower (the external ffmpeg overlaps with the model
  better) and its bundled FFmpeg decodes a little differently, so it is
  an alternative, not a default. Its real value: **no ffmpeg binary
  needed**.
* **The file takes its worst frame** — never an average: one explicit
  frame makes the whole file explicit. The deciding frame is named in the
  reason (its index, its timecode, its class and its probability).
* **Sub-crops are *not* scored by default** (`tiles="off"`): a tight crop
  of a face or a hand reads as *porn 0.97* to both models, so the tiles
  invent explicit verdicts out of innocent close-ups (a smiling man: 0.93
  as a whole frame → 0.97 on the best tile). `--tiles 3x3` / `2x2` stays
  available for hunting hidden messages, where a false positive costs
  nothing.
* **Subtitles** (embedded tracks and sidecar `.srt`/`.vtt`/`.ass`) are
  parsed and scanned; a hit is a problem area with the line number and
  timecode. A subtitle hit is a text verdict, so it stands on its own.
* **Skip files** (`--skip-file`) start an interval at each sampled frame that
  meets `--skip-threshold` (6 by default) and end it at the next later sampled
  frame rated exactly 1; that safe frame is excluded. If no later safe frame
  is found, the interval ends at media EOF. Same-action overlapping intervals
  merge, but intervals are never merged across a known safe sample. At reduced
  scan FPS, frames between samples are not individually judged, so the safe
  boundary is the next *sampled* safe frame, not a frame-perfect guarantee.
  Subtitle hits use their cue start/end times; audio files can therefore
  produce timed mute entries for subtitle hits. Audio classification remains
  a placeholder: the acoustic profile is informational and does not detect
  spoken content.
* **Synthetic content** (a 3D render, a drawing, a UI screenshot) is
  detected and rated **2 — "not rated, scan it yourself"**: the picture
  model is trained on photographs, so on such content its score does not
  track nudity. Two signals must agree: the file is **noise-free**
  (camera footage has sensor noise) **and** the face detector finds
  **no real face** in the samples (it is trained on photographed faces, so
  a drawn or rendered face is not found — a music video's singer is found
  in every frame). The flagged frames are still listed with their
  scores. Thresholds (all configurable, hard-coded defaults):
  `NSFW_SYNTHETIC_SAMPLES` (24), `NSFW_SYNTHETIC_NOISE_PCTL` (25),
  `NSFW_SYNTHETIC_NOISE_MAX` (0), `NSFW_SYNTHETIC_RATIO` (0.7),
  `NSFW_SYNTHETIC_FACE_RATIO` (0.2), `NSFW_SYNTHETIC_EXPLICIT` (0.85),
  `NSFW_SYNTHETIC_REVEALING` (0.5), `NSFW_SYNTHETIC_FLAT_MIN` (0.35).
* **The face detector is optional** (and deprecated as a default): it
  lives in its own importable module (`nsfw_scanner.faces`) and the core
  calls the **same functions** — `detect_faces(img, face_session) ->
  [(x1, y1, x2, y2, prob), ...]` — so any callable with that signature
  can be plugged in (`face_detector=`). It costs 14–36 ms per frame and
  **the file ratings do not change without it** — only the fine-grained
  "5" grades get coarser (the gender model then runs on the whole frame
  instead of a face crop) — so the media path does not use it per frame
  (13–18 % faster), while the synthetic gate still uses it (a couple of
  dozen samples over the whole file) and the single-image path keeps it.
  **One switch, no knobs**: `--face-crop` turns the per-frame face crop
  back on.
* **Audio** has no classifier yet: the scan reports the acoustic profile
  (loudness, speech likelihood) and the subtitle verdict, and leaves the
  rating alone. The comparative evaluation harness is being prepared; see
  `tools/evaluate_audio.py` and the audio tiers in the ROADMAP.
* **ffmpeg + ffprobe are required.** Without them the scanner raises
  `MediaDependencyError` with the fix in the message — it never guesses.
  Override the binaries with `NSFW_FFMPEG` / `NSFW_FFPROBE`.
* **Hardware decoding** (`--hwaccel auto|cuda|qsv|d3d11va|vaapi`): “auto”
  tries the hardware decoders in order and falls back to the software one
  when the hardware refuses the file. Measured on a GTX 980 + NVDEC: the
  file verdict is the same, but the hardware frames are **not
  bit-identical** to the software ones (≤31/255 of rounding) and 3 of 357
  borderline frames flip their class — and at 1 fps it is only ~2 %
  faster, because the model is the bottleneck. It earns its keep in the
  decode-heavy modes (`--every-frame`, 4K). The default is `none`.

### Benchmarking it yourself (CPU vs GPU, model vs model)

`--benchmark` **runs the scan twice** — once with the software decoder,
once with the GPU — and prints the comparison (stderr; stdout stays a
clean dict). `--benchmark-model` runs **all four** combinations: both
decoders × both models (the joint 5-class one and the fast binary one).

```bash
nsfw-scan "clip.mp4" --benchmark          # CPU vs GPU           (2 scans)
nsfw-scan "clip.mp4" --benchmark-model    # CPU/GPU × joint/binary (4 scans)
```

The known-target cache is bypassed for the benchmark (otherwise there
would be nothing to time) — the file is really scanned every time.

Real output from this repo (`test2Hindi.mp4`, 5:57, 357 frames at 1 fps,
GTX 980 + NVDEC):

```
benchmark (backend: auto, models: joint) - test2Hindi.mp4 — 05:57.01 long, 8.1 MB
  2 run(s): none/joint, auto/joint
  ------------------------------------------------------------------------------
  run                   total     loop  inference   ms/frame frames/s x realtime rating flags
  none / joint (cpu)    49.0s    39.0s      33.2s      92.98     7.28     7.388     10   148
  auto / joint (cuda)   49.7s    40.0s      33.7s      94.54     7.19     7.384     10   148
  ------------------------------------------------------------------------------
  fastest: none / joint (49.0s)
  every run agreed on the verdict (rating 10)
  note: the hardware frames are not bit-identical to the software ones,
        so a borderline frame can flip its class.
```

Each run shows the **progress bar** while it runs, labelled with the
model and the decoder it is using (`scanning clip.mp4 [joint · cpu] —
05:57.01 long, 8.1 MB, 357 frames sampled of 25 …`), and the table
repeats the decoder that was *actually* used in brackets — `auto` may
have fallen back to the software decoder, and the table says so. The
**× realtime** column is the file's own length against the wall time:
how many times the file's duration the scan covered.

The benchmark does **not** print the results (they are huge): it prints
the table and a hint. To keep them, write them to a file:

```bash
nsfw-scan clip.mp4 --benchmark --output bench.json
```

With `--benchmark-model` you get the 2×2 matrix (the `flags` column is
the number of problem frames). This is a real run — but the box was
under load at the time, which is why one line is an outlier (see the
caveat below):

```
benchmark (models: joint, binary) - test2Hindi.mp4
  4 run(s): none/joint, auto/joint, none/binary, auto/binary
  ------------------------------------------------------------------------------
  run                   total     loop  inference   ms/frame frames/s rating flags
  none / joint          47.7s    37.5s      31.5s      88.28     7.49     10   148
  auto / joint          66.2s    56.7s      50.1s     140.44     5.39     10   148
  none / binary         49.2s    39.8s      33.7s      94.38     7.26     10   148
  auto / binary         48.7s    39.3s      33.0s      92.43     7.33     10   148
  ------------------------------------------------------------------------------
  fastest: none / joint (47.7s)
  every run agreed on the verdict (rating 10)
```

Note how the *model* choice moves the numbers less than you would
expect: the binary model is 3–4× faster per frame (7 ms vs 32 ms), but
it leaves more frames undecided, and **every undecided frame then pays
for the face detector and the gender model** — so on a file full of
borderline frames the "fast" model can end up nearly as slow. That is
the real lesson of the benchmark: the picture model is not the only
cost, and the pipeline can be re-balanced (skip the face detector when
the picture model already decided, or run the gender model on the whole
frame instead of a face crop).

**Why the GPU is so close to the CPU:** the decoder is a small part of
the loop. Of the 39 s the frame loop takes, 33 s is the *model* — the
decode, the PNG encode and the pipe are ~6 s (≈17 ms/frame). The
per-frame cost is the picture model (32–44 ms for the joint one, ~7 ms
for the binary one) **plus the face detector (~22 ms) plus the gender
model (~20 ms)** — the last two only run when the picture model leaves
the rating open. So at 1 fps the GPU can only win a few per cent, and
it earns its keep in the decode-heavy modes (`--every-frame`, 4K),
where the decode dominates. (The next real lever is not the GPU at all
but the **PNG round-trip**: ffmpeg encodes a PNG and PIL decodes it —
a raw-video pipe would take that away.)

Two caveats worth knowing:
* the hardware frames are **not bit-identical** to the software ones
  (≤31/255 of rounding), so a borderline frame can flip its class —
  the file verdict usually holds, the per-frame numbers do not;
* the measurements move when the machine is busy (we measured a 66 s
  outlier while the box was under load) — **run the benchmark on an
  idle machine**.

### Progress (stderr)
The CLI prints the progress to **stderr**, so stdout stays a clean dict:

```
scanning test.mp4 — 13:17.56 long, 797 frames sampled of 30.0 (1.0 of 30.0 fps)
  frames     392/797   49.2%  @ 06:31.00   25.2s elapsed   ETA 26.0s  15.5 frames/s  15.5x realtime
done: rating 2 · 6 problem area(s) · 41.3s
```

`frames/s` is the throughput, `× realtime` is how much video a wall-second
covers, and the header shows the sampling ratio (1 of 30 fps) so the cost
is visible before the scan starts. `--no-progress` silences it.

### Report your own scan back to the site

The CLI prints the **sha256** of the file it scanned — that is the only
thing the site ever needs (never the file itself):

```bash
$ nsfw-scan "D:\Videos\My Clip.mp4" --details
{'scanner': 'media-video', 'rating': 10, 'sha256': 'a167bd08…',
 'problems': [{'kind': 'subtitle', 'timecode': 1.5, 'line_no': 2, …}]}

$ nsfw-scan --version
nsfw-scan 0.2.0
```

If you scanned the file with another tool, hash it yourself:

```bash
certutil -hashfile "D:\Videos\My Clip.mp4" SHA256   # Windows
sha256sum clip.mp4                                 # Linux / macOS
```

Then, on the site (`/?p=scanner`): paste the sha256 into *“Do we know
this file?”* to see what we know about those bytes, and — with a free
account — send your own result in *“Report your own scan”*. The site
keeps the last N results per file and compares each with its own
verdict, which is how the scanner's quality is measured against an
independent scan.

### Through the queue

The queue runs the same scanner in the background (the CLI, not the
website — a public form must not be able to read a file from the host):

```bash
run.bat scan-add --kind media "D:\Videos\My Clip.mp4"
run.bat scan-once --limit 5
```

The website only accepts a **sha256**: “do we know these bytes?” is
answered from the database, and a logged-in user can report their own
scan result there.
