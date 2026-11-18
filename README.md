<p align="center"><img src="assets/logo.svg" alt="modesty-lib" width="110" height="110"></p>

# modesty-lib

> Core scanning library (Python + Rust)

**Part of the [Modesty toolkit](https://github.com/sniffis-commits/modesty-meta)** — see the
[big README](https://github.com/sniffis-commits/modesty-meta) for the full picture, the rating scale and all repositories.

## What this is

The engine of Modesty. A Python package (`modesty_nsfw_scanner`) that rates websites, image URLs, image files and media files on a 1–10 scale using ONNX models, plus a Rust mirror of the decision logic. This is a library — no CLI, no web server, no UI.

## How to


```bash
pip install ./python                      # from this repo's python/ folder
# or, published:
pip install modesty_nsfw_scanner
```
```python
from nsfw_scanner import scan, rating_phrase
r = scan("photo.jpg", details=True)
print(r["rating"], rating_phrase(r["rating"]))     # e.g. 10 'explicit'
```
- `python/src/nsfw_scanner/scanner.py` — website + image rating.
- `python/src/nsfw_scanner/media.py` — video/audio scan. **Needs ffmpeg + ffprobe** on PATH.
- `python/src/nsfw_scanner/known.py` — match a hash against a local DB / hash list before scanning.
- `rust/`, `ringd/` — Rust core and the shared-memory frame daemon.
Models are **not** bundled in the PyPI wheel; they download on first use from https://github.com/sniffis-commits/modesty-models.


## Layout

| Path | Purpose |
|---|---|
| `python/` | source from the monorepo |
| `rust/` | source from the monorepo |
| `ringd/` | source from the monorepo |

## More

- Full overview and other repos: [modesty-meta](https://github.com/sniffis-commits/modesty-meta)
- Website: [https://modesty.cia.org.il](https://modesty.cia.org.il)
- Model weights: [modesty-models](https://github.com/sniffis-commits/modesty-models)
- Licence: MIT (see `LICENSE`). Third-party data/model terms are on the
  [site's licence page](https://modesty.cia.org.il/?p=licenses).
