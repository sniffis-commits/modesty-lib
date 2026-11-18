<p align="center">
  <img src="assets/logo.svg" alt="Modesty" width="110" height="110">
</p>

<p align="center">
  <img src="assets/modesty-brand.svg" alt="Modesty — NSFW protection" width="260">
</p>

# modesty-lib

> Core scanning library (Python + Rust)

The Modesty scanner core. `python/` is the `modesty_nsfw_scanner` Python package (images, websites, video, audio, rated 1–10 with ONNX). `rust/` mirrors the core in Rust; `ringd/` is the shared-memory frame ring daemon. Models are **not** bundled — downloaded on first use from `modesty-models`.

## Layout

| Path | What |
|---|---|
| `python/` | source from the monorepo |
| `rust/` | source from the monorepo |
| `ringd/` | source from the monorepo |

## Details

| | |
|---|---|
| Language | Python + Rust |
| License | MIT (code) · models Apache-2.0 / CC |
| Website | [https://modesty.cia.org.il](https://modesty.cia.org.il) |
| Repository map | [REPO_MAP.md](https://github.com/sniffis-commits/nfsw_scanner/blob/main/docs/REPO_MAP.md) |
| Split from | private monorepo `sniffis-commits/nfsw_scanner` |

## Quick start

```sh
pip install ./python
```

## About Modesty

Privacy-first NSFW scanner. Local scanning by default; uploads are opt-in.
These public repos are split from a private monorepo — see the
[repository map](https://github.com/sniffis-commits/nfsw_scanner/blob/main/docs/REPO_MAP.md) for the full local-to-GitHub naming. Model weights are
in `modesty-models`; the free domain DB release is in `modesty-db`.

Third-party data/model terms are on the [site's license page](https://modesty.cia.org.il/?p=licenses).
