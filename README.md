<!-- logo -->
<p align="center">
  <img src="https://modesty.eu.org/images/logos/modesty-lib.svg" alt="Modesty" width="96" height="96">
</p>

# modesty-lib

> Core scanning library (Python + Rust)

The Modesty scanner core. Python package `modesty_nsfw_scanner` rates websites, images, video and audio on a 1–10 scale with ONNX models. Rust mirror (`nsfw-lib-rust`) in the same repo, plus the shared-memory frame ring daemon (`nsfw-ringd`).

## Details

| | |
|---|---|
| Language | Python + Rust |
| License | MIT (code) · model weights Apache-2.0 / CC in modesty-models |
| Website | [https://modesty.eu.org](https://modesty.eu.org) |
| Full repository map | [REPO_MAP.md](https://github.com/sniffis-commits/nfsw_scanner/blob/main/docs/REPO_MAP.md) |
| Status | Scaffold — code is being split from the private monorepo |

## Quick start

```sh
pip install modesty_nsfw_scanner
```

## About Modesty

Modesty is a privacy-first NSFW content scanner. Scanning runs locally by
default; uploads are always opt-in. The public packages are split out of the
private monorepo into standalone repositories — see the
[repository map](https://github.com/sniffis-commits/nfsw_scanner/blob/main/docs/REPO_MAP.md) for the full local-to-GitHub naming.

Licensed under MIT (code) · model weights Apache-2.0 / CC in modesty-models. See the website's
[license review](https://modesty.eu.org/?p=licenses) for third-party data and model terms.
