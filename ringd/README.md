# Modesty - nsfw protection ringd

Small Rust daemon (ringd) that sits between the Python scanner and the media
scanner, buffering frames so the image model is not re-decoded per frame.
Used by `nsfw-scan` when hardware decoding or shared-frame transport is
desired.

## Structure

```
packages/nsfw-ringd/
├── Cargo.toml          # Rust dependencies
├── Cargo.lock
└── src/                # Rust daemon source
```

## Build

Requires the Rust stable toolchain for `x86_64-pc-windows-msvc`:

```bash
cargo build --release   # from packages/nsfw-ringd/
```

## Usage

The ringd daemon is optional. The scanner falls back to decoding each frame
fresh when ringd is not running. Enable it with `--hwaccel auto` on the CLI
or via the GUI toggle.

## License

MIT
