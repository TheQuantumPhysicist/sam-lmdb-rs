# sam-lmdb-rs

Idiomatic and safe APIs for interacting with the
[Symas Lightning Memory-Mapped Database (LMDB)](https://symas.com/lmdb/).

This repo is a fork of [mozilla/lmdb-rs](https://github.com/mozilla/lmdb-rs)
with fixes for issues encountered when using it.

## Building from Source

```bash
git clone https://github.com/TheQuantumPhysicist/sam-lmdb-rs
cd sam-lmdb-rs
cargo build --release
```

## lmdb source

The lmdb source code is copied from: https://github.com/monero-project/monero

## Features

* [x] lmdb-sys.
* [x] Cursors.
* [x] Zero-copy put API.
* [x] Nested transactions.
* [x] Database statistics.
