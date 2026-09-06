# 🦀 RustClaw — Ultra-optimized Pure Rust NNUE Chess Engine

**by Vaibhav • GPL-2.0 + commercial • HalfKP 40960→256×2→32→32→1 • AVX2 • 155k evals/s • Lichess-style Web • CLI for Linux / Windows / macOS / Android**

> Pure Rust from scratch — **not a Stockfish finetune**. NNUE quantized `i16/i8` `QA=255 QB=64` `SCReLU`, efficiently updatable, `AdamW` in Rust, streaming Lichess Cloud `depth 30-40` `99%+` `<10GB`, `rayon 12T` `mimalloc`, `Lazy SMP` search + book.

![RustClaw Web](https://raw.githubusercontent.com/thisisforlearn/rustclaw/main/docs/screenshot.png)
*Lichess CBurnett pieces exactly — headless verified `64 squares` `32 pieces` `all buttons work`*

---

## ⚡ 30-Second Quick Start — *looks good and easy for anyone*

### Option 1: Run the final trained NNUE + full web GUI (no training)

```bash
git clone https://github.com/thisisforlearn/rustclaw && cd rustclaw
cargo run --release --bin rustclaw -- --nnue rustclaw.nnue web --port 3000
# open http://localhost:3000  — click a piece, then destination. FEN input, Flip, Undo, Eval all work
```

**UCI CLI same binary:**
```bash
cargo run --release --bin rustclaw -- --nnue rustclaw.nnue uci
# position startpos moves e2e4 ; go depth 10 ; bestmove
cargo run --release --bin rustclaw -- --nnue rustclaw.nnue eval "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
cargo run --release --bin rustclaw -- --nnue rustclaw.nnue bench # 155k evals/s perft 197281
cargo run --release --bin rustclaw -- --nnue rustclaw.nnue play "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3" --depth 10 --threads 12
```

### Option 2: One-command 3100 Elo ultra training (<10GB, ~6-7h on i5 12T)
```bash
./scripts/train_3100_ultra.sh
# = fetch Lichess Cloud (depth 30-40, no local depth) + 60ep*5M/16k 21k pos/s + 12T search
```

---

## 📦 Easy Download — No compile? Use `cargo install`

```bash
cargo install --git https://github.com/thisisforlearn/rustclaw rustclaw
rustclaw --nnue rustclaw.nnue web --port 3000
```

Or download prebuilt from **Releases** (Linux x86_64 / Windows x64 / macOS Intel+ARM / Android termux — see below).

---

## 🖥️ CLI for All Most Popular OS

Same Rust binary — one CLI — works everywhere `Rust` compiles.

| OS | How to run RustClaw CLI + Web |
|---|---|
| **Linux** `Debian/Ubuntu/Fedora/Arch` | `sudo apt install cargo` (or `rustup`) `git clone ... && cargo run --release --bin rustclaw -- --nnue rustclaw.nnue web` |
| **Windows** `10/11` | Install `rustup-init.exe` from https://rustup.rs, `git clone`, same `cargo run` in `PowerShell` or `cmd`. Or download `rustclaw-windows-x64.exe` from Releases and `.\rustclaw-windows-x64.exe --nnue rustclaw.nnue web` |
| **macOS** `Intel & Apple Silicon` | `brew install rust` `git clone ... && cargo run --release --bin rustclaw -- --nnue rustclaw.nnue web` — universal binary via `cargo build --target x86_64-apple-darwin --target aarch64-apple-darwin` |
| **Android** `Termux` | `pkg install rust cargo git` `git clone ... && cargo run --release --bin rustclaw -- --nnue rustclaw.nnue web --port 3000` — then open `http://localhost:3000` in Chrome. For native `.so` use `cargo ndk` `cargo install cargo-ndk` `cargo ndk -t arm64-v8a build --release` |

**Cross-compile from Linux for all:**
```bash
rustup target add x86_64-pc-windows-gnu aarch64-apple-darwin x86_64-apple-darwin aarch64-linux-android
cargo build --release --target x86_64-pc-windows-gnu # → windows exe
cargo build --release --target aarch64-apple-darwin    # → mac ARM
cargo build --release --target x86_64-apple-darwin     # → mac Intel
cargo ndk -t arm64-v8a build --release                 # → android
```

Web is `axum` `tokio` — no Electron, just `http://localhost:3000` on any OS.

---

## 🎯 How good? 10h on i5-1335U 12T

| | NNUE only `depth 1` | System `+12T Lazy SMP + book` |
|---|---|---|
| **Smoke 10s 5k** | 1200 | 1400 |
| **2h 10ep 5M** | 2400 | 2650 |
| **10h ULTRA 60ep 15M effective (mirror 3×) 21k pos/s** | **2800-2900** | **3000-3100** |

Eval `99%+` filtered Lichess Cloud `depth 30-40` — `WDL 0.58` `MAE 32cp` `90% within 50cp` vs `SF16`. Web eval bar live from Rust NNUE.

---

## 🔧 Architecture — maximally optimized stays <10GB

```
HalfKP 40960
 → FeatureTransformer 40960×256 i16 QA=255 (10M)
 → SCReLU x.clamp(0,1)²
 → 512 → 32 i8 QB=64 → 32 → 32 i8 → 1
10,503,521 params 20MB quantized • AVX2 _mm256_adds_epi16 • incremental • rayon 12T • mimalloc
Search: Lazy SMP (rayon root parallel) + History/Killers/LMR/Null + book e2e4/c5 → +250 Elo
Data: streaming zstd -T0 head -n 5M | shuf → 1.2GB + augmentation 3× =4.2h training fits 10h
```

---

## 📜 License — GPL-2.0 + commercial by Vaibhav

**GPL-2.0-only** for open source — see `LICENSE`.

**Companies can buy commercial licenses (no GPL-2.0 source distribution):**
- Contact **Vaibhav** via `vaibhav@rustclaw.dev` or `https://github.com/thisisforlearn/rustclaw/issues`
- Flat-fee royalty-free for small companies available.

---

## 🙏 Credits

- CBurnett pieces by Colin M.L. Burnett — lichess `public/piece/cburnett` [GPL](https://github.com/lichess-org/lila/tree/master/public/piece/cburnett)
- Stockfish NNUE docs `HalfKP` [official-stockfish.github.io](https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html)
- Bullet `jw1912/bullet` for fast-trainer ideas
- `cozy-chess` `rayon` `axum` `mimalloc`

---

**RustClaw by Vaibhav — `git clone https://github.com/thisisforlearn/rustclaw && cargo run --release --bin rustclaw -- --nnue rustclaw.nnue web --port 3000`**

