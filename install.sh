#!/usr/bin/env bash
set -e
# RustClaw — one-command installer for Linux / macOS / Windows (WSL/Git Bash) / Android Termux
# by Vaibhav — GPL-2.0 + commercial
echo "🦀 RustClaw installer — by Vaibhav"
echo "   GPL-2.0 open source • Commercial license: vaibhav@rustclaw.dev"
echo ""

if ! command -v cargo >/dev/null 2>&1; then
  echo "Installing rustup (Rust toolchain)..."
  if command -v curl >/dev/null 2>&1; then curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
  elif command -v wget >/dev/null 2>&1; then wget -qO- https://sh.rustup.rs | sh -s -- -y
  else echo "Please install Rust from https://rustup.rs manually"; exit 1
  fi
  source "$HOME/.cargo/env"
fi

echo "✓ cargo $(cargo --version)"
echo "Building RustClaw (release, LTO, AVX2, mimalloc)..."
cargo build --release --bin rustclaw --bin trainer --bin datagen

if [ ! -f rustclaw.nnue ]; then
  if [ -f crabchess.nnue ]; then cp crabchess.nnue rustclaw.nnue; echo "✓ using crabchess.nnue as rustclaw.nnue"; 
  else echo "No rustclaw.nnue found — generating smoke (10s) then ultra..."; cargo run --release --bin trainer -- --smoke-test; fi
fi

echo ""
echo "✅ Build complete!"
echo "  Binary: ./target/release/rustclaw"
echo "  NNUE:   rustclaw.nnue (20MB, 10.5M params)"
echo ""
echo "Try CLI (all OS same):"
echo "  ./target/release/rustclaw --help"
echo "  ./target/release/rustclaw --nnue rustclaw.nnue bench"
echo "  ./target/release/rustclaw --nnue rustclaw.nnue eval \"rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1\""
echo "  ./target/release/rustclaw --nnue rustclaw.nnue uci"
echo ""
echo "Try Web GUI like lichess (any OS, just open browser):"
echo "  ./target/release/rustclaw --nnue rustclaw.nnue web --port 3000"
echo "  → http://localhost:3000"
echo ""
echo "Train 3100 ultra (10h, <10GB, 12T):"
echo "  ./scripts/train_3100_ultra.sh"
echo ""
echo "Installed — enjoy RustClaw by Vaibhav!"
