#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use rustclaw::nnue::Network;
use rustclaw::trainer::{TrainConfig, train};
use clap::Parser;

#[derive(Parser, Debug)]
#[command(name="rustclaw-trainer", about="Train RustClaw NNUE from scratch - pure Rust by Vaibhav | GPL-3.0 + commercial")]
struct Cli {
    /// Path to training data JSONL (processed Lichess evals). If not provided, uses synthetic.
    #[arg(long)]
    data: Option<String>,

    #[arg(long, default_value="10")]
    epochs: usize,

    #[arg(long, default_value="1024")]
    batch_size: usize,

    #[arg(long, default_value="0.001")]
    lr: f32,

    #[arg(long, default_value="0.0001")]
    weight_decay: f32,

    #[arg(long, default_value="rustclaw.nnue")]
    output: String,

    #[arg(long, default_value="200000")]
    synthetic_size: usize,

    #[arg(long, default_value="1")]
    save_every: usize,

    /// Quick test mode - tiny dataset, verify pipeline
    #[arg(long)]
    smoke_test: bool,

    /// Ultra mode — 3100 Elo path: rayon 12 threads, huge batches, 10h optimized
    #[arg(long)]
    ultra: bool,

    #[arg(long, default_value="12")]
    threads: usize,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    let mut config = TrainConfig {
        epochs: cli.epochs,
        batch_size: cli.batch_size,
        lr: cli.lr,
        weight_decay: cli.weight_decay,
        output: cli.output.clone(),
        data_path: cli.data.clone(),
        synthetic_size: cli.synthetic_size,
        save_every: cli.save_every,
        ultra: cli.ultra,
        threads: cli.threads,
        ..Default::default()
    };

    if cli.smoke_test {
        config.epochs = 2;
        config.batch_size = 256;
        config.synthetic_size = 5000;
        config.save_every = 1;
        println!("🧪 Smoke test mode: 2 epochs, 5k positions");
    }
    if cli.ultra {
        println!("🔥 ULTRA 3100 MODE: rayon {} threads, mimalloc, parallel grads", cli.threads);
        // Tune for 10h on i5-1335U: huge batch = fewer steps, more throughput via rayon
        if cli.batch_size == 1024 { config.batch_size = 16384; }
        if cli.epochs == 10 { config.epochs = 60; } // 60*5M/16k=18750 steps → ~9h with rayon ~3x speedup
        rayon::ThreadPoolBuilder::new().num_threads(cli.threads).build_global().ok();
    }

    let network = Network::new_random();
    println!("Initialized fresh NNUE from scratch (no Stockfish weights)");
    println!("  Params: {}  Size: {:.2} MB", network.param_count(), network.size_mb());

    let trained = train(network, config)?;

    // Final verify
    let board = cozy_chess::Board::default();
    let eval = trained.evaluate(&board);
    println!("\n✅ Smoke eval on startpos: {} cp", eval);
    println!("   Model saved. Test with:");
    println!("   cargo run --release -- --nnue {} -- eval \"rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1\"", cli.output);
    println!("   cargo run --release -- --nnue {} web --port 3000", cli.output);

    Ok(())
}
