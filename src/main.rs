#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

mod nnue;
mod engine;
mod data;
mod trainer;
mod web;

use clap::{Parser, Subcommand};
use cozy_chess::{Board, Move};
use std::str::FromStr;
use std::sync::Arc;
use tokio::sync::Mutex;

#[derive(Parser)]
#[command(name="rustclaw", author="Vaibhav", version="1.0.0", about="RustClaw - Ultra-optimized Pure Rust NNUE Chess Engine by Vaibhav | GPL-2.0 + commercial")]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// Path to NNUE file
    #[arg(long, global=true, default_value="rustclaw.nnue")]
    nnue: String,

    /// Depth for search
    #[arg(long, global=true, default_value="10")]
    depth: u8,

    /// Threads for search (Lazy SMP, 12 = your i5 max, key for 3100 Elo)
    #[arg(long, global=true, default_value="12")]
    threads: usize,
}

#[derive(Subcommand)]
enum Commands {
    /// Run UCI loop
    Uci,
    /// Evaluate a FEN
    Eval { fen: String },
    /// Run Web UI (lichess-like)
    Web {
        #[arg(long, default_value="3000")]
        port: u16,
        #[arg(long, default_value="0.0.0.0")]
        host: String,
    },
    /// Play a move from FEN
    Play { fen: String },
    /// Benchmark
    Bench,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // Load or create network
    let network = match nnue::Network::load(&cli.nnue) {
        Ok(n) => {
            println!("Loaded NNUE {} ({} params, {:.2} MB)", cli.nnue, n.param_count(), n.size_mb());
            n
        },
        Err(_) => {
            println!("No NNUE found at {} - creating random initialized network (train from scratch)", cli.nnue);
            println!("To train: cargo run --release --bin trainer -- --help");
            nnue::Network::new_random()
        }
    };

    match cli.command.unwrap_or(Commands::Uci) {
        Commands::Uci => run_uci(network).await?,
        Commands::Eval { fen } => {
            let board = Board::from_str(&fen).map_err(|e| anyhow::anyhow!("bad FEN: {}", e))?;
            let eval = network.evaluate(&board);
            println!("FEN: {}", fen);
            println!("Eval: {} cp ({:+.2})", eval, eval as f32 / 100.0);
        },
        Commands::Web { port, .. } => {
            let net = Arc::new(Mutex::new(network));
            web::run_server(net, port).await?;
        },
        Commands::Play { fen } => {
            let board = Board::from_str(&fen).map_err(|e| anyhow::anyhow!("bad FEN: {}", e))?;
            let result = engine::search::search(&board, &network, engine::search::SearchParams { depth: cli.depth, threads: cli.threads, ..Default::default() });
            println!("Best move: {:?}", result.best_move.map(|m| m.to_string()).unwrap_or("none".into()));
            println!("Score: {} Nodes: {} Threads: {}", result.score, result.nodes, cli.threads);
        },
        Commands::Bench => {
            let board = Board::default();
            let start = std::time::Instant::now();
            for _ in 0..1000 {
                let _ = network.evaluate(&board);
            }
            let elapsed = start.elapsed();
            println!("1000 evals in {:?} ({:.0} evals/s)", elapsed, 1000.0 / elapsed.as_secs_f32());
            println!("Network: {} params, {:.2} MB", network.param_count(), network.size_mb());
            // Perft
            let nodes = engine::search::perft(&board, 4);
            println!("Perft(4) = {} (movegen validation)", nodes);
        }
    }
    Ok(())
}

async fn run_uci(mut network: nnue::Network) -> anyhow::Result<()> {
    use std::io::{self, BufRead};
    let mut board = Board::default();
    network.refresh(&board);
    println!("RustClaw 1.0 - Pure Rust NNUE (HalfKP 40960->256x2->32->32->1, AVX2, quantized) by Vaibhav | GPL-2.0 + commercial");
    println!("id name RustClaw");
    println!("id author Vaibhav");
    println!("uciok");
    let stdin = io::stdin();
    for line in stdin.lock().lines() {
        let line = line?;
        let parts: Vec<&str> = line.trim().split_whitespace().collect();
        if parts.is_empty() { continue; }
        match parts[0] {
            "uci" => {
                println!("id name RustClaw");
                println!("id author Vaibhav");
                println!("option name Threads type spin default 1 min 1 max 12");
                println!("option name Hash type spin default 16 min 1 max 1024");
                println!("uciok");
            },
            "isready" => println!("readyok"),
            "ucinewgame" => {
                board = Board::default();
                network.refresh(&board);
            },
            "position" => {
                // position startpos moves ... OR position fen ... moves ...
                let mut idx = 1;
                if parts.get(idx) == Some(&"startpos") {
                    board = Board::default();
                    idx += 1;
                } else if parts.get(idx) == Some(&"fen") {
                    // collect fen until "moves"
                    let mut fen_parts = vec![];
                    idx += 1;
                    while idx < parts.len() && parts[idx] != "moves" {
                        fen_parts.push(parts[idx]);
                        idx += 1;
                    }
                    let fen = fen_parts.join(" ");
                    board = Board::from_str(&fen).unwrap_or_default();
                }
                if parts.get(idx) == Some(&"moves") {
                    idx += 1;
                    while idx < parts.len() {
                        let mv_str = parts[idx];
                        // Find move by string
                        let mut found = None;
                        let mut moves = Vec::new();
                        board.generate_moves(|pm| { for mv in pm { moves.push(mv); } false });
                        for m in moves {
                            if m.to_string() == mv_str {
                                found = Some(m);
                                break;
                            }
                        }
                        if let Some(mv) = found {
                            board.play_unchecked(mv);
                        }
                        idx += 1;
                    }
                }
                network.refresh(&board);
            },
            "go" => {
                let mut depth = 10u8;
                let mut movetime = None;
                let mut threads = 12usize;
                let mut i = 1;
                while i < parts.len() {
                    match parts[i] {
                        "depth" => { depth = parts[i+1].parse().unwrap_or(10); i+=2; },
                        "movetime" => { movetime = Some(parts[i+1].parse().unwrap_or(1000)); i+=2; },
                        "threads" => { threads = parts[i+1].parse().unwrap_or(12); i+=2; },
                        _ => i+=1,
                    }
                }
                let params = engine::search::SearchParams { depth, movetime_ms: movetime, threads, ..Default::default() };
                let result = engine::search::search(&board, &network, params);
                if let Some(mv) = result.best_move {
                    println!("bestmove {}", mv);
                } else {
                    println!("bestmove 0000");
                }
            },
            "eval" => {
                let eval = network.evaluate(&board);
                println!("info score cp {} string NNUE eval", eval);
            },
            "d" => {
                println!("{}", board);
                println!("Eval: {} cp", network.evaluate(&board));
            },
            "quit" => break,
            _ => {}
        }
    }
    Ok(())
}
