#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use clap::Parser;
use chess_engine_rust::data::{TrainingPosition, dataset_info};
use cozy_chess::Board;
use std::str::FromStr;
use std::fs::File;
use std::io::{BufWriter, Write};

#[derive(Parser, Debug)]
#[command(name="datagen", about="Generate training data for CrabChess NNUE")]
struct Cli {
    #[arg(long, default_value="50000")]
    count: usize,
    #[arg(long, default_value="data/selfplay.jsonl")]
    output: String,
    /// Use random playouts with material eval (fast, no search)
    #[arg(long)]
    fast: bool,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    println!("{}", dataset_info());
    println!("Generating {} positions -> {} (fast={})", cli.count, cli.output, cli.fast);

    std::fs::create_dir_all(std::path::Path::new(&cli.output).parent().unwrap())?;
    let file = File::create(&cli.output)?;
    let mut w = BufWriter::new(file);

    let start_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    let mut board = Board::from_str(start_fen).map_err(|e| anyhow::anyhow!("{}", e))?;
    let mut generated = 0;
    let mut rng: u64 = 0x12345678;

    while generated < cli.count {
        let mut moves = Vec::new();
        board.generate_moves(|pm| { for mv in pm { moves.push(mv); } false });
        if moves.is_empty() {
            board = Board::from_str(start_fen).map_err(|e| anyhow::anyhow!("{}", e))?;
            continue;
        }
        // xorshift
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        let mv = moves[(rng as usize) % moves.len()];
        board.play_unchecked(mv);

        // Material eval as synthetic target
        let mut score = 0;
        for color in [cozy_chess::Color::White, cozy_chess::Color::Black] {
            let sign = if color == board.side_to_move() { 1 } else { -1 };
            for (piece, val) in [(cozy_chess::Piece::Pawn,100),(cozy_chess::Piece::Knight,320),(cozy_chess::Piece::Bishop,330),(cozy_chess::Piece::Rook,500),(cozy_chess::Piece::Queen,900)] {
                score += sign * board.colored_pieces(color, piece).len() as i32 * val;
            }
        }
        score = score.clamp(-1500, 1500);
        let result = if score > 100 { 1.0 } else if score < -100 { 0.0 } else { 0.5 };

        let pos = TrainingPosition {
            fen: format!("{}", board),
            eval_cp: score as i16,
            result,
            best_move: None,
        };
        serde_json::to_writer(&mut w, &pos)?;
        writeln!(&mut w)?;
        generated += 1;
        if generated % 10000 == 0 { println!("  {}/{} positions", generated, cli.count); w.flush()?; }

        // Random restart to diversify
        if generated % 40 == 0 {
            rng ^= rng << 13;
            if rng % 7 == 0 {
                board = Board::from_str(start_fen).map_err(|e| anyhow::anyhow!("{}", e))?;
            }
        }
    }
    w.flush()?;
    println!("✅ Generated {} positions -> {}", generated, cli.output);
    println!("   Train with: cargo run --release --bin trainer -- --data {} --epochs 10 --batch-size 1024", cli.output);
    Ok(())
}
