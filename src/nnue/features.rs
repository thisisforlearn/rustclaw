//! HalfKP feature set - pure Rust from scratch
//! No Stockfish weights are ever loaded. Architecture is compatible but all weights are random init.
//!
//! HalfKP: (our_king_square, piece_square, piece_type, piece_color)
//! Total features = 64 * 64 * 5 * 2 = 40960
//! We use 41024 with padding (64*10*64) for alignment.
//!
//! Reference: https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html

use cozy_chess::{Board, Color, Piece, Square};

pub const FEATURE_SIZE: usize = 41024; // 64*10*64 padded, actual 40960 used
pub const INPUT_SIZE: usize = 40960;
pub const KING_BUCKETS: usize = 64;
pub const PIECE_TYPES: usize = 5; // pawn, knight, bishop, rook, queen (king excluded)
pub const COLORS: usize = 2;

/// HalfKP feature index
/// p_idx = piece_type * 2 + piece_color
/// halfkp_idx = piece_square + (p_idx + king_square * 10) * 64
#[inline(always)]
pub fn halfkp_index(king_sq: Square, piece_sq: Square, piece: Piece, color: Color) -> usize {
    let p_type = match piece {
        Piece::Pawn => 0,
        Piece::Knight => 1,
        Piece::Bishop => 2,
        Piece::Rook => 3,
        Piece::Queen => 4,
        Piece::King => unreachable!("king not encoded in HalfKP"),
    };
    let p_color = match color {
        Color::White => 0,
        Color::Black => 1,
    };
    let p_idx = p_type * 2 + p_color;
    let ks = king_sq as usize;
    let ps = piece_sq as usize;
    ps + (p_idx + ks * 10) * 64
}

#[inline(always)]
fn orient_square(sq: Square, perspective: Color) -> Square {
    match perspective {
        Color::White => sq,
        Color::Black => sq.flip_rank(),
    }
}

/// Extract active feature indices for a given perspective
pub fn active_features(board: &Board, perspective: Color) -> Vec<usize> {
    let mut features = Vec::with_capacity(32);
    let king_sq_raw = board.king(perspective);
    let king_sq = orient_square(king_sq_raw, perspective);

    for color in [Color::White, Color::Black] {
        for piece in [Piece::Pawn, Piece::Knight, Piece::Bishop, Piece::Rook, Piece::Queen] {
            let mut bb = board.colored_pieces(color, piece);
            while !bb.is_empty() {
                let sq_raw = bb.next_square().unwrap();
                bb ^= sq_raw.bitboard();
                let sq = orient_square(sq_raw, perspective);
                // For black perspective, flip colors
                let feat_color = match perspective {
                    Color::White => color,
                    Color::Black => if color == Color::White { Color::Black } else { Color::White },
                };
                let idx = halfkp_index(king_sq, sq, piece, feat_color);
                features.push(idx);
            }
        }
    }
    features
}

/// For incremental update: compute added/removed features between two positions
pub fn diff_features(old_board: &Board, new_board: &Board, perspective: Color) -> (Vec<usize>, Vec<usize>) {
    let old = active_features(old_board, perspective);
    let new = active_features(new_board, perspective);
    // naive diff - in real engine this is incrementally tracked per move
    let mut added = Vec::new();
    let mut removed = Vec::new();
    for &f in &new { if !old.contains(&f) { added.push(f); } }
    for &f in &old { if !new.contains(&f) { removed.push(f); } }
    (added, removed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cozy_chess::Board;

    #[test]
    fn test_feature_count() {
        let board: Board = Board::default();
        let feats_w = active_features(&board, Color::White);
        let feats_b = active_features(&board, Color::Black);
        assert_eq!(feats_w.len(), 30); // 16 white pieces - king =15, 16 black =15? Actually pawns etc.
        assert!(feats_w.iter().all(|&x| x < INPUT_SIZE));
        assert!(feats_b.iter().all(|&x| x < INPUT_SIZE));
    }
}
