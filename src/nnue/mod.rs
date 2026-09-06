//! CrabChess NNUE - Pure Rust from scratch, maximally optimized
//! Architecture: HalfKP 40960 -> 256x2 -> 32 -> 32 -> 1
//! Quantized inference, AVX2 SIMD, efficiently updatable accumulators
//! No Stockfish weights - all weights are randomly initialized and trained from zero.

pub mod features;
pub mod accumulator;
pub mod layers;

use accumulator::{Accumulator, FeatureTransformer, L1_SIZE, QA, QB, SCALE};
use layers::{Layer2, Layer3, OutputLayer, L2_SIZE, L3_SIZE};
use cozy_chess::{Board, Color};
use std::fs::File;
use std::io::{Read, Write};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

pub const NNUE_MAGIC: u32 = 0x434E4E55; // "CNNU" Crab NNUE
pub const NNUE_VERSION: u32 = 1;

#[derive(Clone)]
pub struct Network {
    pub ft: FeatureTransformer,
    pub l2: Layer2,
    pub l3: Layer3,
    pub output: OutputLayer,
    // Accumulators for search (stack)
    pub acc_white: Accumulator,
    pub acc_black: Accumulator,
}

impl Network {
    pub fn new_random() -> Self {
        let ft = FeatureTransformer::new_random();
        let acc_white = Accumulator::new(&ft.biases);
        let acc_black = Accumulator::new(&ft.biases);
        Self {
            ft: ft.clone(),
            l2: Layer2::new_random(),
            l3: Layer3::new_random(),
            output: OutputLayer::new_random(),
            acc_white,
            acc_black,
        }
    }

    /// Refresh both accumulators from board position
    pub fn refresh(&mut self, board: &Board) {
        self.acc_white.refresh(board, Color::White, &self.ft);
        self.acc_black.refresh(board, Color::Black, &self.ft);
    }

    /// Evaluate position - returns centipawns from STM perspective
    #[inline(always)]
    pub fn evaluate(&self, board: &Board) -> i32 {
        // For lazy eval we use current accumulators; caller must ensure they are up to date
        // For simple eval we refresh
        let mut aw = self.acc_white.clone();
        let mut ab = self.acc_black.clone();
        aw.refresh(board, Color::White, &self.ft);
        ab.refresh(board, Color::Black, &self.ft);
        layers::forward_network(&aw.values, &ab.values, &self.l2, &self.l3, &self.output, board.side_to_move())
    }

    /// Fast evaluate with precomputed accumulators
    #[inline(always)]
    pub fn evaluate_with_acc(&self, acc_w: &[i16; L1_SIZE], acc_b: &[i16; L1_SIZE], stm: Color) -> i32 {
        layers::forward_network(acc_w, acc_b, &self.l2, &self.l3, &self.output, stm)
    }

    /// Save quantized network to file
    pub fn save(&self, path: &str) -> anyhow::Result<()> {
        let mut f = File::create(path)?;
        f.write_u32::<LittleEndian>(NNUE_MAGIC)?;
        f.write_u32::<LittleEndian>(NNUE_VERSION)?;
        // FT weights
        for &w in &self.ft.weights { f.write_i16::<LittleEndian>(w)?; }
        for &b in &self.ft.biases { f.write_i16::<LittleEndian>(b)?; }
        for &w in &self.l2.weights { f.write_i8(w)?; }
        for &b in &self.l2.biases { f.write_i32::<LittleEndian>(b)?; }
        for &w in &self.l3.weights { f.write_i8(w)?; }
        for &b in &self.l3.biases { f.write_i32::<LittleEndian>(b)?; }
        for &w in &self.output.weights { f.write_i8(w)?; }
        f.write_i32::<LittleEndian>(self.output.bias)?;
        Ok(())
    }

    pub fn load(path: &str) -> anyhow::Result<Self> {
        let mut f = File::open(path)?;
        let magic = f.read_u32::<LittleEndian>()?;
        assert_eq!(magic, NNUE_MAGIC, "bad NNUE magic");
        let _ver = f.read_u32::<LittleEndian>()?;
        let mut ft_weights = vec![0i16; features::INPUT_SIZE * L1_SIZE];
        for w in &mut ft_weights { *w = f.read_i16::<LittleEndian>()?; }
        let mut ft_biases = vec![0i16; L1_SIZE];
        for b in &mut ft_biases { *b = f.read_i16::<LittleEndian>()?; }
        let mut l2_weights = vec![0i8; layers::L1_OUT * layers::L2_SIZE];
        for w in &mut l2_weights { *w = f.read_i8()? as i8; }
        let mut l2_biases = vec![0i32; layers::L2_SIZE];
        for b in &mut l2_biases { *b = f.read_i32::<LittleEndian>()?; }
        let mut l3_weights = vec![0i8; layers::L2_SIZE * layers::L3_SIZE];
        for w in &mut l3_weights { *w = f.read_i8()? as i8; }
        let mut l3_biases = vec![0i32; layers::L3_SIZE];
        for b in &mut l3_biases { *b = f.read_i32::<LittleEndian>()?; }
        let mut out_weights = vec![0i8; layers::L3_SIZE];
        for w in &mut out_weights { *w = f.read_i8()? as i8; }
        let out_bias = f.read_i32::<LittleEndian>()?;
        let ft = FeatureTransformer { weights: ft_weights, biases: ft_biases };
        let acc_white = Accumulator::new(&ft.biases);
        let acc_black = Accumulator::new(&ft.biases);
        Ok(Self {
            ft,
            l2: Layer2 { weights: l2_weights, biases: l2_biases },
            l3: Layer3 { weights: l3_weights, biases: l3_biases },
            output: OutputLayer { weights: out_weights, bias: out_bias },
            acc_white,
            acc_black,
        })
    }

    pub fn param_count(&self) -> usize {
        self.ft.weights.len() + self.ft.biases.len()
            + self.l2.weights.len() + self.l2.biases.len()
            + self.l3.weights.len() + self.l3.biases.len()
            + self.output.weights.len() + 1
    }

    pub fn size_mb(&self) -> f64 {
        // quantized size
        let bytes = self.ft.weights.len()*2 + self.ft.biases.len()*2
            + self.l2.weights.len() + self.l2.biases.len()*4
            + self.l3.weights.len() + self.l3.biases.len()*4
            + self.output.weights.len() + 4;
        bytes as f64 / 1_048_576.0
    }
}

impl Default for Network {
    fn default() -> Self { Self::new_random() }
}
