//! Efficiently Updatable Accumulator - the core of NNUE speed
//! Two accumulators (white/black perspective), each 256 dims
//! Quantized to i16 with QA=255, QB=64 scaling
//!
//! Update is O(1) per move: only changed features are added/subtracted
//! Full refresh only on king moves.

use super::features::{active_features, INPUT_SIZE};
use cozy_chess::{Board, Color};

pub const L1_SIZE: usize = 256;
pub const QA: i32 = 255;
pub const QB: i32 = 64;
pub const QAB: i32 = QA * QB; // 16320
pub const SCALE: i32 = 400; // for cp conversion

/// Quantized weights: feature -> L1
#[derive(Clone)]
pub struct FeatureTransformer {
    pub weights: Vec<i16>, // [INPUT_SIZE * L1_SIZE]
    pub biases: Vec<i16>,  // [L1_SIZE]
}

impl FeatureTransformer {
    pub fn new_random() -> Self {
        use rand_distr::{Distribution, Normal};
        use rand::SeedableRng;
        use rand::rngs::StdRng;
        let mut rng = StdRng::seed_from_u64(0xC0FFEE);
        let normal = Normal::new(0.0, 0.02).unwrap();
        let mut weights = vec![0i16; INPUT_SIZE * L1_SIZE];
        // Xavier-like init scaled to QA
        for w in &mut weights {
            let v: f64 = normal.sample(&mut rng);
            *w = (v * QA as f64).clamp(-127.0, 127.0) as i16;
        }
        let biases = vec![0i16; L1_SIZE];
        Self { weights, biases }
    }

    #[inline(always)]
    pub fn weight(&self, feature: usize, neuron: usize) -> i16 {
        self.weights[feature * L1_SIZE + neuron]
    }
}

/// Accumulator for one perspective
#[derive(Clone, Debug)]
pub struct Accumulator {
    pub values: [i16; L1_SIZE],
    pub computed: bool,
}

impl Accumulator {
    pub fn new(biases: &[i16]) -> Self {
        let mut values = [0i16; L1_SIZE];
        values.copy_from_slice(&biases[..L1_SIZE]);
        Self { values, computed: true }
    }

    #[inline(always)]
    pub fn refresh(&mut self, board: &Board, perspective: Color, ft: &FeatureTransformer) {
        self.values.copy_from_slice(&ft.biases);
        for feat in active_features(board, perspective) {
            let base = feat * L1_SIZE;
            for i in 0..L1_SIZE {
                self.values[i] = self.values[i].wrapping_add(ft.weights[base + i]);
            }
        }
        self.computed = true;
    }

    /// Incremental update: add + remove features
    #[inline(always)]
    pub fn update(&mut self, added: &[usize], removed: &[usize], ft: &FeatureTransformer) {
        for &feat in added {
            let base = feat * L1_SIZE;
            for i in 0..L1_SIZE {
                self.values[i] = self.values[i].wrapping_add(ft.weights[base + i]);
            }
        }
        for &feat in removed {
            let base = feat * L1_SIZE;
            for i in 0..L1_SIZE {
                self.values[i] = self.values[i].wrapping_sub(ft.weights[base + i]);
            }
        }
    }

    /// SIMD optimized update using AVX2 where available
    #[inline(always)]
    pub fn update_simd(&mut self, added: &[usize], removed: &[usize], ft: &FeatureTransformer) {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe { self.update_avx2(added, removed, ft); return; }
            }
        }
        self.update(added, removed, ft);
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn update_avx2(&mut self, added: &[usize], removed: &[usize], ft: &FeatureTransformer) {
        use std::arch::x86_64::*;
        // Process 16 i16 at a time = 256 bits
        for &feat in added {
            let base = feat * L1_SIZE;
            let w_ptr = ft.weights.as_ptr().add(base) as *const __m256i;
            let a_ptr = self.values.as_mut_ptr() as *mut __m256i;
            for i in (0..L1_SIZE).step_by(16) {
                let w = _mm256_loadu_si256(w_ptr.add(i/16));
                let a = _mm256_loadu_si256(a_ptr.add(i/16));
                let res = _mm256_adds_epi16(a, w); // saturating
                _mm256_storeu_si256(a_ptr.add(i/16), res);
            }
        }
        for &feat in removed {
            let base = feat * L1_SIZE;
            let w_ptr = ft.weights.as_ptr().add(base) as *const __m256i;
            let a_ptr = self.values.as_mut_ptr() as *mut __m256i;
            for i in (0..L1_SIZE).step_by(16) {
                let w = _mm256_loadu_si256(w_ptr.add(i/16));
                let a = _mm256_loadu_si256(a_ptr.add(i/16));
                let res = _mm256_subs_epi16(a, w);
                _mm256_storeu_si256(a_ptr.add(i/16), res);
            }
        }
    }
}

/// Dual accumulator stack for incremental search
#[derive(Clone)]
pub struct AccumulatorStack {
    pub white: Vec<Accumulator>,
    pub black: Vec<Accumulator>,
    pub ft: FeatureTransformer,
}

impl AccumulatorStack {
    pub fn new(ft: FeatureTransformer) -> Self {
        let acc_w = Accumulator::new(&ft.biases);
        let acc_b = Accumulator::new(&ft.biases);
        Self { white: vec![acc_w], black: vec![acc_b], ft }
    }

    pub fn refresh(&mut self, board: &Board) {
        self.white.last_mut().unwrap().refresh(board, Color::White, &self.ft);
        self.black.last_mut().unwrap().refresh(board, Color::Black, &self.ft);
    }
}
