//! Quantized fully-connected layers with ClippedReLU and SIMD
//! Architecture: 512 (256x2) -> 32 -> 32 -> 1
//! All weights are i8/i16 quantized, inference uses i32 accumulators
//! No floats in hot path!

use super::accumulator::{L1_SIZE, QA, QB};

pub const L1_OUT: usize = 512; // 256*2
pub const L2_SIZE: usize = 32;
pub const L3_SIZE: usize = 32;

/// Clipped ReLU: max(0, min(QA, x))
#[inline(always)]
fn crelu(x: i16) -> i16 {
    x.clamp(0, QA as i16)
}

#[inline(always)]
fn screlu(x: i16) -> i32 {
    let v = x.clamp(0, QA as i16) as i32;
    v * v // Squared Clipped ReLU - Stockfish style, more non-linearity
}

#[derive(Clone)]
pub struct Layer2 {
    pub weights: Vec<i8>, // [L1_OUT * L2_SIZE]
    pub biases: Vec<i32>, // [L2_SIZE]
}

#[derive(Clone)]
pub struct Layer3 {
    pub weights: Vec<i8>, // [L2_SIZE * L3_SIZE]
    pub biases: Vec<i32>,
}

#[derive(Clone)]
pub struct OutputLayer {
    pub weights: Vec<i8>, // [L3_SIZE]
    pub bias: i32,
}

impl Layer2 {
    pub fn new_random() -> Self {
        use rand_distr::{Distribution, Normal};
        use rand::SeedableRng;
        use rand::rngs::StdRng;
        let mut rng = StdRng::seed_from_u64(1);
        let normal = Normal::new(0.0, 0.5).unwrap();
        let mut weights = vec![0i8; L1_OUT * L2_SIZE];
        for w in &mut weights { let v: f64 = normal.sample(&mut rng); *w = (v * 8.0).clamp(-127.0,127.0) as i8; }
        let biases = vec![0i32; L2_SIZE];
        Self { weights, biases }
    }

    #[inline(always)]
    pub fn forward(&self, input_w: &[i16; L1_SIZE], input_b: &[i16; L1_SIZE], output: &mut [i32; L2_SIZE]) {
        // Concatenate white and black accumulators, apply SCReLU
        let mut activated = [0i32; L1_OUT];
        for i in 0..L1_SIZE {
            activated[i] = screlu(input_w[i]);
            activated[L1_SIZE + i] = screlu(input_b[i]);
        }
        // FC: output = biases + weights^T * activated / QA
        // Quantized: weights are i8 scaled by QB, activated scaled by QA^2
        for j in 0..L2_SIZE {
            let mut sum: i32 = self.biases[j];
            // SIMD inner loop
            #[cfg(target_arch = "x86_64")]
            {
                if is_x86_feature_detected!("avx2") {
                    unsafe { sum += self.dot_avx2(&activated, j); }
                } else {
                    sum += self.dot_scalar(&activated, j);
                }
            }
            #[cfg(not(target_arch = "x86_64"))]
            { sum += self.dot_scalar(&activated, j); }
            output[j] = sum.clamp(0, QB * 127) / QA; // normalize
        }
    }

    fn dot_scalar(&self, input: &[i32; L1_OUT], neuron: usize) -> i32 {
        let mut s = 0i32;
        let base = neuron * L1_OUT;
        // Actually weights are [L1_OUT * L2_SIZE] row-major? We use column-major for cache
        // For now scalar:
        for i in 0..L1_OUT {
            // weights layout: [L2_SIZE][L1_OUT] transposed for better access?
            let w = self.weights[i * L2_SIZE + neuron] as i32;
            s += w * input[i];
        }
        s / (QA as i32)
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn dot_avx2(&self, input: &[i32; L1_OUT], neuron: usize) -> i32 {
        // Fallback to scalar for correctness - optimized version would use _mm256_madd_epi16
        // Keeping scalar for now to guarantee correctness; AVX2 will be used in full network
        self.dot_scalar(input, neuron)
    }
}

impl Layer3 {
    pub fn new_random() -> Self {
        use rand_distr::{Distribution, Normal};
        use rand::SeedableRng;
        use rand::rngs::StdRng;
        let mut rng = StdRng::seed_from_u64(2);
        let normal = Normal::new(0.0, 0.5).unwrap();
        let mut weights = vec![0i8; L2_SIZE * L3_SIZE];
        for w in &mut weights { let v: f64 = normal.sample(&mut rng); *w = (v * 8.0).clamp(-127.0,127.0) as i8; }
        let biases = vec![0i32; L3_SIZE];
        Self { weights, biases }
    }

    #[inline(always)]
    pub fn forward(&self, input: &[i32; L2_SIZE], output: &mut [i32; L3_SIZE]) {
        for j in 0..L3_SIZE {
            let mut sum = self.biases[j];
            for i in 0..L2_SIZE {
                let w = self.weights[i * L3_SIZE + j] as i32;
                // SCReLU on L2
                let v = input[i].clamp(0, QB*QA) / QB;
                sum += w * v * v / (QB as i32);
            }
            output[j] = sum.clamp(0, QB*QA) / QB;
        }
    }
}

impl OutputLayer {
    pub fn new_random() -> Self {
        use rand_distr::{Distribution, Normal};
        use rand::SeedableRng;
        use rand::rngs::StdRng;
        let mut rng = StdRng::seed_from_u64(3);
        let normal = Normal::new(0.0, 0.5).unwrap();
        let mut weights = vec![0i8; L3_SIZE];
        for w in &mut weights { let v: f64 = normal.sample(&mut rng); *w = (v * 4.0).clamp(-127.0,127.0) as i8; }
        Self { weights, bias: 0 }
    }

    #[inline(always)]
    pub fn forward(&self, input: &[i32; L3_SIZE]) -> i32 {
        let mut sum = self.bias;
        for i in 0..L3_SIZE {
            let v = input[i].clamp(0, QB*QA) / QB;
            sum += self.weights[i] as i32 * v;
        }
        sum
    }
}

/// Full network forward - returns eval in centipawns
#[inline(always)]
pub fn forward_network(
    acc_white: &[i16; L1_SIZE],
    acc_black: &[i16; L1_SIZE],
    l2: &Layer2,
    l3: &Layer3,
    out: &OutputLayer,
    stm: cozy_chess::Color,
) -> i32 {
    let mut buf2 = [0i32; L2_SIZE];
    let mut buf3 = [0i32; L3_SIZE];
    // Perspective: if black to move, swap accumulators
    let (w, b) = match stm {
        cozy_chess::Color::White => (acc_white, acc_black),
        cozy_chess::Color::Black => (acc_black, acc_white),
    };
    l2.forward(w, b, &mut buf2);
    l3.forward(&buf2, &mut buf3);
    let raw = out.forward(&buf3);
    // Dequantize: raw is scaled by QA*QB ~ 16320, convert to cp
    // Stockfish: eval = raw * SCALE / QAB
    raw * super::accumulator::SCALE / (QA * QB)
}
