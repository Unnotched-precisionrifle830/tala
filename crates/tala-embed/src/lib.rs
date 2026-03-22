//! TALA Embedding Engine — SIMD-accelerated vector ops, quantization, and HNSW index.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

// ===========================================================================
// Aligned allocation
// ===========================================================================

/// 64-byte aligned f32 vector for SIMD operations.
pub struct AlignedVec {
    ptr: *mut f32,
    len: usize,
    cap: usize,
}

unsafe impl Send for AlignedVec {}
unsafe impl Sync for AlignedVec {}

impl AlignedVec {
    pub fn new(len: usize) -> Self {
        if len == 0 {
            return Self {
                ptr: std::ptr::NonNull::dangling().as_ptr(),
                len: 0,
                cap: 0,
            };
        }
        let layout = std::alloc::Layout::from_size_align(len * 4, 64).unwrap();
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) as *mut f32 };
        if ptr.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        Self {
            ptr,
            len,
            cap: len,
        }
    }

    pub fn as_slice(&self) -> &[f32] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl Drop for AlignedVec {
    fn drop(&mut self) {
        if self.cap > 0 {
            let layout = std::alloc::Layout::from_size_align(self.cap * 4, 64).unwrap();
            unsafe { std::alloc::dealloc(self.ptr as *mut u8, layout) };
        }
    }
}

impl Clone for AlignedVec {
    fn clone(&self) -> Self {
        let mut new = Self::new(self.len);
        new.as_mut_slice().copy_from_slice(self.as_slice());
        new
    }
}

impl std::ops::Deref for AlignedVec {
    type Target = [f32];
    #[inline]
    fn deref(&self) -> &[f32] {
        self.as_slice()
    }
}

impl std::ops::DerefMut for AlignedVec {
    #[inline]
    fn deref_mut(&mut self) -> &mut [f32] {
        self.as_mut_slice()
    }
}

impl From<Vec<f32>> for AlignedVec {
    fn from(v: Vec<f32>) -> Self {
        let mut aligned = Self::new(v.len());
        aligned.as_mut_slice().copy_from_slice(&v);
        aligned
    }
}

impl From<&[f32]> for AlignedVec {
    fn from(s: &[f32]) -> Self {
        let mut aligned = Self::new(s.len());
        aligned.as_mut_slice().copy_from_slice(s);
        aligned
    }
}

// ===========================================================================
// Scalar operations (portable fallback)
// ===========================================================================

pub mod scalar {
    #[inline]
    pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    #[inline]
    pub fn norm_sq(a: &[f32]) -> f32 {
        a.iter().map(|x| x * x).sum()
    }

    #[inline]
    pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let dot = dot_product(a, b);
        let na = norm_sq(a).sqrt();
        let nb = norm_sq(b).sqrt();
        dot / (na * nb + f32::EPSILON)
    }

    #[inline]
    pub fn l2_distance_sq(a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y) * (x - y))
            .sum()
    }
}

// ===========================================================================
// AVX2 + FMA operations (x86_64 only)
// ===========================================================================

#[cfg(target_arch = "x86_64")]
pub mod avx2 {
    use std::arch::x86_64::*;

    #[target_feature(enable = "avx2")]
    unsafe fn hsum_256(v: __m256) -> f32 {
        let hi = _mm256_extractf128_ps(v, 1);
        let lo = _mm256_castps256_ps128(v);
        let sum = _mm_add_ps(lo, hi);
        let shuf = _mm_movehdup_ps(sum);
        let sums = _mm_add_ps(sum, shuf);
        let shuf2 = _mm_movehl_ps(sums, sums);
        let result = _mm_add_ss(sums, shuf2);
        _mm_cvtss_f32(result)
    }

    /// Dot product — 4-way unrolled to saturate FMA throughput.
    /// Single accumulator is latency-bound (4 cycles × N iters). Four independent
    /// chains reduce critical path 4×, making us throughput-bound instead.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot_product(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len();
        let pa = a.as_ptr();
        let pb = b.as_ptr();
        let unrolled = n / 32; // 4 × 8 floats per iteration

        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();
        let mut acc3 = _mm256_setzero_ps();

        for i in 0..unrolled {
            let base = i * 32;
            let va0 = _mm256_loadu_ps(pa.add(base));
            let vb0 = _mm256_loadu_ps(pb.add(base));
            let va1 = _mm256_loadu_ps(pa.add(base + 8));
            let vb1 = _mm256_loadu_ps(pb.add(base + 8));
            let va2 = _mm256_loadu_ps(pa.add(base + 16));
            let vb2 = _mm256_loadu_ps(pb.add(base + 16));
            let va3 = _mm256_loadu_ps(pa.add(base + 24));
            let vb3 = _mm256_loadu_ps(pb.add(base + 24));
            acc0 = _mm256_fmadd_ps(va0, vb0, acc0);
            acc1 = _mm256_fmadd_ps(va1, vb1, acc1);
            acc2 = _mm256_fmadd_ps(va2, vb2, acc2);
            acc3 = _mm256_fmadd_ps(va3, vb3, acc3);
        }

        acc0 = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
        let mut total = hsum_256(acc0);

        // Remainder: up to 31 elements
        for i in (unrolled * 32)..n {
            total += *a.get_unchecked(i) * *b.get_unchecked(i);
        }
        total
    }

    /// Cosine similarity — 2-way unrolled (6 accumulators + 4 temps = 10 regs, fits in YMM).
    /// 4-way would need 20 regs, spilling on AVX2's 16 YMM budget.
    /// For dim=384: 24 iters, 6 chains × 24 deep = 96 cycle latency at FMA=4.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len();
        let pa = a.as_ptr();
        let pb = b.as_ptr();
        let unrolled = n / 16; // 2 × 8 floats per iteration

        let mut dot0 = _mm256_setzero_ps();
        let mut dot1 = _mm256_setzero_ps();
        let mut na0 = _mm256_setzero_ps();
        let mut na1 = _mm256_setzero_ps();
        let mut nb0 = _mm256_setzero_ps();
        let mut nb1 = _mm256_setzero_ps();

        for i in 0..unrolled {
            let base = i * 16;
            let va0 = _mm256_loadu_ps(pa.add(base));
            let vb0 = _mm256_loadu_ps(pb.add(base));
            let va1 = _mm256_loadu_ps(pa.add(base + 8));
            let vb1 = _mm256_loadu_ps(pb.add(base + 8));

            dot0 = _mm256_fmadd_ps(va0, vb0, dot0);
            dot1 = _mm256_fmadd_ps(va1, vb1, dot1);

            na0 = _mm256_fmadd_ps(va0, va0, na0);
            na1 = _mm256_fmadd_ps(va1, va1, na1);

            nb0 = _mm256_fmadd_ps(vb0, vb0, nb0);
            nb1 = _mm256_fmadd_ps(vb1, vb1, nb1);
        }

        dot0 = _mm256_add_ps(dot0, dot1);
        na0 = _mm256_add_ps(na0, na1);
        nb0 = _mm256_add_ps(nb0, nb1);

        let mut dot = hsum_256(dot0);
        let mut na_sq = hsum_256(na0);
        let mut nb_sq = hsum_256(nb0);

        for i in (unrolled * 16)..n {
            let ai = *a.get_unchecked(i);
            let bi = *b.get_unchecked(i);
            dot += ai * bi;
            na_sq += ai * ai;
            nb_sq += bi * bi;
        }

        dot / (na_sq.sqrt() * nb_sq.sqrt() + f32::EPSILON)
    }

    /// L2 distance squared — 4-way unrolled.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn l2_distance_sq(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len();
        let pa = a.as_ptr();
        let pb = b.as_ptr();
        let unrolled = n / 32;

        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();
        let mut acc3 = _mm256_setzero_ps();

        for i in 0..unrolled {
            let base = i * 32;
            let d0 = _mm256_sub_ps(_mm256_loadu_ps(pa.add(base)), _mm256_loadu_ps(pb.add(base)));
            let d1 = _mm256_sub_ps(_mm256_loadu_ps(pa.add(base + 8)), _mm256_loadu_ps(pb.add(base + 8)));
            let d2 = _mm256_sub_ps(_mm256_loadu_ps(pa.add(base + 16)), _mm256_loadu_ps(pb.add(base + 16)));
            let d3 = _mm256_sub_ps(_mm256_loadu_ps(pa.add(base + 24)), _mm256_loadu_ps(pb.add(base + 24)));
            acc0 = _mm256_fmadd_ps(d0, d0, acc0);
            acc1 = _mm256_fmadd_ps(d1, d1, acc1);
            acc2 = _mm256_fmadd_ps(d2, d2, acc2);
            acc3 = _mm256_fmadd_ps(d3, d3, acc3);
        }

        acc0 = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
        let mut total = hsum_256(acc0);

        for i in (unrolled * 32)..n {
            let d = *a.get_unchecked(i) - *b.get_unchecked(i);
            total += d * d;
        }
        total
    }
}

// ===========================================================================
// Dispatch — runtime ISA selection
// ===========================================================================

/// Cosine similarity with automatic SIMD dispatch.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { avx2::cosine_similarity(a, b) };
        }
    }
    scalar::cosine_similarity(a, b)
}

/// Dot product with automatic SIMD dispatch.
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { avx2::dot_product(a, b) };
        }
    }
    scalar::dot_product(a, b)
}

/// L2 distance squared with automatic SIMD dispatch.
pub fn l2_distance_sq(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { avx2::l2_distance_sq(a, b) };
        }
    }
    scalar::l2_distance_sq(a, b)
}

// ===========================================================================
// Batch operations
// ===========================================================================

/// Compute cosine similarity of `query` against every vector in `corpus`.
/// `corpus` is a flat buffer: vector i = corpus[i*dim .. (i+1)*dim].
pub fn batch_cosine(query: &[f32], corpus: &[f32], dim: usize, results: &mut [f32]) {
    let n = results.len();
    for i in 0..n {
        results[i] = cosine_similarity(query, &corpus[i * dim..(i + 1) * dim]);
    }
}

/// Parallel batch cosine using rayon.
pub fn batch_cosine_parallel(query: &[f32], corpus: &[f32], dim: usize, results: &mut [f32]) {
    use rayon::prelude::*;
    let query = query.to_vec(); // clone for Send
    results
        .par_chunks_mut(256)
        .enumerate()
        .for_each(|(chunk_idx, chunk)| {
            let base = chunk_idx * 256;
            for (j, slot) in chunk.iter_mut().enumerate() {
                let i = base + j;
                *slot = cosine_similarity(&query, &corpus[i * dim..(i + 1) * dim]);
            }
        });
}

// ===========================================================================
// Quantization
// ===========================================================================

pub mod quantize {
    /// Symmetric f32 → int8 quantization. Returns (quantized, scale).
    pub fn f32_to_int8(src: &[f32]) -> (Vec<i8>, f32) {
        let abs_max = src
            .iter()
            .map(|x| x.abs())
            .fold(0.0f32, f32::max);
        let scale = if abs_max == 0.0 {
            1.0
        } else {
            abs_max / 127.0
        };
        let quantized: Vec<i8> = src
            .iter()
            .map(|&x| (x / scale).round().clamp(-127.0, 127.0) as i8)
            .collect();
        (quantized, scale)
    }

    /// int8 → f32 dequantization.
    pub fn int8_to_f32(src: &[i8], scale: f32) -> Vec<f32> {
        src.iter().map(|&x| x as f32 * scale).collect()
    }

    /// Scalar f32 → f16 conversion (IEEE 754 half-precision).
    pub fn f32_to_f16(src: &[f32]) -> Vec<u16> {
        src.iter().map(|&x| f32_to_f16_one(x)).collect()
    }

    /// Scalar f16 → f32 conversion.
    pub fn f16_to_f32(src: &[u16]) -> Vec<f32> {
        src.iter().map(|&x| f16_to_f32_one(x)).collect()
    }

    fn f32_to_f16_one(value: f32) -> u16 {
        let bits = value.to_bits();
        let sign = ((bits >> 16) & 0x8000) as u16;
        let exp = ((bits >> 23) & 0xFF) as i32;
        let mantissa = bits & 0x7F_FFFF;

        if exp == 0xFF {
            // Inf or NaN
            return sign | 0x7C00 | if mantissa != 0 { 0x0200 } else { 0 };
        }

        let new_exp = exp - 127 + 15;
        if new_exp >= 31 {
            return sign | 0x7C00; // overflow → Inf
        }
        if new_exp <= 0 {
            return sign; // underflow → zero (simplified)
        }

        sign | ((new_exp as u16) << 10) | ((mantissa >> 13) as u16)
    }

    fn f16_to_f32_one(value: u16) -> f32 {
        let sign = ((value >> 15) & 1) as u32;
        let exp = ((value >> 10) & 0x1F) as u32;
        let mantissa = (value & 0x3FF) as u32;

        if exp == 0 {
            if mantissa == 0 {
                return f32::from_bits(sign << 31);
            }
            // Subnormal — simplified
            let scale = 2.0f32.powi(-14) * (mantissa as f32 / 1024.0);
            return if sign == 1 { -scale } else { scale };
        }
        if exp == 31 {
            return if mantissa == 0 {
                f32::from_bits((sign << 31) | 0x7F80_0000)
            } else {
                f32::NAN
            };
        }

        let f_exp = (exp as i32 - 15 + 127) as u32;
        f32::from_bits((sign << 31) | (f_exp << 23) | (mantissa << 13))
    }
}

// ===========================================================================
// HNSW — Hierarchical Navigable Small World index
// ===========================================================================

/// Min-distance entry for BinaryHeap (reversed Ord → min-heap for candidates).
struct MinDist(f32, usize);

impl PartialEq for MinDist {
    fn eq(&self, other: &Self) -> bool {
        self.1 == other.1
    }
}
impl Eq for MinDist {}
impl PartialOrd for MinDist {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for MinDist {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .0
            .partial_cmp(&self.0)
            .unwrap_or(Ordering::Equal)
    }
}

/// Max-distance entry for BinaryHeap (natural Ord → max-heap for result bound).
struct MaxDist(f32, usize);

impl PartialEq for MaxDist {
    fn eq(&self, other: &Self) -> bool {
        self.1 == other.1
    }
}
impl Eq for MaxDist {}
impl PartialOrd for MaxDist {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for MaxDist {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0
            .partial_cmp(&other.0)
            .unwrap_or(Ordering::Equal)
    }
}

struct HnswNode {
    vector: AlignedVec,           // 64-byte aligned for SIMD
    norm_sq: f32,                 // cached ||vector||², avoids recomputation in distance
    connections: Vec<Vec<usize>>, // connections[level]
    level: usize,
}

pub struct HnswIndex {
    dim: usize,
    m: usize,
    m_max0: usize,
    ef_construction: usize,
    entry_point: Option<usize>,
    max_level: usize,
    level_mult: f64,
    nodes: Vec<HnswNode>,
    rng: SmallRng,
    // Generation-based visited tracking: avoids HashSet allocation per search.
    // visited[i] == visit_gen means node i was visited in the current search.
    visited: Vec<u32>,
    visit_gen: u32,
}

impl HnswIndex {
    pub fn new(dim: usize, m: usize, ef_construction: usize) -> Self {
        Self::with_seed(dim, m, ef_construction, 42)
    }

    pub fn with_seed(dim: usize, m: usize, ef_construction: usize, seed: u64) -> Self {
        Self {
            dim,
            m,
            m_max0: m * 2,
            ef_construction,
            entry_point: None,
            max_level: 0,
            level_mult: 1.0 / (m as f64).ln(),
            nodes: Vec::new(),
            rng: SmallRng::seed_from_u64(seed),
            visited: Vec::new(),
            visit_gen: 0,
        }
    }

    fn random_level(&mut self) -> usize {
        let r: f64 = self.rng.gen::<f64>().max(1e-10);
        let level = (-r.ln() * self.level_mult).floor() as usize;
        level.min(32)
    }

    /// L2² via cached norms: ||a||² + ||b||² - 2·dot(a,b).
    /// Reduces per-distance work from sub+FMA to just dot product.
    #[inline]
    fn distance_cached(&self, query: &[f32], query_norm: f32, node_idx: usize) -> f32 {
        let node = &self.nodes[node_idx];
        let d = query_norm + node.norm_sq - 2.0 * dot_product(query, &node.vector);
        // Clamp to avoid negative values from floating point error
        if d > 0.0 { d } else { 0.0 }
    }

    /// Advance the visit generation. Resets visited state in O(1).
    fn next_visit_gen(&mut self) {
        self.visit_gen = self.visit_gen.wrapping_add(1);
        if self.visit_gen == 0 {
            // Overflow: clear the array (happens once every ~4B searches)
            self.visited.fill(0);
            self.visit_gen = 1;
        }
    }

    /// Insert a vector. Returns its index.
    pub fn insert(&mut self, vector: Vec<f32>) -> usize {
        let idx = self.nodes.len();
        let level = self.random_level();
        let aligned = AlignedVec::from(vector);
        let norm_sq = scalar::norm_sq(&aligned);

        let mut connections = Vec::with_capacity(level + 1);
        for l in 0..=level {
            let cap = if l == 0 { self.m_max0 } else { self.m };
            connections.push(Vec::with_capacity(cap));
        }

        self.nodes.push(HnswNode {
            vector: aligned,
            norm_sq,
            connections,
            level,
        });

        // Extend visited array for the new node
        self.visited.push(0);

        if self.entry_point.is_none() {
            self.entry_point = Some(idx);
            self.max_level = level;
            return idx;
        }

        let mut ep = self.entry_point.unwrap();
        let query_norm = self.nodes[idx].norm_sq;

        // Descend from top level to level+1 (greedy, ef=1)
        if self.max_level > level {
            for l in ((level + 1)..=self.max_level).rev() {
                ep = self.search_layer_one(idx, query_norm, ep, l);
            }
        }

        // From min(level, max_level) down to 0: find ef_construction nearest, connect
        let top = level.min(self.max_level);
        for l in (0..=top).rev() {
            let neighbors = self.search_layer(idx, query_norm, ep, self.ef_construction, l);
            let m_max = if l == 0 { self.m_max0 } else { self.m };

            let selected: Vec<usize> = neighbors.iter().take(m_max).map(|&(n, _)| n).collect();

            self.nodes[idx].connections[l] = selected.clone();

            // Bidirectional connections + pruning
            for &neighbor in &selected {
                if neighbor < self.nodes.len() && l <= self.nodes[neighbor].level {
                    self.nodes[neighbor].connections[l].push(idx);
                    if self.nodes[neighbor].connections[l].len() > m_max {
                        self.prune_connections(neighbor, l, m_max);
                    }
                }
            }

            if let Some(&(first, _)) = neighbors.first() {
                ep = first;
            }
        }

        if level > self.max_level {
            self.entry_point = Some(idx);
            self.max_level = level;
        }

        idx
    }

    fn prune_connections(&mut self, node: usize, level: usize, m_max: usize) {
        let node_norm = self.nodes[node].norm_sq;
        let node_vec_ptr = self.nodes[node].vector.as_ptr();
        let dim = self.dim;

        let mut scored: Vec<(f32, usize)> = self.nodes[node].connections[level]
            .iter()
            .map(|&c| {
                // SAFETY: node_vec_ptr is valid for dim elements and we don't
                // modify the nodes vector during this closure.
                let dist = unsafe {
                    let node_slice = std::slice::from_raw_parts(node_vec_ptr, dim);
                    let d = node_norm + self.nodes[c].norm_sq
                        - 2.0 * dot_product(node_slice, &self.nodes[c].vector);
                    if d > 0.0 { d } else { 0.0 }
                };
                (dist, c)
            })
            .collect();
        scored.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
        scored.truncate(m_max);
        self.nodes[node].connections[level] = scored.into_iter().map(|(_, c)| c).collect();
    }

    /// Greedy single-nearest search at one layer.
    fn search_layer_one(&self, query_idx: usize, query_norm: f32, ep: usize, level: usize) -> usize {
        let query = &self.nodes[query_idx].vector;
        let mut current = ep;
        let mut current_dist = self.distance_cached(query, query_norm, ep);

        loop {
            let mut changed = false;
            if level <= self.nodes[current].level {
                for &neighbor in &self.nodes[current].connections[level] {
                    let dist = self.distance_cached(query, query_norm, neighbor);
                    if dist < current_dist {
                        current = neighbor;
                        current_dist = dist;
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }
        current
    }

    /// Beam search returning up to `ef` nearest neighbors sorted by distance.
    fn search_layer(
        &mut self,
        query_idx: usize,
        query_norm: f32,
        ep: usize,
        ef: usize,
        level: usize,
    ) -> Vec<(usize, f32)> {
        self.next_visit_gen();
        self.visited[ep] = self.visit_gen;

        let query_ptr = self.nodes[query_idx].vector.as_ptr();
        let dim = self.dim;

        // SAFETY: query_ptr remains valid because we only append to self.nodes
        // (never remove or reallocate vectors) during search. The query node at
        // query_idx was already pushed before this call.
        let query = unsafe { std::slice::from_raw_parts(query_ptr, dim) };

        let ep_dist = self.distance_cached(query, query_norm, ep);

        let mut candidates: BinaryHeap<MinDist> = BinaryHeap::new();
        candidates.push(MinDist(ep_dist, ep));

        // Max-heap for results: farthest on top, O(log ef) insert vs O(ef) Vec::insert
        let mut results: BinaryHeap<MaxDist> = BinaryHeap::with_capacity(ef + 1);
        results.push(MaxDist(ep_dist, ep));

        while let Some(MinDist(c_dist, c_idx)) = candidates.pop() {
            let worst = results.peek().map(|r| r.0).unwrap_or(f32::MAX);
            if c_dist > worst && results.len() >= ef {
                break;
            }

            if level <= self.nodes[c_idx].level {
                // Clone connections to release borrow on self.nodes, enabling
                // mutable access to self.visited. Clone is cheap (~128B for M=16)
                // and faster than index-based iteration (no bounds checks).
                let conns = self.nodes[c_idx].connections[level].clone();
                for neighbor in conns {
                    if self.visited[neighbor] == self.visit_gen {
                        continue;
                    }
                    self.visited[neighbor] = self.visit_gen;

                    let dist = self.distance_cached(query, query_norm, neighbor);
                    let worst = results.peek().map(|r| r.0).unwrap_or(f32::MAX);

                    if dist < worst || results.len() < ef {
                        candidates.push(MinDist(dist, neighbor));
                        results.push(MaxDist(dist, neighbor));
                        if results.len() > ef {
                            results.pop(); // remove farthest
                        }
                    }
                }
            }
        }

        // Drain into sorted Vec (ascending by distance)
        let mut sorted: Vec<(usize, f32)> = results.into_iter().map(|MaxDist(d, i)| (i, d)).collect();
        sorted.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        sorted
    }

    /// Beam search for external queries (not in the index).
    fn search_layer_ext(
        &mut self,
        query: &[f32],
        query_norm: f32,
        ep: usize,
        ef: usize,
        level: usize,
    ) -> Vec<(usize, f32)> {
        self.next_visit_gen();
        self.visited[ep] = self.visit_gen;

        let ep_dist = self.distance_cached(query, query_norm, ep);

        let mut candidates: BinaryHeap<MinDist> = BinaryHeap::new();
        candidates.push(MinDist(ep_dist, ep));

        let mut results: BinaryHeap<MaxDist> = BinaryHeap::with_capacity(ef + 1);
        results.push(MaxDist(ep_dist, ep));

        while let Some(MinDist(c_dist, c_idx)) = candidates.pop() {
            let worst = results.peek().map(|r| r.0).unwrap_or(f32::MAX);
            if c_dist > worst && results.len() >= ef {
                break;
            }

            if level <= self.nodes[c_idx].level {
                let conns = self.nodes[c_idx].connections[level].clone();
                for neighbor in conns {
                    if self.visited[neighbor] == self.visit_gen {
                        continue;
                    }
                    self.visited[neighbor] = self.visit_gen;

                    let dist = self.distance_cached(query, query_norm, neighbor);
                    let worst = results.peek().map(|r| r.0).unwrap_or(f32::MAX);

                    if dist < worst || results.len() < ef {
                        candidates.push(MinDist(dist, neighbor));
                        results.push(MaxDist(dist, neighbor));
                        if results.len() > ef {
                            results.pop();
                        }
                    }
                }
            }
        }

        let mut sorted: Vec<(usize, f32)> = results.into_iter().map(|MaxDist(d, i)| (i, d)).collect();
        sorted.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        sorted
    }

    /// Search for the k nearest neighbors of `query`.
    pub fn search(&mut self, query: &[f32], k: usize, ef: usize) -> Vec<(usize, f32)> {
        if self.entry_point.is_none() || self.nodes.is_empty() {
            return Vec::new();
        }

        let query_norm = scalar::norm_sq(query);
        let mut ep = self.entry_point.unwrap();
        let mut ep_dist = self.distance_cached(query, query_norm, ep);

        // Greedy descent from top to level 1
        for l in (1..=self.max_level).rev() {
            loop {
                let mut changed = false;
                if l <= self.nodes[ep].level {
                    for &neighbor in &self.nodes[ep].connections[l] {
                        let dist = self.distance_cached(query, query_norm, neighbor);
                        if dist < ep_dist {
                            ep = neighbor;
                            ep_dist = dist;
                            changed = true;
                        }
                    }
                }
                if !changed {
                    break;
                }
            }
        }

        // Full beam search at level 0
        let mut results = self.search_layer_ext(query, query_norm, ep, ef.max(k), 0);
        results.truncate(k);
        results
    }

    /// Access a stored vector by index (for exact re-ranking after approximate search).
    #[inline]
    pub fn get_vector(&self, idx: usize) -> &[f32] {
        &self.nodes[idx].vector
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}
