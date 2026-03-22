//! Criterion benchmarks for TALA embedding engine.
//!
//! Targets from spec-02:
//!   Single cosine (dim=384, f32):     < 20ns  (AVX2)
//!   Batch cosine (1K vectors):        < 50µs  (AVX2, single core)
//!   Batch cosine (100K vectors):      < 5ms   (AVX2, 16 cores)
//!   HNSW search (10K vectors, top-10): < 1ms  (AVX2, ef=50)
//!   f32→int8 quantize (1K × 384):    < 1ms

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use tala_embed::*;

fn random_vec(dim: usize, rng: &mut impl Rng) -> Vec<f32> {
    (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect()
}

fn random_aligned(dim: usize, rng: &mut impl Rng) -> AlignedVec {
    let mut v = AlignedVec::new(dim);
    for x in v.as_mut_slice() {
        *x = rng.gen::<f32>() * 2.0 - 1.0;
    }
    v
}

fn random_corpus(n: usize, dim: usize, rng: &mut impl Rng) -> Vec<f32> {
    (0..n * dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect()
}

// ===========================================================================
// Cosine similarity: scalar vs SIMD vs dispatched
// ===========================================================================

fn bench_cosine(c: &mut Criterion) {
    let mut group = c.benchmark_group("cosine_similarity");
    let dims = [128, 256, 384, 512, 768];
    let mut rng = SmallRng::seed_from_u64(42);

    for &dim in &dims {
        let a = random_vec(dim, &mut rng);
        let b = random_vec(dim, &mut rng);

        group.bench_with_input(BenchmarkId::new("scalar", dim), &dim, |bench, _| {
            bench.iter(|| scalar::cosine_similarity(&a, &b))
        });

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                let a_ref = &a;
                let b_ref = &b;
                group.bench_with_input(BenchmarkId::new("avx2", dim), &dim, |bench, _| {
                    bench.iter(|| unsafe { avx2::cosine_similarity(a_ref, b_ref) })
                });
            }
        }

        group.bench_with_input(BenchmarkId::new("dispatch", dim), &dim, |bench, _| {
            bench.iter(|| cosine_similarity(&a, &b))
        });
    }

    group.finish();
}

// ===========================================================================
// Dot product
// ===========================================================================

fn bench_dot_product(c: &mut Criterion) {
    let mut group = c.benchmark_group("dot_product");
    let mut rng = SmallRng::seed_from_u64(42);

    for &dim in &[384, 768] {
        let a = random_vec(dim, &mut rng);
        let b = random_vec(dim, &mut rng);

        group.bench_with_input(BenchmarkId::new("scalar", dim), &dim, |bench, _| {
            bench.iter(|| scalar::dot_product(&a, &b))
        });

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                let a_ref = &a;
                let b_ref = &b;
                group.bench_with_input(BenchmarkId::new("avx2", dim), &dim, |bench, _| {
                    bench.iter(|| unsafe { avx2::dot_product(a_ref, b_ref) })
                });
            }
        }
    }

    group.finish();
}

// ===========================================================================
// Batch cosine: single-threaded and parallel
// ===========================================================================

fn bench_batch_cosine(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_cosine");
    let dim = 384;
    let mut rng = SmallRng::seed_from_u64(42);

    for &n in &[1_000usize, 10_000, 100_000] {
        let query = random_vec(dim, &mut rng);
        let corpus = random_corpus(n, dim, &mut rng);
        let mut results = vec![0.0f32; n];

        group.throughput(Throughput::Elements(n as u64));

        group.bench_with_input(BenchmarkId::new("single_thread", n), &n, |b, _| {
            b.iter(|| {
                batch_cosine(&query, &corpus, dim, &mut results);
                results[0]
            })
        });

        if n >= 10_000 {
            group.bench_with_input(BenchmarkId::new("parallel", n), &n, |b, _| {
                b.iter(|| {
                    batch_cosine_parallel(&query, &corpus, dim, &mut results);
                    results[0]
                })
            });
        }
    }

    group.finish();
}

// ===========================================================================
// Quantization: f32 → int8 and f32 → f16
// ===========================================================================

fn bench_quantize(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantization");
    let dim = 384;
    let mut rng = SmallRng::seed_from_u64(42);

    for &n in &[1_000usize, 10_000] {
        let vectors: Vec<Vec<f32>> = (0..n).map(|_| random_vec(dim, &mut rng)).collect();

        group.throughput(Throughput::Elements(n as u64));

        group.bench_with_input(BenchmarkId::new("f32_to_int8", n), &n, |b, _| {
            b.iter(|| {
                let mut total_bytes = 0usize;
                for v in &vectors {
                    let (q, _scale) = quantize::f32_to_int8(v);
                    total_bytes += q.len();
                }
                total_bytes
            })
        });

        group.bench_with_input(BenchmarkId::new("f32_to_f16", n), &n, |b, _| {
            b.iter(|| {
                let mut total = 0usize;
                for v in &vectors {
                    total += quantize::f32_to_f16(v).len();
                }
                total
            })
        });

        // Roundtrip: f32 → int8 → f32
        let quantized: Vec<(Vec<i8>, f32)> =
            vectors.iter().map(|v| quantize::f32_to_int8(v)).collect();
        group.bench_with_input(BenchmarkId::new("int8_to_f32", n), &n, |b, _| {
            b.iter(|| {
                let mut total = 0usize;
                for (q, scale) in &quantized {
                    total += quantize::int8_to_f32(q, *scale).len();
                }
                total
            })
        });
    }

    group.finish();
}

// ===========================================================================
// HNSW: insert throughput
// ===========================================================================

fn bench_hnsw_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_insert");
    let dim = 384;

    for &n in &[1_000usize, 5_000, 10_000] {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            let mut rng = SmallRng::seed_from_u64(42);
            let vectors: Vec<Vec<f32>> = (0..n).map(|_| random_vec(dim, &mut rng)).collect();

            b.iter(|| {
                let mut index = HnswIndex::with_seed(dim, 16, 100, 42);
                for v in &vectors {
                    index.insert(v.clone());
                }
                index.len()
            });
        });
    }

    group.finish();
}

// ===========================================================================
// HNSW: search latency at different ef values
// ===========================================================================

fn bench_hnsw_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_search");
    let dim = 384;
    let n = 10_000;

    // Build index once
    let mut rng = SmallRng::seed_from_u64(42);
    let mut index = HnswIndex::with_seed(dim, 16, 100, 42);
    for _ in 0..n {
        index.insert(random_vec(dim, &mut rng));
    }

    let query = random_vec(dim, &mut rng);

    for &ef in &[10, 50, 100, 200] {
        group.bench_with_input(BenchmarkId::new("ef", ef), &ef, |b, &ef| {
            b.iter(|| index.search(&query, 10, ef))
        });
    }

    group.finish();
}

// ===========================================================================
// Aligned allocation overhead
// ===========================================================================

fn bench_aligned_alloc(c: &mut Criterion) {
    let mut group = c.benchmark_group("aligned_alloc");
    let mut rng = SmallRng::seed_from_u64(42);

    for &dim in &[384, 768] {
        group.bench_with_input(BenchmarkId::new("alloc_fill", dim), &dim, |b, &dim| {
            b.iter(|| {
                let mut v = AlignedVec::new(dim);
                for x in v.as_mut_slice() {
                    *x = 1.0;
                }
                v.as_slice()[0]
            })
        });

        let a = random_aligned(dim, &mut rng);
        let b = random_aligned(dim, &mut rng);
        group.bench_with_input(
            BenchmarkId::new("cosine_aligned", dim),
            &dim,
            |bench, _| bench.iter(|| cosine_similarity(a.as_slice(), b.as_slice())),
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_cosine,
    bench_dot_product,
    bench_batch_cosine,
    bench_quantize,
    bench_hnsw_insert,
    bench_hnsw_search,
    bench_aligned_alloc,
);
criterion_main!(benches);
