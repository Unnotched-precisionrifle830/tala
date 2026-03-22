//! Criterion benchmarks for TALA storage engine.
//!
//! Tests WAL throughput, hot buffer flush, and end-to-end semantic query latency.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use tala_store::*;

fn random_id(rng: &mut impl Rng) -> [u8; 16] {
    let mut id = [0u8; 16];
    rng.fill(&mut id);
    id
}

fn random_embedding(dim: usize, rng: &mut impl Rng) -> Vec<f32> {
    (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect()
}

// ---------------------------------------------------------------------------
// WAL: append throughput
// ---------------------------------------------------------------------------

fn bench_wal_append(c: &mut Criterion) {
    let mut group = c.benchmark_group("wal_append");
    let dim = 384;

    for &n in &[100usize, 1_000, 10_000] {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            let mut rng = SmallRng::seed_from_u64(42);
            let ids: Vec<[u8; 16]> = (0..n).map(|_| random_id(&mut rng)).collect();
            let embeddings: Vec<Vec<f32>> =
                (0..n).map(|_| random_embedding(dim, &mut rng)).collect();
            let cmds: Vec<String> = (0..n).map(|i| format!("kubectl apply -f deploy_{}.yaml", i)).collect();

            b.iter(|| {
                let dir = tempfile::tempdir().unwrap();
                let mut wal = Wal::create(dir.path().join("bench.wal")).unwrap();
                for i in 0..n {
                    wal.append(&ids[i], i as u64, &embeddings[i], &cmds[i]).unwrap();
                }
                wal.sync().unwrap();
                wal.entry_count()
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// WAL: replay throughput
// ---------------------------------------------------------------------------

fn bench_wal_replay(c: &mut Criterion) {
    let mut group = c.benchmark_group("wal_replay");
    let dim = 384;

    for &n in &[1_000usize, 10_000] {
        // Write WAL once
        let dir = tempfile::tempdir().unwrap();
        let wal_path = dir.path().join("bench.wal");
        {
            let mut rng = SmallRng::seed_from_u64(42);
            let mut wal = Wal::create(&wal_path).unwrap();
            for i in 0..n {
                let id = random_id(&mut rng);
                let emb = random_embedding(dim, &mut rng);
                wal.append(&id, i as u64, &emb, &format!("cmd_{}", i)).unwrap();
            }
            wal.sync().unwrap();
        }

        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                let entries = replay_wal(&wal_path).unwrap();
                entries.len()
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Hot buffer: flush to TBF segment
// ---------------------------------------------------------------------------

fn bench_hot_flush(c: &mut Criterion) {
    let mut group = c.benchmark_group("hot_flush");
    let dim = 384;

    for &n in &[1_000usize, 10_000, 64_000] {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            let mut rng = SmallRng::seed_from_u64(42);

            b.iter(|| {
                let mut buf = HotBuffer::new(dim, n + 1);
                for i in 0..n {
                    buf.push(
                        random_id(&mut rng),
                        i as u64,
                        rng.gen(),
                        rng.gen(),
                        1u8,
                        random_embedding(dim, &mut rng),
                        if i > 0 { vec![i - 1] } else { vec![] },
                    );
                }
                let segment = buf.flush();
                segment.len()
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Semantic query: build HNSW index, then search
// ---------------------------------------------------------------------------

fn bench_semantic_query(c: &mut Criterion) {
    let mut group = c.benchmark_group("semantic_query");
    let dim = 384;

    for &n in &[1_000usize, 10_000] {
        let mut rng = SmallRng::seed_from_u64(42);
        let mut engine = QueryEngine::new(dim);
        for i in 0..n {
            engine.insert(
                random_id(&mut rng),
                i as u64,
                format!("command_{}", i),
                random_embedding(dim, &mut rng),
            );
        }

        let query = random_embedding(dim, &mut rng);

        group.bench_with_input(
            BenchmarkId::new("search_top10", n),
            &n,
            |b, _| b.iter(|| engine.search(&query, 10)),
        );

        group.bench_with_input(
            BenchmarkId::new("search_top50", n),
            &n,
            |b, _| b.iter(|| engine.search(&query, 50)),
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Edge candidate selection: HNSW-based O(log n) vs brute-force O(n)
// ---------------------------------------------------------------------------

fn bench_edge_candidates(c: &mut Criterion) {
    let mut group = c.benchmark_group("edge_candidates");
    let dim = 384;
    let k = 5;

    for &n in &[1_000usize, 10_000] {
        let mut rng = SmallRng::seed_from_u64(42);
        let mut engine = QueryEngine::new(dim);
        let embeddings: Vec<Vec<f32>> =
            (0..n).map(|_| random_embedding(dim, &mut rng)).collect();
        for i in 0..n {
            engine.insert(
                random_id(&mut rng),
                i as u64,
                format!("cmd_{}", i),
                embeddings[i].clone(),
            );
        }

        let query = random_embedding(dim, &mut rng);

        // HNSW-based: O(log n) candidate selection + exact re-rank
        group.bench_with_input(
            BenchmarkId::new("hnsw", n),
            &n,
            |b, _| b.iter(|| engine.find_edge_candidates(&query, k)),
        );

        // Brute-force: O(n) cosine against all stored vectors
        group.bench_with_input(
            BenchmarkId::new("bruteforce", n),
            &n,
            |b, _| {
                b.iter(|| {
                    let mut sims: Vec<f32> = Vec::with_capacity(n);
                    for i in 0..n {
                        sims.push(tala_embed::cosine_similarity(&query, &embeddings[i]));
                    }
                    sims.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap());
                    sims.truncate(k);
                    sims
                })
            },
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// End-to-end ingest: WAL append + hot buffer fill + flush
// ---------------------------------------------------------------------------

fn bench_ingest_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("ingest_pipeline");
    let dim = 384;

    for &n in &[1_000usize, 10_000] {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            let mut rng = SmallRng::seed_from_u64(42);
            let ids: Vec<[u8; 16]> = (0..n).map(|_| random_id(&mut rng)).collect();
            let embeddings: Vec<Vec<f32>> =
                (0..n).map(|_| random_embedding(dim, &mut rng)).collect();
            let cmds: Vec<String> = (0..n).map(|i| format!("cmd_{}", i)).collect();

            b.iter(|| {
                let dir = tempfile::tempdir().unwrap();

                // Phase 1: WAL
                let mut wal = Wal::create(dir.path().join("ingest.wal")).unwrap();
                for i in 0..n {
                    wal.append(&ids[i], i as u64, &embeddings[i], &cmds[i]).unwrap();
                }
                wal.sync().unwrap();

                // Phase 2: Hot buffer fill + flush to segment
                let mut buf = HotBuffer::new(dim, n + 1);
                for i in 0..n {
                    buf.push(
                        ids[i],
                        i as u64,
                        0,
                        0.95,
                        1,
                        embeddings[i].clone(),
                        if i > 0 { vec![i - 1] } else { vec![] },
                    );
                }
                let segment = buf.flush();
                segment.len()
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_wal_append,
    bench_wal_replay,
    bench_hot_flush,
    bench_semantic_query,
    bench_edge_candidates,
    bench_ingest_pipeline,
);
criterion_main!(benches);
