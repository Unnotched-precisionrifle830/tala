//! Criterion benchmarks for TBF binary format operations.
//!
//! Targets from spec-01:
//!   Node insertion: < 1ms
//!   Query latency:  < 50ms

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::Rng;
use tala_wire::*;

fn random_id(rng: &mut impl Rng) -> [u8; 16] {
    let mut id = [0u8; 16];
    rng.fill(&mut id);
    id
}

fn random_embedding(dim: usize, rng: &mut impl Rng) -> Vec<f32> {
    (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect()
}

// ---------------------------------------------------------------------------
// Columnar write: serialize N nodes into columnar layout
// ---------------------------------------------------------------------------

fn bench_columnar_write(c: &mut Criterion) {
    let mut group = c.benchmark_group("columnar_write");
    for &n in &[1_000usize, 10_000, 100_000] {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            let mut rng = rand::thread_rng();
            let ids: Vec<_> = (0..n).map(|_| random_id(&mut rng)).collect();
            let timestamps: Vec<u64> = (0..n).map(|i| i as u64 * 1_000_000).collect();
            let hashes: Vec<u64> = (0..n).map(|_| rng.gen()).collect();
            let confs: Vec<f32> = (0..n).map(|_| rng.gen()).collect();
            let statuses: Vec<u8> = (0..n).map(|_| rng.gen_range(0u8..4)).collect();

            b.iter(|| {
                let mut col = ColumnarBuffer::with_capacity(n);
                for i in 0..n {
                    col.push(&ids[i], timestamps[i], hashes[i], confs[i], statuses[i]);
                }
                col.serialize()
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Columnar read: sequential scan over timestamp column
// ---------------------------------------------------------------------------

fn bench_columnar_scan(c: &mut Criterion) {
    let mut group = c.benchmark_group("columnar_scan");
    for &n in &[1_000usize, 10_000, 100_000] {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            let mut rng = rand::thread_rng();
            let mut col = ColumnarBuffer::with_capacity(n);
            for _ in 0..n {
                col.push(
                    &random_id(&mut rng),
                    rng.gen(),
                    rng.gen(),
                    rng.gen(),
                    rng.gen_range(0u8..4),
                );
            }
            let (data, offsets) = col.serialize();
            let reader = ColumnReader::new(&data);
            let ts_offset = offsets[1];

            b.iter(|| {
                let mut sum = 0u64;
                for i in 0..n {
                    sum = sum.wrapping_add(reader.read_u64(ts_offset, i));
                }
                sum
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Embedding write: push N aligned vectors
// ---------------------------------------------------------------------------

fn bench_embedding_write(c: &mut Criterion) {
    let mut group = c.benchmark_group("embedding_write");
    let dim = 384;
    for &n in &[1_000usize, 10_000, 100_000] {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            let mut rng = rand::thread_rng();
            let vecs: Vec<Vec<f32>> = (0..n).map(|_| random_embedding(dim, &mut rng)).collect();

            b.iter(|| {
                let mut writer = EmbeddingWriter::new(dim);
                for v in &vecs {
                    writer.push(v);
                }
                writer.as_bytes().len()
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Embedding read: sequential vs random access
// ---------------------------------------------------------------------------

fn bench_embedding_read(c: &mut Criterion) {
    let mut group = c.benchmark_group("embedding_read");
    let dim = 384;
    let n = 100_000usize;

    let mut rng = rand::thread_rng();
    let mut writer = EmbeddingWriter::new(dim);
    for _ in 0..n {
        writer.push(&random_embedding(dim, &mut rng));
    }
    let data = writer.as_bytes().to_vec();
    let reader = EmbeddingReader::new(&data, dim);

    // Sequential
    group.throughput(Throughput::Elements(n as u64));
    group.bench_function("sequential_100k", |b| {
        b.iter(|| {
            let mut sum = 0.0f32;
            for i in 0..n {
                sum += reader.get(i)[0];
            }
            sum
        });
    });

    // Random access
    let indices: Vec<usize> = (0..n).map(|_| rng.gen_range(0..n)).collect();
    group.bench_function("random_100k", |b| {
        b.iter(|| {
            let mut sum = 0.0f32;
            for &i in &indices {
                sum += reader.get(i)[0];
            }
            sum
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// CSR: build and traverse
// ---------------------------------------------------------------------------

fn bench_csr_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("csr_build");
    for &n in &[1_000usize, 10_000, 100_000] {
        let avg_degree = 5;
        group.throughput(Throughput::Elements((n * avg_degree) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            let mut rng = rand::thread_rng();
            let edges: Vec<(usize, usize, u8, f32)> = (0..n)
                .flat_map(|from| {
                    (0..avg_degree).map(move |_| {
                        let to = from.wrapping_add(1 + (rand::random::<usize>() % 100)).min(n - 1);
                        (from, to, 0u8, 1.0f32)
                    })
                })
                .collect();

            b.iter(|| {
                let mut builder = CsrBuilder::new(n);
                for &(from, to, rel, w) in &edges {
                    builder.add_edge(from, to, rel, w);
                }
                builder.build()
            });
        });
    }
    group.finish();
}

fn bench_csr_traverse(c: &mut Criterion) {
    let mut group = c.benchmark_group("csr_traverse");
    let n = 100_000usize;
    let avg_degree = 5;

    let mut rng = rand::thread_rng();
    let mut builder = CsrBuilder::new(n);
    for from in 0..n {
        for _ in 0..avg_degree {
            let to = (from + 1 + rng.gen_range(0..100)).min(n - 1);
            builder.add_edge(from, to, 0, 1.0);
        }
    }
    let index = builder.build();

    let query_nodes: Vec<usize> = (0..10_000).map(|_| rng.gen_range(0..n)).collect();

    group.throughput(Throughput::Elements(10_000));
    group.bench_function("lookup_10k_nodes", |b| {
        b.iter(|| {
            let mut total_edges = 0usize;
            for &node in &query_nodes {
                total_edges += index.edges_from(node).len();
            }
            total_edges
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Bloom filter: insert and lookup
// ---------------------------------------------------------------------------

fn bench_bloom(c: &mut Criterion) {
    let mut group = c.benchmark_group("bloom_filter");

    for &n in &[10_000usize, 100_000] {
        let mut rng = rand::thread_rng();
        let keys: Vec<[u8; 16]> = (0..n).map(|_| random_id(&mut rng)).collect();
        let absent_keys: Vec<[u8; 16]> = (0..1_000).map(|_| random_id(&mut rng)).collect();

        group.bench_with_input(BenchmarkId::new("insert", n), &n, |b, &n| {
            b.iter(|| {
                let mut bloom = BloomFilter::new(n, 0.01);
                for key in &keys {
                    bloom.insert(key);
                }
                bloom.size_bytes()
            });
        });

        // Build bloom once for lookup bench
        let mut bloom = BloomFilter::new(n, 0.01);
        for key in &keys {
            bloom.insert(key);
        }

        group.bench_with_input(BenchmarkId::new("lookup_hit", n), &n, |b, _| {
            b.iter(|| {
                let mut hits = 0u32;
                for key in &keys[..1_000] {
                    if bloom.contains(key) {
                        hits += 1;
                    }
                }
                hits
            });
        });

        group.bench_with_input(BenchmarkId::new("lookup_miss", n), &n, |b, _| {
            b.iter(|| {
                let mut hits = 0u32;
                for key in &absent_keys {
                    if bloom.contains(key) {
                        hits += 1;
                    }
                }
                hits
            });
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Full segment write: end-to-end
// ---------------------------------------------------------------------------

fn bench_segment_write(c: &mut Criterion) {
    let mut group = c.benchmark_group("segment_write");
    let dim = 384;

    for &n in &[1_000usize, 10_000, 100_000] {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            let mut rng = rand::thread_rng();
            let ids: Vec<[u8; 16]> = (0..n).map(|_| random_id(&mut rng)).collect();
            let embeddings: Vec<Vec<f32>> =
                (0..n).map(|_| random_embedding(dim, &mut rng)).collect();

            b.iter(|| {
                let mut writer = SegmentWriter::new(dim);
                for i in 0..n {
                    writer.push_node(
                        &ids[i],
                        i as u64 * 1_000_000,
                        rng.gen(),
                        rng.gen(),
                        1u8,
                        &embeddings[i],
                    );
                    // Add ~3 edges per node
                    if i > 0 {
                        for j in 1..=3.min(i) {
                            writer.add_edge(i - j, i, 0, 0.9);
                        }
                    }
                }
                writer.finish().len()
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_columnar_write,
    bench_columnar_scan,
    bench_embedding_write,
    bench_embedding_read,
    bench_csr_build,
    bench_csr_traverse,
    bench_bloom,
    bench_segment_write,
);
criterion_main!(benches);
