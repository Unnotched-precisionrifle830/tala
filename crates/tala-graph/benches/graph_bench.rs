//! Criterion benchmarks for TALA narrative graph operations.
//!
//! Tests graph construction, edge formation, BFS traversal, and narrative extraction
//! at various scales and depths.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use tala_core::IntentId;
use tala_graph::NarrativeGraph;

/// Build a graph of `n` nodes with `avg_degree` random forward edges each.
fn build_graph(n: usize, avg_degree: usize, rng: &mut impl Rng) -> (NarrativeGraph, Vec<IntentId>) {
    let mut graph = NarrativeGraph::new();
    let ids: Vec<IntentId> = (0..n).map(|_| IntentId::random()).collect();

    for (i, &id) in ids.iter().enumerate() {
        graph.insert_node(id, i as u64 * 1_000_000, 0.95);
    }

    for (i, &id) in ids.iter().enumerate() {
        for _ in 0..avg_degree {
            let target_idx = rng.gen_range(0..n);
            if target_idx != i {
                graph.add_edge(
                    id,
                    ids[target_idx],
                    tala_core::RelationType::Causal,
                    rng.gen::<f32>(),
                );
            }
        }
    }

    (graph, ids)
}

// ---------------------------------------------------------------------------
// Node insertion (no edges)
// ---------------------------------------------------------------------------

fn bench_node_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("node_insert");
    for &n in &[1_000usize, 10_000, 100_000] {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            let ids: Vec<IntentId> = (0..n).map(|_| IntentId::random()).collect();
            b.iter(|| {
                let mut graph = NarrativeGraph::new();
                for (i, &id) in ids.iter().enumerate() {
                    graph.insert_node(id, i as u64, 0.95);
                }
                graph.node_count()
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Edge formation: insert node + connect to top-K similar existing nodes
// ---------------------------------------------------------------------------

fn bench_edge_formation(c: &mut Criterion) {
    let mut group = c.benchmark_group("edge_formation");
    let mut rng = SmallRng::seed_from_u64(42);

    for &n in &[1_000usize, 5_000, 10_000] {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            // Pre-generate similarity scores for each node
            let ids: Vec<IntentId> = (0..n).map(|_| IntentId::random()).collect();
            let k = 5;

            b.iter(|| {
                let mut graph = NarrativeGraph::new();
                for (i, &id) in ids.iter().enumerate() {
                    graph.insert_node(id, i as u64, 0.95);
                    if i > 0 {
                        // Generate random similarity scores against existing nodes
                        let mut similarities: Vec<(IntentId, f32)> = (0..i)
                            .map(|j| (ids[j], rng.gen::<f32>()))
                            .collect();
                        graph.form_edges(id, &mut similarities, k);
                    }
                }
                graph.edge_count()
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// BFS traversal at varying depths
// ---------------------------------------------------------------------------

fn bench_bfs_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("bfs_forward");
    let mut rng = SmallRng::seed_from_u64(42);

    for &n in &[10_000usize, 100_000] {
        let (graph, ids) = build_graph(n, 5, &mut rng);
        let start = ids[0];

        for &depth in &[1, 3, 5, 10] {
            group.bench_with_input(
                BenchmarkId::new(format!("n={n}"), depth),
                &depth,
                |b, &depth| {
                    b.iter(|| graph.bfs_forward(start, depth))
                },
            );
        }
    }
    group.finish();
}

fn bench_bfs_backward(c: &mut Criterion) {
    let mut group = c.benchmark_group("bfs_backward");
    let mut rng = SmallRng::seed_from_u64(42);

    let n = 100_000;
    let (graph, ids) = build_graph(n, 5, &mut rng);
    let start = ids[n / 2]; // middle node

    for &depth in &[1, 3, 5, 10] {
        group.bench_with_input(BenchmarkId::from_parameter(depth), &depth, |b, &depth| {
            b.iter(|| graph.bfs_backward(start, depth))
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Narrative extraction (bidirectional BFS)
// ---------------------------------------------------------------------------

fn bench_narrative_extraction(c: &mut Criterion) {
    let mut group = c.benchmark_group("narrative_extraction");
    let mut rng = SmallRng::seed_from_u64(42);

    for &n in &[10_000usize, 100_000] {
        let (graph, ids) = build_graph(n, 5, &mut rng);
        let root = ids[n / 2];

        for &depth in &[3, 5] {
            group.bench_with_input(
                BenchmarkId::new(format!("n={n}"), depth),
                &depth,
                |b, &depth| {
                    b.iter(|| {
                        let (nodes, edges) = graph.extract_narrative(root, depth);
                        (nodes.len(), edges.len())
                    })
                },
            );
        }
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Graph scaling: edge count growth
// ---------------------------------------------------------------------------

fn bench_graph_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph_scaling");
    let mut rng = SmallRng::seed_from_u64(42);

    for &degree in &[2, 5, 10, 20] {
        let n = 10_000;
        group.bench_with_input(
            BenchmarkId::new("build_10k_degree", degree),
            &degree,
            |b, &degree| {
                let ids: Vec<IntentId> = (0..n).map(|_| IntentId::random()).collect();
                b.iter(|| {
                    let mut graph = NarrativeGraph::new();
                    for (i, &id) in ids.iter().enumerate() {
                        graph.insert_node(id, i as u64, 0.95);
                    }
                    for (i, &id) in ids.iter().enumerate() {
                        for _ in 0..degree {
                            let target = rng.gen_range(0..n);
                            if target != i {
                                graph.add_edge(
                                    id,
                                    ids[target],
                                    tala_core::RelationType::Temporal,
                                    rng.gen(),
                                );
                            }
                        }
                    }
                    graph.edge_count()
                })
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_node_insert,
    bench_edge_formation,
    bench_bfs_forward,
    bench_bfs_backward,
    bench_narrative_extraction,
    bench_graph_scaling,
);
criterion_main!(benches);
