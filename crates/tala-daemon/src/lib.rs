//! TALA Daemon — Top-level orchestrator tying all TALA subsystems together.
//!
//! Provides the `IngestPipeline` (raw command -> stored intent with graph edges),
//! the `Daemon` facade (ingest, query, replay, insights), and a `DaemonBuilder`
//! for configuration.
//!
//! This is a library crate. The binary, Unix socket, and TCP server are future work.

use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use tala_core::{
    Context, Insight, IntentExtractor, IntentId, IntentStore, ReplayStep, TalaError,
};
use tala_graph::NarrativeGraph;
use tala_intent::IntentPipeline;
use tala_kai::InsightEngine;
use tala_store::StorageEngine;

// ===========================================================================
// Daemon-level Metrics
// ===========================================================================

/// Per-phase timing metrics for the daemon pipeline. All times in nanoseconds.
pub struct DaemonMetrics {
    pub extract_ns: AtomicU64,
    pub extract_count: AtomicU64,
    pub store_insert_ns: AtomicU64,
    pub store_insert_count: AtomicU64,
    pub edge_formation_ns: AtomicU64,
    pub edge_formation_count: AtomicU64,
    pub query_ns: AtomicU64,
    pub query_count: AtomicU64,
    pub replay_ns: AtomicU64,
    pub replay_count: AtomicU64,
    pub insight_ns: AtomicU64,
    pub insight_count: AtomicU64,
}

impl DaemonMetrics {
    pub fn new() -> Self {
        Self {
            extract_ns: AtomicU64::new(0),
            extract_count: AtomicU64::new(0),
            store_insert_ns: AtomicU64::new(0),
            store_insert_count: AtomicU64::new(0),
            edge_formation_ns: AtomicU64::new(0),
            edge_formation_count: AtomicU64::new(0),
            query_ns: AtomicU64::new(0),
            query_count: AtomicU64::new(0),
            replay_ns: AtomicU64::new(0),
            replay_count: AtomicU64::new(0),
            insight_ns: AtomicU64::new(0),
            insight_count: AtomicU64::new(0),
        }
    }
}

impl Default for DaemonMetrics {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// IngestPipeline
// ===========================================================================

/// The core ingest path: raw command string -> stored intent with graph edges.
///
/// Pipeline steps:
/// 1. Extract an `Intent` via `IntentPipeline` (tokenize, embed, classify).
/// 2. Insert into `StorageEngine` (WAL, hot buffer, HNSW index).
/// 3. Find edge candidates via semantic query on the store.
/// 4. Form edges in the `NarrativeGraph`.
pub struct IngestPipeline {
    extractor: IntentPipeline,
    store: StorageEngine,
    graph: Mutex<NarrativeGraph>,
    daemon_metrics: Arc<DaemonMetrics>,
}

impl IngestPipeline {
    /// Create a new ingest pipeline.
    fn new(store: StorageEngine, daemon_metrics: Arc<DaemonMetrics>) -> Self {
        Self {
            extractor: IntentPipeline::new(),
            store,
            graph: Mutex::new(NarrativeGraph::new()),
            daemon_metrics,
        }
    }

    /// Ingest a raw command string with the given execution context.
    ///
    /// Returns the `IntentId` assigned to the newly created intent.
    pub fn ingest(&self, raw: &str, context: &Context) -> Result<IntentId, TalaError> {
        // 1. Extract intent from raw command.
        let extract_t0 = Instant::now();
        let intent = self.extractor.extract(raw, context)?;
        let extract_ns = extract_t0.elapsed().as_nanos() as u64;
        self.daemon_metrics.extract_ns.fetch_add(extract_ns, Relaxed);
        self.daemon_metrics.extract_count.fetch_add(1, Relaxed);

        let id = intent.id;
        let embedding = intent.embedding.clone();
        let timestamp = intent.timestamp;
        let confidence = intent.confidence;

        // 2. Insert into storage engine (WAL + HNSW + hot buffer).
        let store_t0 = Instant::now();
        self.store.insert(intent)?;
        let store_ns = store_t0.elapsed().as_nanos() as u64;
        self.daemon_metrics
            .store_insert_ns
            .fetch_add(store_ns, Relaxed);
        self.daemon_metrics.store_insert_count.fetch_add(1, Relaxed);

        // 3. Find edge candidates via semantic search. We search for the top-5
        //    nearest neighbors, excluding self (which may or may not appear
        //    depending on HNSW timing). The store returns (IntentId, cosine_sim).
        let edge_t0 = Instant::now();
        let edge_k = 5;
        let mut candidates = self.store.query_semantic(&embedding, edge_k + 1)?;

        // Remove self from candidates if present.
        candidates.retain(|&(cand_id, _)| cand_id != id);
        candidates.truncate(edge_k);

        // 4. Insert node and form edges in the narrative graph.
        let mut graph = self.graph.lock().map_err(|_| {
            TalaError::SegmentCorrupted("graph lock poisoned".to_string())
        })?;
        graph.insert_node(id, timestamp, confidence);
        if !candidates.is_empty() {
            graph.form_edges(id, &mut candidates, edge_k);
        }
        let edge_ns = edge_t0.elapsed().as_nanos() as u64;
        self.daemon_metrics
            .edge_formation_ns
            .fetch_add(edge_ns, Relaxed);
        self.daemon_metrics
            .edge_formation_count
            .fetch_add(1, Relaxed);

        Ok(id)
    }
}

// ===========================================================================
// Daemon
// ===========================================================================

/// Top-level orchestrator. Provides a unified interface to all TALA subsystems:
/// ingest, semantic query, replay planning, and insight generation.
pub struct Daemon {
    pipeline: IngestPipeline,
    dim: usize,
    daemon_metrics: Arc<DaemonMetrics>,
}

impl Daemon {
    /// Ingest a raw command string with execution context.
    ///
    /// Extracts the intent, stores it, indexes its embedding, and forms edges
    /// in the narrative graph.
    pub fn ingest(&self, raw: &str, context: &Context) -> Result<IntentId, TalaError> {
        self.pipeline.ingest(raw, context)
    }

    /// Semantic search: find the `k` intents most similar to `embedding`.
    ///
    /// Returns `(IntentId, cosine_similarity)` pairs sorted by similarity descending.
    pub fn query(&self, embedding: &[f32], k: usize) -> Result<Vec<(IntentId, f32)>, TalaError> {
        let t0 = Instant::now();
        let result = self.pipeline.store.query_semantic(embedding, k);
        let ns = t0.elapsed().as_nanos() as u64;
        self.daemon_metrics.query_ns.fetch_add(ns, Relaxed);
        self.daemon_metrics.query_count.fetch_add(1, Relaxed);
        result
    }

    /// Build a replay plan rooted at `root`, traversing up to `depth` hops forward.
    ///
    /// Uses BFS on the narrative graph to discover the reachable subgraph, then
    /// topologically sorts it via `tala_weave::build_plan` to produce an ordered
    /// sequence of `ReplayStep`s.
    pub fn replay(
        &self,
        root: IntentId,
        depth: usize,
    ) -> Result<Vec<ReplayStep>, TalaError> {
        let t0 = Instant::now();

        let graph = self.pipeline.graph.lock().map_err(|_| {
            TalaError::SegmentCorrupted("graph lock poisoned".to_string())
        })?;

        if !graph.contains_node(root) {
            return Err(TalaError::NodeNotFound(root));
        }

        // BFS forward to discover the subgraph.
        let reachable = graph.bfs_forward(root, depth);

        // Build command map from the store.
        let mut commands: HashMap<IntentId, String> = HashMap::new();
        for &id in &reachable {
            if let Some(intent) = self.pipeline.store.get(id)? {
                commands.insert(id, intent.raw_command);
            }
        }

        let result = tala_weave::build_plan(&graph, &reachable, &commands);

        let ns = t0.elapsed().as_nanos() as u64;
        self.daemon_metrics.replay_ns.fetch_add(ns, Relaxed);
        self.daemon_metrics.replay_count.fetch_add(1, Relaxed);

        result
    }

    /// Generate insights from the current intent corpus.
    ///
    /// Runs pattern detection, summarization, and k-means clustering over all
    /// stored intent embeddings with `k_clusters` clusters. Produces a mix of
    /// `RecurringPattern`, `Summary`, and `FailureCluster` insights.
    pub fn insights(&self, k_clusters: usize) -> Result<Vec<Insight>, TalaError> {
        let t0 = Instant::now();

        let graph = self.pipeline.graph.lock().map_err(|_| {
            TalaError::SegmentCorrupted("graph lock poisoned".to_string())
        })?;

        let node_ids = graph.node_ids();
        if node_ids.is_empty() {
            return Ok(Vec::new());
        }

        // Collect all intents from the store.
        let mut intents = Vec::with_capacity(node_ids.len());
        for &id in &node_ids {
            if let Some(intent) = self.pipeline.store.get(id)? {
                intents.push(intent);
            }
        }

        if intents.is_empty() {
            return Ok(Vec::new());
        }

        // Sort by timestamp for pattern detection.
        intents.sort_by_key(|i| i.timestamp);

        let engine = InsightEngine::new();
        let mut insights = Vec::new();

        // 1. Pattern detection.
        insights.extend(engine.detect_patterns(&intents));

        // 2. Summary.
        insights.push(engine.summarize(&intents));

        // 3. Clustering (only if we have enough points).
        let effective_k = k_clusters.min(intents.len());
        if effective_k > 0 {
            let flat_embeddings: Vec<f32> = intents
                .iter()
                .flat_map(|i| i.embedding.iter().copied())
                .collect();

            if let Ok(cluster_result) =
                engine.analyze_clusters(&flat_embeddings, self.dim, effective_k)
            {
                // Group intents by cluster assignment.
                for c in 0..effective_k {
                    let cluster_intent_ids: Vec<IntentId> = cluster_result
                        .assignments
                        .iter()
                        .enumerate()
                        .filter(|&(_, &assignment)| assignment == c)
                        .map(|(i, _)| intents[i].id)
                        .collect();

                    if !cluster_intent_ids.is_empty() {
                        insights.push(Insight {
                            kind: tala_core::InsightKind::FailureCluster,
                            description: format!(
                                "Cluster {} contains {} intents",
                                c,
                                cluster_intent_ids.len()
                            ),
                            intent_ids: cluster_intent_ids,
                            confidence: if cluster_result.converged {
                                0.8
                            } else {
                                0.5
                            },
                        });
                    }
                }
            }
        }

        let ns = t0.elapsed().as_nanos() as u64;
        self.daemon_metrics.insight_ns.fetch_add(ns, Relaxed);
        self.daemon_metrics.insight_count.fetch_add(1, Relaxed);

        Ok(insights)
    }

    /// Access the underlying storage engine.
    pub fn store(&self) -> &StorageEngine {
        &self.pipeline.store
    }

    /// Access the daemon-level metrics.
    pub fn daemon_metrics(&self) -> &Arc<DaemonMetrics> {
        &self.daemon_metrics
    }
}

// ===========================================================================
// DaemonBuilder
// ===========================================================================

/// Builder for configuring and constructing a `Daemon`.
pub struct DaemonBuilder {
    dim: usize,
    hot_capacity: usize,
}

impl DaemonBuilder {
    /// Create a new builder with default settings (dim=384, hot_capacity=10_000).
    pub fn new() -> Self {
        Self {
            dim: 384,
            hot_capacity: 10_000,
        }
    }

    /// Set the embedding dimension.
    pub fn dim(mut self, dim: usize) -> Self {
        self.dim = dim;
        self
    }

    /// Set the hot buffer capacity (number of intents before segment flush).
    pub fn hot_capacity(mut self, hot_capacity: usize) -> Self {
        self.hot_capacity = hot_capacity;
        self
    }

    /// Build an in-memory daemon (no WAL, no persistence).
    pub fn build_in_memory(self) -> Daemon {
        let store = StorageEngine::in_memory(self.dim, self.hot_capacity);
        let daemon_metrics = Arc::new(DaemonMetrics::new());
        Daemon {
            pipeline: IngestPipeline::new(store, Arc::clone(&daemon_metrics)),
            dim: self.dim,
            daemon_metrics,
        }
    }

    /// Build a persistent daemon backed by the given directory.
    ///
    /// Creates WAL and segment files in `dir`.
    pub fn build(self, dir: impl AsRef<Path>) -> Result<Daemon, TalaError> {
        let store = StorageEngine::open(self.dim, dir, self.hot_capacity)?;
        let daemon_metrics = Arc::new(DaemonMetrics::new());
        Ok(Daemon {
            pipeline: IngestPipeline::new(store, Arc::clone(&daemon_metrics)),
            dim: self.dim,
            daemon_metrics,
        })
    }
}

impl Default for DaemonBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use tala_core::Context;

    fn test_context() -> Context {
        Context {
            cwd: "/home/user/project".to_string(),
            env_hash: 42,
            session_id: 1,
            shell: "zsh".to_string(),
            user: "testuser".to_string(),
        }
    }

    fn in_memory_daemon() -> Daemon {
        DaemonBuilder::new()
            .dim(384)
            .hot_capacity(1000)
            .build_in_memory()
    }

    // -----------------------------------------------------------------------
    // Ingest pipeline tests
    // -----------------------------------------------------------------------

    #[test]
    fn ingest_returns_valid_id() {
        let daemon = in_memory_daemon();
        let ctx = test_context();

        let id = daemon.ingest("cargo build --release", &ctx).unwrap();

        // Verify the intent is stored.
        let intent = daemon.store().get(id).unwrap();
        assert!(intent.is_some());
        let intent = intent.unwrap();
        assert_eq!(intent.id, id);
        assert_eq!(intent.raw_command, "cargo build --release");
    }

    #[test]
    fn ingest_empty_command_fails() {
        let daemon = in_memory_daemon();
        let ctx = test_context();

        let result = daemon.ingest("", &ctx);
        assert!(result.is_err());

        let result = daemon.ingest("   ", &ctx);
        assert!(result.is_err());
    }

    #[test]
    fn ingest_multiple_creates_edges() {
        let daemon = in_memory_daemon();
        let ctx = test_context();

        // Ingest several similar commands to trigger edge formation.
        let mut ids = Vec::new();
        for i in 0..10 {
            let id = daemon
                .ingest(&format!("cargo build --profile={i}"), &ctx)
                .unwrap();
            ids.push(id);
        }

        // The graph should have nodes and edges.
        let graph = daemon.pipeline.graph.lock().unwrap();
        assert_eq!(graph.node_count(), 10);
        // After the first insert, subsequent inserts should form edges.
        assert!(graph.edge_count() > 0);
    }

    // -----------------------------------------------------------------------
    // Query tests
    // -----------------------------------------------------------------------

    #[test]
    fn query_after_ingest() {
        let daemon = in_memory_daemon();
        let ctx = test_context();

        // Ingest some commands.
        for i in 0..20 {
            daemon
                .ingest(&format!("test_command_{i}"), &ctx)
                .unwrap();
        }

        // Query with the embedding of a known command.
        let pipeline = IntentPipeline::new();
        let query_embedding = pipeline.embed("test_command_10");

        let results = daemon.query(&query_embedding, 5).unwrap();
        assert_eq!(results.len(), 5);

        // All similarity scores should be valid.
        for &(_, sim) in &results {
            assert!(
                sim >= -1.0 && sim <= 1.0,
                "sim {sim} out of range"
            );
        }
    }

    #[test]
    fn query_empty_store_returns_empty() {
        let daemon = in_memory_daemon();
        let query = vec![0.0f32; 384];
        let results = daemon.query(&query, 5).unwrap();
        assert!(results.is_empty());
    }

    // -----------------------------------------------------------------------
    // Replay tests
    // -----------------------------------------------------------------------

    #[test]
    fn replay_after_ingest() {
        let daemon = in_memory_daemon();
        let ctx = test_context();

        // Ingest a chain of commands.
        let mut ids = Vec::new();
        for cmd in &["mkdir -p src", "cd src", "touch main.rs", "cargo init"] {
            let id = daemon.ingest(cmd, &ctx).unwrap();
            ids.push(id);
        }

        // Replay from the first node.
        let plan = daemon.replay(ids[0], 10).unwrap();
        assert!(!plan.is_empty());

        // The first step must be the root.
        assert_eq!(plan[0].intent_id, ids[0]);
        assert_eq!(plan[0].command, "mkdir -p src");
    }

    #[test]
    fn replay_unknown_root_fails() {
        let daemon = in_memory_daemon();
        let fake_id = IntentId::random();

        let result = daemon.replay(fake_id, 5);
        assert!(result.is_err());
    }

    #[test]
    fn replay_single_node() {
        let daemon = in_memory_daemon();
        let ctx = test_context();

        let id = daemon.ingest("echo hello", &ctx).unwrap();
        let plan = daemon.replay(id, 0).unwrap();

        assert_eq!(plan.len(), 1);
        assert_eq!(plan[0].intent_id, id);
        assert_eq!(plan[0].command, "echo hello");
    }

    // -----------------------------------------------------------------------
    // Insights tests
    // -----------------------------------------------------------------------

    #[test]
    fn insights_after_ingest() {
        let daemon = in_memory_daemon();
        let ctx = test_context();

        // Ingest a mix of commands.
        for _ in 0..3 {
            daemon.ingest("cargo build", &ctx).unwrap();
            daemon.ingest("cargo test", &ctx).unwrap();
            daemon.ingest("cargo build", &ctx).unwrap();
        }

        let insights = daemon.insights(2).unwrap();
        // Should have at least a summary insight.
        assert!(!insights.is_empty());

        // Check that we got a Summary insight.
        let has_summary = insights
            .iter()
            .any(|i| i.kind == tala_core::InsightKind::Summary);
        assert!(has_summary, "expected a Summary insight");
    }

    #[test]
    fn insights_empty_store() {
        let daemon = in_memory_daemon();
        let insights = daemon.insights(3).unwrap();
        assert!(insights.is_empty());
    }

    // -----------------------------------------------------------------------
    // Builder tests
    // -----------------------------------------------------------------------

    #[test]
    fn builder_defaults() {
        let daemon = DaemonBuilder::new().build_in_memory();
        let ctx = test_context();

        // Should work with the default 384-dim embeddings.
        let id = daemon.ingest("ls -la", &ctx).unwrap();
        assert!(daemon.store().get(id).unwrap().is_some());
    }

    #[test]
    fn builder_persistent() {
        let dir = tempfile::tempdir().unwrap();
        let daemon = DaemonBuilder::new()
            .dim(384)
            .hot_capacity(100)
            .build(dir.path())
            .unwrap();

        let ctx = test_context();
        let id = daemon.ingest("git status", &ctx).unwrap();
        assert!(daemon.store().get(id).unwrap().is_some());
    }
}
