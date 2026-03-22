//! TALA Kai — Insight engine: pattern detection, clustering, and prediction.
//!
//! Provides analysis capabilities over intent histories:
//! - K-means clustering of intent embeddings
//! - N-gram pattern detection in command sequences
//! - Frequency-based next-intent prediction
//! - Narrative summarization

use std::collections::HashMap;

use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

use tala_core::{Insight, InsightKind, Intent, Status, TalaError};

// ===========================================================================
// K-means clustering
// ===========================================================================

/// Result of k-means clustering.
#[derive(Clone, Debug)]
pub struct ClusterResult {
    /// Cluster assignment for each input point (index into `centroids`).
    pub assignments: Vec<usize>,
    /// Final centroid vectors, shape: k × dim.
    pub centroids: Vec<Vec<f32>>,
    /// Number of iterations until convergence.
    pub iterations: usize,
    /// Whether the algorithm converged before hitting max iterations.
    pub converged: bool,
}

/// Run Lloyd's k-means on a set of embedding vectors.
///
/// # Arguments
/// * `embeddings` - Flat buffer of embeddings, each of length `dim`.
/// * `dim` - Dimensionality of each embedding.
/// * `k` - Number of clusters.
/// * `max_iter` - Maximum iterations before stopping.
/// * `seed` - RNG seed for deterministic centroid initialization.
///
/// # Errors
/// Returns `TalaError::DimensionMismatch` if `embeddings.len()` is not divisible by `dim`,
/// or if `k` is zero or exceeds the number of points.
pub fn kmeans(
    embeddings: &[f32],
    dim: usize,
    k: usize,
    max_iter: usize,
    seed: u64,
) -> Result<ClusterResult, TalaError> {
    if dim == 0 {
        return Err(TalaError::DimensionMismatch {
            expected: 1,
            got: 0,
        });
    }
    if embeddings.len() % dim != 0 {
        return Err(TalaError::DimensionMismatch {
            expected: (embeddings.len() / dim) * dim,
            got: embeddings.len(),
        });
    }

    let n = embeddings.len() / dim;

    if k == 0 || k > n {
        return Err(TalaError::DimensionMismatch {
            expected: k.min(n).max(1),
            got: k,
        });
    }

    // Initialize centroids by sampling k distinct data points.
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut indices: Vec<usize> = (0..n).collect();
    indices.shuffle(&mut rng);

    let mut centroids: Vec<Vec<f32>> = indices[..k]
        .iter()
        .map(|&i| embeddings[i * dim..(i + 1) * dim].to_vec())
        .collect();

    let mut assignments = vec![0usize; n];
    let mut converged = false;

    let mut iter = 0;
    while iter < max_iter {
        // Assignment step: assign each point to nearest centroid.
        let mut changed = false;
        for i in 0..n {
            let point = &embeddings[i * dim..(i + 1) * dim];
            let mut best_cluster = 0;
            let mut best_dist = f32::MAX;
            for (c, centroid) in centroids.iter().enumerate() {
                let dist = tala_embed::l2_distance_sq(point, centroid);
                if dist < best_dist {
                    best_dist = dist;
                    best_cluster = c;
                }
            }
            if assignments[i] != best_cluster {
                assignments[i] = best_cluster;
                changed = true;
            }
        }

        if !changed {
            converged = true;
            break;
        }

        // Update step: recompute centroids as mean of assigned points.
        let mut sums = vec![vec![0.0f32; dim]; k];
        let mut counts = vec![0usize; k];

        for i in 0..n {
            let c = assignments[i];
            counts[c] += 1;
            let point = &embeddings[i * dim..(i + 1) * dim];
            for d in 0..dim {
                sums[c][d] += point[d];
            }
        }

        for c in 0..k {
            if counts[c] > 0 {
                let count_f = counts[c] as f32;
                for d in 0..dim {
                    centroids[c][d] = sums[c][d] / count_f;
                }
            }
            // Empty clusters keep their previous centroid.
        }

        iter += 1;
    }

    Ok(ClusterResult {
        assignments,
        centroids,
        iterations: iter,
        converged,
    })
}

// ===========================================================================
// Pattern detection (n-gram frequency)
// ===========================================================================

/// A detected pattern: a command sequence that recurs above threshold.
#[derive(Clone, Debug)]
pub struct Pattern {
    /// The command sequence (bigram or trigram).
    pub commands: Vec<String>,
    /// Number of occurrences in the corpus.
    pub count: usize,
}

/// Detect recurring command sequences (bigrams and trigrams) from a sorted list of intents.
///
/// Intents MUST be pre-sorted by timestamp. Returns patterns whose frequency meets
/// or exceeds `min_count`.
pub fn detect_patterns(intents: &[Intent], min_count: usize) -> Vec<Pattern> {
    if intents.len() < 2 {
        return Vec::new();
    }

    let commands: Vec<&str> = intents.iter().map(|i| i.raw_command.as_str()).collect();
    let mut ngram_counts: HashMap<Vec<&str>, usize> = HashMap::new();

    // Bigrams
    for window in commands.windows(2) {
        let key = window.to_vec();
        *ngram_counts.entry(key).or_insert(0) += 1;
    }

    // Trigrams
    if commands.len() >= 3 {
        for window in commands.windows(3) {
            let key = window.to_vec();
            *ngram_counts.entry(key).or_insert(0) += 1;
        }
    }

    let mut patterns: Vec<Pattern> = ngram_counts
        .into_iter()
        .filter(|(_, count)| *count >= min_count)
        .map(|(cmds, count)| Pattern {
            commands: cmds.into_iter().map(String::from).collect(),
            count,
        })
        .collect();

    // Sort by count descending, then by sequence length descending for stability.
    patterns.sort_unstable_by(|a, b| {
        b.count
            .cmp(&a.count)
            .then_with(|| b.commands.len().cmp(&a.commands.len()))
    });

    patterns
}

// ===========================================================================
// Next-intent prediction
// ===========================================================================

/// Predict the most likely next command given recent history.
///
/// Uses a simple frequency model: looks at the last 1, 2, and 3 commands in `history`
/// and finds the most common successor in `corpus`. Longer context matches are preferred
/// (trigram > bigram > unigram).
///
/// Returns `None` if no match is found.
pub fn predict_next(history: &[String], corpus: &[Intent]) -> Option<String> {
    if history.is_empty() || corpus.len() < 2 {
        return None;
    }

    let corpus_cmds: Vec<&str> = corpus.iter().map(|i| i.raw_command.as_str()).collect();

    // Try trigram context first, then bigram, then unigram.
    for context_len in (1..=3.min(history.len())).rev() {
        let context: Vec<&str> = history[history.len() - context_len..]
            .iter()
            .map(|s| s.as_str())
            .collect();

        let mut successor_counts: HashMap<&str, usize> = HashMap::new();
        let window_size = context_len + 1;

        if corpus_cmds.len() < window_size {
            continue;
        }

        for window in corpus_cmds.windows(window_size) {
            let prefix = &window[..context_len];
            if prefix == context.as_slice() {
                let successor = window[context_len];
                *successor_counts.entry(successor).or_insert(0) += 1;
            }
        }

        if let Some((&best_cmd, _)) = successor_counts.iter().max_by_key(|(_, &count)| count) {
            return Some(best_cmd.to_string());
        }
    }

    None
}

// ===========================================================================
// Narrative summarization
// ===========================================================================

/// Summary statistics for a set of intents.
#[derive(Clone, Debug)]
pub struct NarrativeSummary {
    /// Total number of intents.
    pub total: usize,
    /// Number with Success outcome.
    pub successes: usize,
    /// Number with Failure outcome.
    pub failures: usize,
    /// Number with no outcome attached.
    pub pending: usize,
    /// Earliest timestamp (nanosecond epoch).
    pub time_start: u64,
    /// Latest timestamp (nanosecond epoch).
    pub time_end: u64,
    /// Most common commands, sorted by frequency descending.
    pub top_commands: Vec<(String, usize)>,
    /// Human-readable summary text.
    pub text: String,
}

/// Produce a summary of a set of intents.
pub fn summarize(intents: &[Intent]) -> NarrativeSummary {
    if intents.is_empty() {
        return NarrativeSummary {
            total: 0,
            successes: 0,
            failures: 0,
            pending: 0,
            time_start: 0,
            time_end: 0,
            top_commands: Vec::new(),
            text: String::from("No intents to summarize."),
        };
    }

    let mut successes = 0usize;
    let mut failures = 0usize;
    let mut pending = 0usize;
    let mut time_start = u64::MAX;
    let mut time_end = 0u64;
    let mut cmd_counts: HashMap<&str, usize> = HashMap::new();

    for intent in intents {
        match &intent.outcome {
            Some(o) => match o.status {
                Status::Success => successes += 1,
                Status::Failure => failures += 1,
                Status::Partial => successes += 1, // count partial as success for rate
                Status::Pending => pending += 1,
            },
            None => pending += 1,
        }

        if intent.timestamp < time_start {
            time_start = intent.timestamp;
        }
        if intent.timestamp > time_end {
            time_end = intent.timestamp;
        }

        *cmd_counts.entry(&intent.raw_command).or_insert(0) += 1;
    }

    let mut top_commands: Vec<(String, usize)> = cmd_counts
        .into_iter()
        .map(|(cmd, count)| (cmd.to_string(), count))
        .collect();
    top_commands.sort_unstable_by(|a, b| b.1.cmp(&a.1));
    top_commands.truncate(10);

    let total = intents.len();
    let success_rate = if total > 0 {
        (successes as f64 / total as f64) * 100.0
    } else {
        0.0
    };

    let duration_ns = time_end.saturating_sub(time_start);
    let duration_s = duration_ns as f64 / 1_000_000_000.0;

    let top_cmd_str = top_commands
        .iter()
        .take(5)
        .map(|(cmd, count)| format!("  {} ({}x)", cmd, count))
        .collect::<Vec<_>>()
        .join("\n");

    let text = format!(
        "{} intents over {:.1}s. Success rate: {:.1}% ({} ok, {} failed, {} pending).\nTop commands:\n{}",
        total, duration_s, success_rate, successes, failures, pending, top_cmd_str
    );

    NarrativeSummary {
        total,
        successes,
        failures,
        pending,
        time_start,
        time_end,
        top_commands,
        text,
    }
}

// ===========================================================================
// InsightEngine — orchestrator
// ===========================================================================

/// Orchestrates all analysis capabilities.
pub struct InsightEngine {
    /// Minimum n-gram count threshold for pattern detection.
    pub pattern_threshold: usize,
    /// K-means seed for deterministic clustering.
    pub seed: u64,
    /// Maximum k-means iterations.
    pub max_kmeans_iter: usize,
}

impl Default for InsightEngine {
    fn default() -> Self {
        Self {
            pattern_threshold: 2,
            seed: 42,
            max_kmeans_iter: 100,
        }
    }
}

impl InsightEngine {
    pub fn new() -> Self {
        Self::default()
    }

    /// Cluster intent embeddings into `k` groups.
    pub fn analyze_clusters(
        &self,
        embeddings: &[f32],
        dim: usize,
        k: usize,
    ) -> Result<ClusterResult, TalaError> {
        kmeans(embeddings, dim, k, self.max_kmeans_iter, self.seed)
    }

    /// Detect recurring patterns in a sorted list of intents.
    /// Returns each pattern as an `Insight` with `InsightKind::RecurringPattern`.
    pub fn detect_patterns(&self, intents: &[Intent]) -> Vec<Insight> {
        let patterns = detect_patterns(intents, self.pattern_threshold);
        patterns
            .into_iter()
            .map(|p| {
                let description = format!(
                    "Recurring sequence ({}x): {}",
                    p.count,
                    p.commands.join(" -> ")
                );
                Insight {
                    kind: InsightKind::RecurringPattern,
                    description,
                    intent_ids: Vec::new(),
                    confidence: (p.count as f32 / intents.len().max(1) as f32).min(1.0),
                }
            })
            .collect()
    }

    /// Predict the next command given recent history and a corpus of past intents.
    /// Returns an `Insight` with `InsightKind::Prediction` if a prediction is found.
    pub fn predict_next(&self, history: &[String], corpus: &[Intent]) -> Option<Insight> {
        predict_next(history, corpus).map(|cmd| Insight {
            kind: InsightKind::Prediction,
            description: format!("Predicted next command: {}", cmd),
            intent_ids: Vec::new(),
            confidence: 0.5, // simple frequency model, moderate confidence
        })
    }

    /// Summarize a set of intents into an `Insight` with `InsightKind::Summary`.
    pub fn summarize(&self, intents: &[Intent]) -> Insight {
        let summary = summarize(intents);
        Insight {
            kind: InsightKind::Summary,
            description: summary.text,
            intent_ids: intents.iter().map(|i| i.id).collect(),
            confidence: 1.0,
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use tala_core::{IntentId, Outcome, Status};

    /// Helper: build a minimal Intent with the given command and timestamp.
    fn make_intent(cmd: &str, ts: u64, status: Status) -> Intent {
        Intent {
            id: IntentId::random(),
            timestamp: ts,
            raw_command: cmd.to_string(),
            embedding: vec![0.0; 4],
            context_hash: 0,
            parent_ids: Vec::new(),
            outcome: Some(Outcome {
                status,
                latency_ns: 1000,
                exit_code: if status == Status::Success { 0 } else { 1 },
            }),
            confidence: 1.0,
        }
    }

    // -----------------------------------------------------------------------
    // K-means tests
    // -----------------------------------------------------------------------

    #[test]
    fn kmeans_two_clusters() {
        // Two well-separated clusters in 2D:
        //   Cluster A near (0, 0): 5 points with slight variation
        //   Cluster B near (10, 10): 5 points with slight variation
        let cluster_a: Vec<[f32; 2]> = vec![
            [0.0, 0.0],
            [0.1, 0.2],
            [0.2, 0.1],
            [0.15, 0.05],
            [0.05, 0.15],
        ];
        let cluster_b: Vec<[f32; 2]> = vec![
            [10.0, 10.0],
            [9.8, 10.1],
            [10.2, 9.9],
            [9.9, 10.2],
            [10.1, 9.8],
        ];
        let mut embeddings = Vec::new();
        for p in &cluster_a {
            embeddings.extend_from_slice(p);
        }
        for p in &cluster_b {
            embeddings.extend_from_slice(p);
        }

        let result = kmeans(&embeddings, 2, 2, 50, 7).unwrap();
        assert_eq!(result.assignments.len(), 10);

        // All points in the first half should share one cluster,
        // all in the second half should share another.
        let first_cluster = result.assignments[0];
        let second_cluster = result.assignments[5];
        assert_ne!(first_cluster, second_cluster);

        for i in 0..5 {
            assert_eq!(result.assignments[i], first_cluster);
        }
        for i in 5..10 {
            assert_eq!(result.assignments[i], second_cluster);
        }
        assert!(result.converged);
    }

    #[test]
    fn kmeans_single_cluster() {
        let embeddings = vec![1.0, 2.0, 3.0, 4.0, 1.1, 2.1, 3.1, 4.1];
        let result = kmeans(&embeddings, 4, 1, 50, 42).unwrap();
        assert_eq!(result.assignments.len(), 2);
        assert_eq!(result.assignments[0], 0);
        assert_eq!(result.assignments[1], 0);
        assert!(result.converged);
    }

    #[test]
    fn kmeans_dim_mismatch() {
        // 5 floats with dim=2 is not evenly divisible
        let embeddings = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = kmeans(&embeddings, 2, 1, 50, 42);
        assert!(result.is_err());
    }

    #[test]
    fn kmeans_k_too_large() {
        let embeddings = vec![1.0, 2.0, 3.0, 4.0]; // 2 points of dim 2
        let result = kmeans(&embeddings, 2, 5, 50, 42);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // Pattern detection tests
    // -----------------------------------------------------------------------

    #[test]
    fn detect_bigrams_and_trigrams() {
        let intents = vec![
            make_intent("git status", 1, Status::Success),
            make_intent("git add .", 2, Status::Success),
            make_intent("git commit", 3, Status::Success),
            make_intent("git status", 4, Status::Success),
            make_intent("git add .", 5, Status::Success),
            make_intent("git commit", 6, Status::Success),
            make_intent("git status", 7, Status::Success),
            make_intent("git add .", 8, Status::Success),
            make_intent("git commit", 9, Status::Success),
        ];

        let patterns = detect_patterns(&intents, 2);
        assert!(!patterns.is_empty());

        // The trigram "git status -> git add . -> git commit" should appear 3 times.
        let trigram = patterns
            .iter()
            .find(|p| p.commands.len() == 3 && p.commands[0] == "git status");
        assert!(trigram.is_some());
        assert_eq!(trigram.unwrap().count, 3);

        // The bigram "git status -> git add ." should appear 3 times.
        let bigram = patterns
            .iter()
            .find(|p| p.commands.len() == 2 && p.commands[0] == "git status");
        assert!(bigram.is_some());
        assert!(bigram.unwrap().count >= 3);
    }

    #[test]
    fn detect_patterns_empty() {
        let patterns = detect_patterns(&[], 2);
        assert!(patterns.is_empty());
    }

    #[test]
    fn detect_patterns_single_intent() {
        let intents = vec![make_intent("ls", 1, Status::Success)];
        let patterns = detect_patterns(&intents, 1);
        assert!(patterns.is_empty());
    }

    // -----------------------------------------------------------------------
    // Prediction tests
    // -----------------------------------------------------------------------

    #[test]
    fn predict_next_simple() {
        let corpus = vec![
            make_intent("cd src", 1, Status::Success),
            make_intent("cargo build", 2, Status::Success),
            make_intent("cargo test", 3, Status::Success),
            make_intent("cd src", 4, Status::Success),
            make_intent("cargo build", 5, Status::Success),
            make_intent("cargo test", 6, Status::Success),
            make_intent("cd src", 7, Status::Success),
            make_intent("cargo build", 8, Status::Success),
        ];

        let history = vec!["cd src".to_string(), "cargo build".to_string()];
        let prediction = predict_next(&history, &corpus);
        assert_eq!(prediction, Some("cargo test".to_string()));
    }

    #[test]
    fn predict_next_no_match() {
        let corpus = vec![
            make_intent("ls", 1, Status::Success),
            make_intent("pwd", 2, Status::Success),
        ];

        let history = vec!["git push".to_string()];
        let prediction = predict_next(&history, &corpus);
        assert_eq!(prediction, None);
    }

    #[test]
    fn predict_next_empty_history() {
        let corpus = vec![make_intent("ls", 1, Status::Success)];
        let prediction = predict_next(&[], &corpus);
        assert_eq!(prediction, None);
    }

    // -----------------------------------------------------------------------
    // Summarization tests
    // -----------------------------------------------------------------------

    #[test]
    fn summarize_basic() {
        let intents = vec![
            make_intent("cargo build", 1_000_000_000, Status::Success),
            make_intent("cargo test", 2_000_000_000, Status::Success),
            make_intent("cargo build", 3_000_000_000, Status::Failure),
            make_intent("cargo build", 4_000_000_000, Status::Success),
        ];

        let summary = summarize(&intents);
        assert_eq!(summary.total, 4);
        assert_eq!(summary.successes, 3);
        assert_eq!(summary.failures, 1);
        assert_eq!(summary.pending, 0);
        assert_eq!(summary.time_start, 1_000_000_000);
        assert_eq!(summary.time_end, 4_000_000_000);

        // "cargo build" should be the most common command.
        assert_eq!(summary.top_commands[0].0, "cargo build");
        assert_eq!(summary.top_commands[0].1, 3);

        assert!(summary.text.contains("4 intents"));
        assert!(summary.text.contains("75.0%"));
    }

    #[test]
    fn summarize_empty() {
        let summary = summarize(&[]);
        assert_eq!(summary.total, 0);
        assert_eq!(summary.text, "No intents to summarize.");
    }

    #[test]
    fn summarize_pending() {
        let mut intent = make_intent("ls", 100, Status::Success);
        intent.outcome = None;
        let summary = summarize(&[intent]);
        assert_eq!(summary.pending, 1);
        assert_eq!(summary.successes, 0);
    }

    // -----------------------------------------------------------------------
    // InsightEngine orchestration tests
    // -----------------------------------------------------------------------

    #[test]
    fn engine_detect_patterns_returns_insights() {
        let intents = vec![
            make_intent("make", 1, Status::Success),
            make_intent("test", 2, Status::Success),
            make_intent("make", 3, Status::Success),
            make_intent("test", 4, Status::Success),
            make_intent("make", 5, Status::Success),
            make_intent("test", 6, Status::Success),
        ];

        let engine = InsightEngine::new();
        let insights = engine.detect_patterns(&intents);
        assert!(!insights.is_empty());
        assert!(insights
            .iter()
            .all(|i| i.kind == InsightKind::RecurringPattern));
    }

    #[test]
    fn engine_predict_returns_insight() {
        let corpus = vec![
            make_intent("a", 1, Status::Success),
            make_intent("b", 2, Status::Success),
            make_intent("a", 3, Status::Success),
            make_intent("b", 4, Status::Success),
        ];

        let engine = InsightEngine::new();
        let insight = engine.predict_next(&["a".to_string()], &corpus);
        assert!(insight.is_some());
        assert_eq!(insight.unwrap().kind, InsightKind::Prediction);
    }

    #[test]
    fn engine_summarize_returns_insight() {
        let intents = vec![
            make_intent("ls", 100, Status::Success),
            make_intent("pwd", 200, Status::Failure),
        ];

        let engine = InsightEngine::new();
        let insight = engine.summarize(&intents);
        assert_eq!(insight.kind, InsightKind::Summary);
        assert_eq!(insight.intent_ids.len(), 2);
        assert_eq!(insight.confidence, 1.0);
    }

    #[test]
    fn engine_cluster_analysis() {
        // 4 points in 2D, two clusters
        let embeddings = vec![
            0.0, 0.0, // cluster A
            0.1, 0.1, // cluster A
            5.0, 5.0, // cluster B
            5.1, 5.1, // cluster B
        ];

        let engine = InsightEngine::new();
        let result = engine.analyze_clusters(&embeddings, 2, 2).unwrap();
        assert_eq!(result.assignments.len(), 4);
        assert_eq!(result.assignments[0], result.assignments[1]);
        assert_eq!(result.assignments[2], result.assignments[3]);
        assert_ne!(result.assignments[0], result.assignments[2]);
    }
}
