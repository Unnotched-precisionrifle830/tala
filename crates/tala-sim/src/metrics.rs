//! Lock-free Prometheus metrics using AtomicU64.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use tala_daemon::DaemonMetrics;
use tala_store::{LockStats, StorageMetrics};

/// Histogram bucket boundaries in microseconds.
const BUCKET_BOUNDS: [u64; 10] = [10, 50, 100, 250, 500, 1_000, 2_500, 5_000, 10_000, 50_000];

/// Lock-free histogram state with fixed buckets.
pub struct HistogramState {
    /// Cumulative bucket counters (bucket[i] counts values <= BUCKET_BOUNDS[i]).
    buckets: [AtomicU64; 10],
    /// Total observed values (count).
    count: AtomicU64,
    /// Sum of all observed values (microseconds).
    sum: AtomicU64,
}

impl HistogramState {
    fn new() -> Self {
        Self {
            buckets: [
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
            ],
            count: AtomicU64::new(0),
            sum: AtomicU64::new(0),
        }
    }

    /// Record an observed value in microseconds.
    pub fn observe(&self, value_us: u64) {
        self.count.fetch_add(1, Ordering::Relaxed);
        self.sum.fetch_add(value_us, Ordering::Relaxed);
        for (i, &bound) in BUCKET_BOUNDS.iter().enumerate() {
            if value_us <= bound {
                self.buckets[i].fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    fn render(&self, name: &str, help: &str, vertical: &str) -> String {
        let mut out = String::new();
        out.push_str(&format!("# HELP {name} {help}\n"));
        out.push_str(&format!("# TYPE {name} histogram\n"));
        for (i, &bound) in BUCKET_BOUNDS.iter().enumerate() {
            let v = self.buckets[i].load(Ordering::Relaxed);
            out.push_str(&format!(
                "{name}_bucket{{vertical=\"{vertical}\",le=\"{bound}\"}} {v}\n"
            ));
        }
        let count = self.count.load(Ordering::Relaxed);
        let sum = self.sum.load(Ordering::Relaxed);
        out.push_str(&format!(
            "{name}_bucket{{vertical=\"{vertical}\",le=\"+Inf\"}} {count}\n"
        ));
        out.push_str(&format!(
            "{name}_sum{{vertical=\"{vertical}\"}} {sum}\n"
        ));
        out.push_str(&format!(
            "{name}_count{{vertical=\"{vertical}\"}} {count}\n"
        ));
        out
    }
}

/// Render a single `LockStats` to Prometheus exposition format with a lock label.
fn render_lock_stats(out: &mut String, lock_name: &str, vertical: &str, stats: &LockStats) {
    let prefix = "tala_lock";

    let acquisitions = stats.acquisitions.load(Ordering::Relaxed);
    let contentions = stats.contentions.load(Ordering::Relaxed);
    let wait_total = stats.total_wait_ns.load(Ordering::Relaxed);
    let hold_total = stats.total_hold_ns.load(Ordering::Relaxed);
    let max_wait = stats.max_wait_ns.load(Ordering::Relaxed);
    let max_hold = stats.max_hold_ns.load(Ordering::Relaxed);

    out.push_str(&format!(
        "{prefix}_acquisitions_total{{vertical=\"{vertical}\",lock=\"{lock_name}\"}} {acquisitions}\n"
    ));
    out.push_str(&format!(
        "{prefix}_contentions_total{{vertical=\"{vertical}\",lock=\"{lock_name}\"}} {contentions}\n"
    ));
    out.push_str(&format!(
        "{prefix}_wait_ns_total{{vertical=\"{vertical}\",lock=\"{lock_name}\"}} {wait_total}\n"
    ));
    out.push_str(&format!(
        "{prefix}_hold_ns_total{{vertical=\"{vertical}\",lock=\"{lock_name}\"}} {hold_total}\n"
    ));
    out.push_str(&format!(
        "{prefix}_max_wait_ns{{vertical=\"{vertical}\",lock=\"{lock_name}\"}} {max_wait}\n"
    ));
    out.push_str(&format!(
        "{prefix}_max_hold_ns{{vertical=\"{vertical}\",lock=\"{lock_name}\"}} {max_hold}\n"
    ));
}

/// Prometheus-style metrics for a single vertical workload.
pub struct Metrics {
    // --- Counters ---
    pub intents_ingested_total: AtomicU64,
    pub intents_success_total: AtomicU64,
    pub intents_failure_total: AtomicU64,
    pub intents_partial_total: AtomicU64,
    pub queries_total: AtomicU64,
    pub replays_total: AtomicU64,
    pub insights_total: AtomicU64,
    pub chaos_events_total: AtomicU64,
    pub chaos_failures_injected: AtomicU64,
    pub chaos_latency_spikes: AtomicU64,
    pub chaos_retries_injected: AtomicU64,

    // --- Gauges ---
    pub graph_node_count: AtomicU64,
    pub graph_edge_count: AtomicU64,
    pub hnsw_index_size: AtomicU64,
    pub active_patterns: AtomicU64,
    pub active_clusters: AtomicU64,
    pub uptime_seconds: AtomicU64,

    // --- Histograms ---
    pub ingest_latency_us: HistogramState,
    pub query_latency_us: HistogramState,

    // --- Pipeline breakdown histograms ---
    pub extract_latency_us: HistogramState,
    pub wal_append_latency_us: HistogramState,
    pub hnsw_insert_latency_us: HistogramState,
    pub edge_search_latency_us: HistogramState,
    pub hot_push_latency_us: HistogramState,
    pub segment_flush_latency_us: HistogramState,

    // --- Store internals gauges ---
    pub hot_buffer_fill_ratio: AtomicU64,
    pub wal_entries_total: AtomicU64,
    pub segments_flushed_total: AtomicU64,
    pub bytes_flushed_total: AtomicU64,

    // --- HNSW internals ---
    pub hnsw_avg_search_visited: AtomicU64,

    // --- External metrics references ---
    pub store_metrics: Option<Arc<StorageMetrics>>,
    pub daemon_metrics: Option<Arc<DaemonMetrics>>,

    // --- Label ---
    pub vertical: String,
}

impl Metrics {
    pub fn new(vertical: String) -> Self {
        Self {
            intents_ingested_total: AtomicU64::new(0),
            intents_success_total: AtomicU64::new(0),
            intents_failure_total: AtomicU64::new(0),
            intents_partial_total: AtomicU64::new(0),
            queries_total: AtomicU64::new(0),
            replays_total: AtomicU64::new(0),
            insights_total: AtomicU64::new(0),
            chaos_events_total: AtomicU64::new(0),
            chaos_failures_injected: AtomicU64::new(0),
            chaos_latency_spikes: AtomicU64::new(0),
            chaos_retries_injected: AtomicU64::new(0),
            graph_node_count: AtomicU64::new(0),
            graph_edge_count: AtomicU64::new(0),
            hnsw_index_size: AtomicU64::new(0),
            active_patterns: AtomicU64::new(0),
            active_clusters: AtomicU64::new(0),
            uptime_seconds: AtomicU64::new(0),
            ingest_latency_us: HistogramState::new(),
            query_latency_us: HistogramState::new(),
            extract_latency_us: HistogramState::new(),
            wal_append_latency_us: HistogramState::new(),
            hnsw_insert_latency_us: HistogramState::new(),
            edge_search_latency_us: HistogramState::new(),
            hot_push_latency_us: HistogramState::new(),
            segment_flush_latency_us: HistogramState::new(),
            hot_buffer_fill_ratio: AtomicU64::new(0),
            wal_entries_total: AtomicU64::new(0),
            segments_flushed_total: AtomicU64::new(0),
            bytes_flushed_total: AtomicU64::new(0),
            hnsw_avg_search_visited: AtomicU64::new(0),
            store_metrics: None,
            daemon_metrics: None,
            vertical,
        }
    }

    /// Render all metrics in Prometheus text exposition format.
    pub fn render(&self) -> String {
        let v = &self.vertical;
        let mut out = String::with_capacity(8192);

        // Counters
        let counters: &[(&str, &str, &AtomicU64)] = &[
            (
                "tala_intents_ingested_total",
                "Total intents ingested.",
                &self.intents_ingested_total,
            ),
            (
                "tala_intents_success_total",
                "Total intents with success outcome.",
                &self.intents_success_total,
            ),
            (
                "tala_intents_failure_total",
                "Total intents with failure outcome.",
                &self.intents_failure_total,
            ),
            (
                "tala_intents_partial_total",
                "Total intents with partial outcome.",
                &self.intents_partial_total,
            ),
            (
                "tala_queries_total",
                "Total semantic queries executed.",
                &self.queries_total,
            ),
            (
                "tala_replays_total",
                "Total replay plans generated.",
                &self.replays_total,
            ),
            (
                "tala_insights_total",
                "Total insight generations.",
                &self.insights_total,
            ),
            (
                "tala_chaos_events_total",
                "Total chaos events triggered.",
                &self.chaos_events_total,
            ),
            (
                "tala_chaos_failures_injected",
                "Total forced failures injected by chaos.",
                &self.chaos_failures_injected,
            ),
            (
                "tala_chaos_latency_spikes",
                "Total latency spikes injected by chaos.",
                &self.chaos_latency_spikes,
            ),
            (
                "tala_chaos_retries_injected",
                "Total retry storms injected by chaos.",
                &self.chaos_retries_injected,
            ),
        ];

        for (name, help, counter) in counters {
            let val = counter.load(Ordering::Relaxed);
            out.push_str(&format!("# HELP {name} {help}\n"));
            out.push_str(&format!("# TYPE {name} counter\n"));
            out.push_str(&format!("{name}{{vertical=\"{v}\"}} {val}\n"));
        }

        // Gauges
        let gauges: &[(&str, &str, &AtomicU64)] = &[
            (
                "tala_graph_node_count",
                "Current number of nodes in the narrative graph.",
                &self.graph_node_count,
            ),
            (
                "tala_graph_edge_count",
                "Current number of edges in the narrative graph.",
                &self.graph_edge_count,
            ),
            (
                "tala_hnsw_index_size",
                "Current number of vectors in the HNSW index.",
                &self.hnsw_index_size,
            ),
            (
                "tala_active_patterns",
                "Number of active detected patterns.",
                &self.active_patterns,
            ),
            (
                "tala_active_clusters",
                "Number of active intent clusters.",
                &self.active_clusters,
            ),
            (
                "tala_uptime_seconds",
                "Simulator uptime in seconds.",
                &self.uptime_seconds,
            ),
            (
                "tala_hot_buffer_fill_ratio",
                "Hot buffer fill ratio (current/capacity, scaled by 1000).",
                &self.hot_buffer_fill_ratio,
            ),
            (
                "tala_wal_entries_total",
                "Total WAL entries written.",
                &self.wal_entries_total,
            ),
            (
                "tala_segments_flushed_total",
                "Total segments flushed to disk.",
                &self.segments_flushed_total,
            ),
            (
                "tala_bytes_flushed_total",
                "Total bytes flushed to segments.",
                &self.bytes_flushed_total,
            ),
            (
                "tala_hnsw_avg_search_visited",
                "Average nodes visited per HNSW search.",
                &self.hnsw_avg_search_visited,
            ),
        ];

        for (name, help, gauge) in gauges {
            let val = gauge.load(Ordering::Relaxed);
            out.push_str(&format!("# HELP {name} {help}\n"));
            out.push_str(&format!("# TYPE {name} gauge\n"));
            out.push_str(&format!("{name}{{vertical=\"{v}\"}} {val}\n"));
        }

        // Histograms
        out.push_str(&self.ingest_latency_us.render(
            "tala_ingest_latency_us",
            "Ingest latency in microseconds.",
            v,
        ));
        out.push_str(&self.query_latency_us.render(
            "tala_query_latency_us",
            "Query latency in microseconds.",
            v,
        ));
        out.push_str(&self.extract_latency_us.render(
            "tala_extract_latency_us",
            "Intent extraction latency in microseconds.",
            v,
        ));
        out.push_str(&self.wal_append_latency_us.render(
            "tala_wal_append_latency_us",
            "WAL append latency in microseconds.",
            v,
        ));
        out.push_str(&self.hnsw_insert_latency_us.render(
            "tala_hnsw_insert_latency_us",
            "HNSW index insert latency in microseconds.",
            v,
        ));
        out.push_str(&self.edge_search_latency_us.render(
            "tala_edge_search_latency_us",
            "Edge candidate search latency in microseconds.",
            v,
        ));
        out.push_str(&self.hot_push_latency_us.render(
            "tala_hot_push_latency_us",
            "Hot buffer push latency in microseconds.",
            v,
        ));
        out.push_str(&self.segment_flush_latency_us.render(
            "tala_segment_flush_latency_us",
            "Segment flush latency in microseconds.",
            v,
        ));

        // Lock contention stats from StorageMetrics
        if let Some(ref sm) = self.store_metrics {
            out.push_str("# HELP tala_lock_acquisitions_total Total lock acquisitions.\n");
            out.push_str("# TYPE tala_lock_acquisitions_total counter\n");
            out.push_str("# HELP tala_lock_contentions_total Total lock contentions (wait > 1us).\n");
            out.push_str("# TYPE tala_lock_contentions_total counter\n");
            out.push_str("# HELP tala_lock_wait_ns_total Cumulative lock wait time in nanoseconds.\n");
            out.push_str("# TYPE tala_lock_wait_ns_total counter\n");
            out.push_str("# HELP tala_lock_hold_ns_total Cumulative lock hold time in nanoseconds.\n");
            out.push_str("# TYPE tala_lock_hold_ns_total counter\n");
            out.push_str("# HELP tala_lock_max_wait_ns Worst-case lock wait in nanoseconds.\n");
            out.push_str("# TYPE tala_lock_max_wait_ns gauge\n");
            out.push_str("# HELP tala_lock_max_hold_ns Worst-case lock hold in nanoseconds.\n");
            out.push_str("# TYPE tala_lock_max_hold_ns gauge\n");

            render_lock_stats(&mut out, "intents", v, &sm.intents_lock);
            render_lock_stats(&mut out, "hnsw", v, &sm.hnsw_lock);
            render_lock_stats(&mut out, "index_map", v, &sm.index_map_lock);
            render_lock_stats(&mut out, "wal", v, &sm.wal_lock);
            render_lock_stats(&mut out, "hot", v, &sm.hot_lock);

            // Pipeline sub-operation counters from StorageMetrics
            let store_counters: &[(&str, &str, &AtomicU64)] = &[
                (
                    "tala_store_wal_append_ns_total",
                    "Cumulative WAL append time in nanoseconds.",
                    &sm.wal_append_ns,
                ),
                (
                    "tala_store_wal_append_count",
                    "Total WAL append operations.",
                    &sm.wal_append_count,
                ),
                (
                    "tala_store_hnsw_insert_ns_total",
                    "Cumulative HNSW insert time in nanoseconds.",
                    &sm.hnsw_insert_ns,
                ),
                (
                    "tala_store_hnsw_insert_count",
                    "Total HNSW insert operations.",
                    &sm.hnsw_insert_count,
                ),
                (
                    "tala_store_hot_push_ns_total",
                    "Cumulative hot buffer push time in nanoseconds.",
                    &sm.hot_push_ns,
                ),
                (
                    "tala_store_hot_push_count",
                    "Total hot buffer push operations.",
                    &sm.hot_push_count,
                ),
                (
                    "tala_store_segment_flush_ns_total",
                    "Cumulative segment flush time in nanoseconds.",
                    &sm.segment_flush_ns,
                ),
                (
                    "tala_store_segment_flush_count",
                    "Total segment flush operations.",
                    &sm.segment_flush_count,
                ),
                (
                    "tala_store_edge_search_ns_total",
                    "Cumulative edge candidate search time in nanoseconds.",
                    &sm.edge_search_ns,
                ),
                (
                    "tala_store_edge_search_count",
                    "Total edge candidate search operations.",
                    &sm.edge_search_count,
                ),
            ];
            for (name, help, counter) in store_counters {
                let val = counter.load(Ordering::Relaxed);
                out.push_str(&format!("# HELP {name} {help}\n"));
                out.push_str(&format!("# TYPE {name} counter\n"));
                out.push_str(&format!("{name}{{vertical=\"{v}\"}} {val}\n"));
            }
        }

        // Daemon-level pipeline phase timing
        if let Some(ref dm) = self.daemon_metrics {
            let daemon_counters: &[(&str, &str, &AtomicU64)] = &[
                (
                    "tala_daemon_extract_ns_total",
                    "Cumulative intent extraction time in nanoseconds.",
                    &dm.extract_ns,
                ),
                (
                    "tala_daemon_extract_count",
                    "Total intent extractions.",
                    &dm.extract_count,
                ),
                (
                    "tala_daemon_store_insert_ns_total",
                    "Cumulative store insert time in nanoseconds.",
                    &dm.store_insert_ns,
                ),
                (
                    "tala_daemon_store_insert_count",
                    "Total store insert operations.",
                    &dm.store_insert_count,
                ),
                (
                    "tala_daemon_edge_formation_ns_total",
                    "Cumulative edge formation time in nanoseconds.",
                    &dm.edge_formation_ns,
                ),
                (
                    "tala_daemon_edge_formation_count",
                    "Total edge formation operations.",
                    &dm.edge_formation_count,
                ),
                (
                    "tala_daemon_query_ns_total",
                    "Cumulative semantic query time in nanoseconds.",
                    &dm.query_ns,
                ),
                (
                    "tala_daemon_query_count",
                    "Total semantic queries via daemon.",
                    &dm.query_count,
                ),
                (
                    "tala_daemon_replay_ns_total",
                    "Cumulative replay time in nanoseconds.",
                    &dm.replay_ns,
                ),
                (
                    "tala_daemon_replay_count",
                    "Total replays via daemon.",
                    &dm.replay_count,
                ),
                (
                    "tala_daemon_insight_ns_total",
                    "Cumulative insight generation time in nanoseconds.",
                    &dm.insight_ns,
                ),
                (
                    "tala_daemon_insight_count",
                    "Total insight generations via daemon.",
                    &dm.insight_count,
                ),
            ];
            for (name, help, counter) in daemon_counters {
                let val = counter.load(Ordering::Relaxed);
                out.push_str(&format!("# HELP {name} {help}\n"));
                out.push_str(&format!("# TYPE {name} counter\n"));
                out.push_str(&format!("{name}{{vertical=\"{v}\"}} {val}\n"));
            }
        }

        out
    }
}
