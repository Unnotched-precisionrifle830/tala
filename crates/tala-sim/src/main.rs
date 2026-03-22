//! TALA Workload Simulator — drives the TALA daemon with realistic vertical workloads.

mod chaos;
mod metrics;
mod verticals;
mod workload;

use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use tala_core::{IntentId, IntentStore, Outcome, Status};
use tala_daemon::DaemonBuilder;
use tala_intent::IntentPipeline;

use crate::chaos::{ChaosEngine, ChaosEvent};
use crate::metrics::Metrics;
use crate::workload::WorkloadGenerator;

fn env_or(name: &str, default: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| default.to_string())
}

fn env_parse<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn main() {
    // -----------------------------------------------------------------------
    // Configuration from environment
    // -----------------------------------------------------------------------
    let vertical = env_or("TALA_VERTICAL", "medical");
    let rate_ms: u64 = env_parse("TALA_RATE_MS", 500);
    let chaos_interval_s: u64 = env_parse("TALA_CHAOS_INTERVAL_S", 60);
    let chaos_probability: f64 = env_parse("TALA_CHAOS_PROBABILITY", 0.15);
    let data_dir = env_or("TALA_DATA_DIR", "/data");
    let metrics_port: u16 = env_parse("TALA_METRICS_PORT", 9090);
    let dim: usize = env_parse("TALA_DIM", 384);
    let hot_capacity: usize = env_parse("TALA_HOT_CAPACITY", 10000);
    let query_interval_s: u64 = env_parse("TALA_QUERY_INTERVAL_S", 10);
    let insight_interval_s: u64 = env_parse("TALA_INSIGHT_INTERVAL_S", 30);
    let replay_interval_s: u64 = env_parse("TALA_REPLAY_INTERVAL_S", 45);

    // -----------------------------------------------------------------------
    // Build daemon
    // -----------------------------------------------------------------------
    let daemon = if data_dir.is_empty() {
        DaemonBuilder::new()
            .dim(dim)
            .hot_capacity(hot_capacity)
            .build_in_memory()
    } else {
        match DaemonBuilder::new()
            .dim(dim)
            .hot_capacity(hot_capacity)
            .build(&data_dir)
        {
            Ok(d) => d,
            Err(e) => {
                eprintln!(
                    "[tala-sim] failed to create persistent daemon at '{}': {}, falling back to in-memory",
                    data_dir, e
                );
                DaemonBuilder::new()
                    .dim(dim)
                    .hot_capacity(hot_capacity)
                    .build_in_memory()
            }
        }
    };
    let daemon = Arc::new(daemon);

    // -----------------------------------------------------------------------
    // Shared state
    // -----------------------------------------------------------------------
    let mut metrics_inner = Metrics::new(vertical.clone());
    // Wire in store and daemon metrics for deep instrumentation rendering.
    metrics_inner.store_metrics = Some(Arc::clone(daemon.store().metrics()));
    metrics_inner.daemon_metrics = Some(Arc::clone(daemon.daemon_metrics()));
    let metrics = Arc::new(metrics_inner);
    let generator = workload::create_generator(&vertical);
    // Shared generator behind Arc (WorkloadGenerator is Send + Sync)
    let generator: Arc<dyn WorkloadGenerator> = Arc::from(generator);
    let pipeline = Arc::new(IntentPipeline::new());
    let recent_ids: Arc<Mutex<Vec<IntentId>>> = Arc::new(Mutex::new(Vec::new()));
    let start_time = Instant::now();

    // -----------------------------------------------------------------------
    // Startup banner
    // -----------------------------------------------------------------------
    eprintln!("================================================================");
    eprintln!("  TALA Workload Simulator");
    eprintln!("  vertical:    {}", vertical);
    eprintln!("  rate:        {}ms", rate_ms);
    eprintln!("  metrics:     http://0.0.0.0:{}/metrics", metrics_port);
    eprintln!("  chaos:       every {}s @ {:.0}%", chaos_interval_s, chaos_probability * 100.0);
    eprintln!("  data_dir:    {}", if data_dir.is_empty() { "(in-memory)" } else { &data_dir });
    eprintln!("  dim:         {}", dim);
    eprintln!("  hot_cap:     {}", hot_capacity);
    eprintln!("================================================================");

    // -----------------------------------------------------------------------
    // Thread: metrics server
    // -----------------------------------------------------------------------
    let metrics_server = {
        let metrics = Arc::clone(&metrics);
        thread::Builder::new()
            .name("metrics-server".into())
            .spawn(move || {
                let addr = format!("0.0.0.0:{}", metrics_port);
                let server = match tiny_http::Server::http(&addr) {
                    Ok(s) => s,
                    Err(e) => {
                        eprintln!("[tala-sim] failed to bind metrics server on {}: {}", addr, e);
                        return;
                    }
                };
                eprintln!("[tala-sim] metrics server listening on {}", addr);
                for request in server.incoming_requests() {
                    let url = request.url().to_string();
                    let response = if url == "/metrics" {
                        let body = metrics.render();
                        tiny_http::Response::from_string(body)
                            .with_header(
                                "Content-Type: text/plain; version=0.0.4; charset=utf-8"
                                    .parse::<tiny_http::Header>()
                                    .expect("static header"),
                            )
                    } else if url == "/health" {
                        tiny_http::Response::from_string("{\"status\":\"ok\"}")
                            .with_header(
                                "Content-Type: application/json"
                                    .parse::<tiny_http::Header>()
                                    .expect("static header"),
                            )
                    } else {
                        tiny_http::Response::from_string("not found")
                            .with_status_code(404)
                    };
                    if let Err(e) = request.respond(response) {
                        eprintln!("[tala-sim] metrics response error: {}", e);
                    }
                }
            })
            .expect("spawn metrics-server thread")
    };

    // -----------------------------------------------------------------------
    // Thread: ingest
    // -----------------------------------------------------------------------
    let ingest_handle = {
        let daemon = Arc::clone(&daemon);
        let metrics = Arc::clone(&metrics);
        let generator = Arc::clone(&generator);
        let recent_ids = Arc::clone(&recent_ids);
        thread::Builder::new()
            .name("ingest".into())
            .spawn(move || {
                let mut rng = SmallRng::from_entropy();
                let mut count: u64 = 0;
                loop {
                    thread::sleep(Duration::from_millis(rate_ms));

                    let (cmd, ctx) = generator.next_command(&mut rng);
                    let t0 = Instant::now();
                    match daemon.ingest(&cmd, &ctx) {
                        Ok(id) => {
                            let elapsed_us = t0.elapsed().as_micros() as u64;
                            metrics.intents_ingested_total.fetch_add(1, Ordering::Relaxed);
                            metrics.ingest_latency_us.observe(elapsed_us);

                            // Simulate outcome and attach
                            let outcome = generator.simulate_outcome(&mut rng);
                            match outcome.status {
                                Status::Success => {
                                    metrics.intents_success_total.fetch_add(1, Ordering::Relaxed);
                                }
                                Status::Failure => {
                                    metrics.intents_failure_total.fetch_add(1, Ordering::Relaxed);
                                }
                                Status::Partial => {
                                    metrics.intents_partial_total.fetch_add(1, Ordering::Relaxed);
                                }
                                Status::Pending => {}
                            }
                            if let Err(e) = daemon.store().attach_outcome(id, outcome) {
                                eprintln!("[tala-sim] attach_outcome error: {}", e);
                            }

                            // Track for replay thread
                            if let Ok(mut ids) = recent_ids.lock() {
                                ids.push(id);
                                // Keep bounded
                                if ids.len() > 1000 {
                                    ids.drain(0..500);
                                }
                            }
                        }
                        Err(e) => {
                            eprintln!("[tala-sim] ingest error: {}", e);
                        }
                    }

                    count += 1;
                    if count % 100 == 0 {
                        let ingested = metrics.intents_ingested_total.load(Ordering::Relaxed);
                        let queries = metrics.queries_total.load(Ordering::Relaxed);
                        let chaos = metrics.chaos_events_total.load(Ordering::Relaxed);
                        eprintln!(
                            "[{}] ingested: {}, queries: {}, chaos: {}",
                            generator.vertical_name(),
                            ingested,
                            queries,
                            chaos,
                        );
                    }
                }
            })
            .expect("spawn ingest thread")
    };

    // -----------------------------------------------------------------------
    // Thread: query
    // -----------------------------------------------------------------------
    let query_handle = {
        let daemon = Arc::clone(&daemon);
        let metrics = Arc::clone(&metrics);
        let generator = Arc::clone(&generator);
        let pipeline = Arc::clone(&pipeline);
        thread::Builder::new()
            .name("query".into())
            .spawn(move || {
                let mut rng = SmallRng::from_entropy();
                let probes = generator.probe_commands();
                loop {
                    thread::sleep(Duration::from_secs(query_interval_s));

                    if probes.is_empty() {
                        continue;
                    }
                    let idx = rng.gen_range(0..probes.len());
                    let embedding = pipeline.embed(probes[idx]);

                    let t0 = Instant::now();
                    match daemon.query(&embedding, 10) {
                        Ok(_results) => {
                            let elapsed_us = t0.elapsed().as_micros() as u64;
                            metrics.queries_total.fetch_add(1, Ordering::Relaxed);
                            metrics.query_latency_us.observe(elapsed_us);
                        }
                        Err(e) => {
                            eprintln!("[tala-sim] query error: {}", e);
                        }
                    }
                }
            })
            .expect("spawn query thread")
    };

    // -----------------------------------------------------------------------
    // Thread: insight
    // -----------------------------------------------------------------------
    let insight_handle = {
        let daemon = Arc::clone(&daemon);
        let metrics = Arc::clone(&metrics);
        thread::Builder::new()
            .name("insight".into())
            .spawn(move || {
                loop {
                    thread::sleep(Duration::from_secs(insight_interval_s));

                    let t0 = Instant::now();
                    match daemon.insights(3) {
                        Ok(insights) => {
                            let _elapsed_us = t0.elapsed().as_micros() as u64;
                            metrics.insights_total.fetch_add(1, Ordering::Relaxed);
                            metrics
                                .active_patterns
                                .store(insights.len() as u64, Ordering::Relaxed);
                        }
                        Err(e) => {
                            eprintln!("[tala-sim] insight error: {}", e);
                        }
                    }
                }
            })
            .expect("spawn insight thread")
    };

    // -----------------------------------------------------------------------
    // Thread: replay
    // -----------------------------------------------------------------------
    let replay_handle = {
        let daemon = Arc::clone(&daemon);
        let metrics = Arc::clone(&metrics);
        let recent_ids = Arc::clone(&recent_ids);
        thread::Builder::new()
            .name("replay".into())
            .spawn(move || {
                let mut rng = SmallRng::from_entropy();
                loop {
                    thread::sleep(Duration::from_secs(replay_interval_s));

                    let maybe_id = {
                        if let Ok(ids) = recent_ids.lock() {
                            if ids.is_empty() {
                                None
                            } else {
                                let idx = rng.gen_range(0..ids.len());
                                Some(ids[idx])
                            }
                        } else {
                            None
                        }
                    };

                    if let Some(root_id) = maybe_id {
                        match daemon.replay(root_id, 3) {
                            Ok(_plan) => {
                                metrics.replays_total.fetch_add(1, Ordering::Relaxed);
                            }
                            Err(e) => {
                                eprintln!("[tala-sim] replay error: {}", e);
                            }
                        }
                    }
                }
            })
            .expect("spawn replay thread")
    };

    // -----------------------------------------------------------------------
    // Thread: chaos
    // -----------------------------------------------------------------------
    let chaos_handle = {
        let daemon = Arc::clone(&daemon);
        let metrics = Arc::clone(&metrics);
        let generator = Arc::clone(&generator);
        thread::Builder::new()
            .name("chaos".into())
            .spawn(move || {
                let engine = ChaosEngine::new(chaos_interval_s, chaos_probability);
                let mut rng = SmallRng::from_entropy();
                loop {
                    thread::sleep(Duration::from_secs(engine.interval_s));

                    if let Some(event) = engine.maybe_trigger(&mut rng) {
                        eprintln!("[tala-sim] chaos: {}", event.name());
                        metrics.chaos_events_total.fetch_add(1, Ordering::Relaxed);

                        match event {
                            ChaosEvent::ForcedFailure => {
                                metrics.chaos_failures_injected.fetch_add(1, Ordering::Relaxed);
                                // Ingest a command and force-attach a failure outcome
                                let (cmd, ctx) = generator.next_command(&mut rng);
                                if let Ok(id) = daemon.ingest(&cmd, &ctx) {
                                    let outcome = Outcome {
                                        status: Status::Failure,
                                        latency_ns: 999_999_999,
                                        exit_code: 137,
                                    };
                                    if let Err(e) = daemon.store().attach_outcome(id, outcome) {
                                        eprintln!("[tala-sim] chaos attach_outcome error: {}", e);
                                    }
                                }
                            }
                            ChaosEvent::LatencySpike { multiplier } => {
                                metrics.chaos_latency_spikes.fetch_add(1, Ordering::Relaxed);
                                // Simulate the spike by sleeping
                                let spike_ms = rate_ms.saturating_mul(multiplier as u64);
                                thread::sleep(Duration::from_millis(spike_ms));
                            }
                            ChaosEvent::RetryStorm { count } => {
                                metrics.chaos_retries_injected.fetch_add(1, Ordering::Relaxed);
                                let (cmd, ctx) = generator.next_command(&mut rng);
                                for _ in 0..count {
                                    if let Err(e) = daemon.ingest(&cmd, &ctx) {
                                        eprintln!("[tala-sim] chaos retry ingest error: {}", e);
                                    }
                                }
                            }
                            ChaosEvent::IngestFailure => {
                                // Try to ingest an empty command — should fail
                                let ctx = tala_core::Context {
                                    cwd: "/chaos".to_string(),
                                    env_hash: 0,
                                    session_id: 0,
                                    shell: "bash".to_string(),
                                    user: "chaos".to_string(),
                                };
                                match daemon.ingest("", &ctx) {
                                    Ok(_) => {
                                        eprintln!("[tala-sim] chaos: empty ingest unexpectedly succeeded");
                                    }
                                    Err(_) => {
                                        // Expected failure
                                    }
                                }
                            }
                            ChaosEvent::DegenerateQuery => {
                                let zero_vec = vec![0.0f32; dim];
                                match daemon.query(&zero_vec, 10) {
                                    Ok(_) => {}
                                    Err(e) => {
                                        eprintln!("[tala-sim] chaos degenerate query error: {}", e);
                                    }
                                }
                            }
                            ChaosEvent::InsightStress { k_clusters } => {
                                match daemon.insights(k_clusters) {
                                    Ok(_) => {}
                                    Err(e) => {
                                        eprintln!("[tala-sim] chaos insight stress error: {}", e);
                                    }
                                }
                            }
                        }
                    }
                }
            })
            .expect("spawn chaos thread")
    };

    // -----------------------------------------------------------------------
    // Thread: gauge updater
    // -----------------------------------------------------------------------
    let gauge_handle = {
        let metrics = Arc::clone(&metrics);
        let store_metrics = Arc::clone(daemon.store().metrics());
        let daemon_metrics_ref = Arc::clone(daemon.daemon_metrics());
        thread::Builder::new()
            .name("gauge-updater".into())
            .spawn(move || {
                loop {
                    thread::sleep(Duration::from_secs(5));
                    let uptime = start_time.elapsed().as_secs();
                    metrics.uptime_seconds.store(uptime, Ordering::Relaxed);

                    // Node/edge count approximation from ingested total
                    // (the daemon does not expose graph counts directly, so we use
                    // the ingested counter as a proxy for node count)
                    let ingested = metrics.intents_ingested_total.load(Ordering::Relaxed);
                    metrics.graph_node_count.store(ingested, Ordering::Relaxed);
                    // Edges are roughly 3-5x nodes after enough intents
                    let estimated_edges = ingested.saturating_mul(4);
                    metrics.graph_edge_count.store(estimated_edges, Ordering::Relaxed);
                    metrics.hnsw_index_size.store(ingested, Ordering::Relaxed);

                    // Store internals from StorageMetrics
                    let hot_len = store_metrics.hot_buffer_len.load(Ordering::Relaxed);
                    let hot_cap = store_metrics.hot_buffer_capacity.load(Ordering::Relaxed);
                    let fill_ratio = if hot_cap > 0 {
                        (hot_len * 1000) / hot_cap // scaled by 1000 for precision
                    } else {
                        0
                    };
                    metrics.hot_buffer_fill_ratio.store(fill_ratio, Ordering::Relaxed);
                    metrics.wal_entries_total.store(
                        store_metrics.wal_entry_count.load(Ordering::Relaxed),
                        Ordering::Relaxed,
                    );
                    metrics.segments_flushed_total.store(
                        store_metrics.segments_flushed_count.load(Ordering::Relaxed),
                        Ordering::Relaxed,
                    );
                    metrics.bytes_flushed_total.store(
                        store_metrics.total_bytes_flushed.load(Ordering::Relaxed),
                        Ordering::Relaxed,
                    );

                    // HNSW average search visited
                    let search_count = store_metrics.hnsw_search_count.load(Ordering::Relaxed);
                    let search_visited = store_metrics.hnsw_search_visited.load(Ordering::Relaxed);
                    let avg_visited = if search_count > 0 {
                        search_visited / search_count
                    } else {
                        0
                    };
                    metrics.hnsw_avg_search_visited.store(avg_visited, Ordering::Relaxed);

                    // Pipeline breakdown histograms from daemon metrics
                    // We compute per-operation averages in us and observe them
                    // into histograms. This gives a rolling view.
                    let extract_count = daemon_metrics_ref.extract_count.load(Ordering::Relaxed);
                    let extract_ns = daemon_metrics_ref.extract_ns.load(Ordering::Relaxed);
                    if extract_count > 0 {
                        let avg_us = (extract_ns / extract_count) / 1000;
                        metrics.extract_latency_us.observe(avg_us);
                    }

                    let wal_count = store_metrics.wal_append_count.load(Ordering::Relaxed);
                    let wal_ns = store_metrics.wal_append_ns.load(Ordering::Relaxed);
                    if wal_count > 0 {
                        let avg_us = (wal_ns / wal_count) / 1000;
                        metrics.wal_append_latency_us.observe(avg_us);
                    }

                    let hnsw_ins_count = store_metrics.hnsw_insert_count.load(Ordering::Relaxed);
                    let hnsw_ins_ns = store_metrics.hnsw_insert_ns.load(Ordering::Relaxed);
                    if hnsw_ins_count > 0 {
                        let avg_us = (hnsw_ins_ns / hnsw_ins_count) / 1000;
                        metrics.hnsw_insert_latency_us.observe(avg_us);
                    }

                    let edge_count = store_metrics.edge_search_count.load(Ordering::Relaxed);
                    let edge_ns = store_metrics.edge_search_ns.load(Ordering::Relaxed);
                    if edge_count > 0 {
                        let avg_us = (edge_ns / edge_count) / 1000;
                        metrics.edge_search_latency_us.observe(avg_us);
                    }

                    let hot_count = store_metrics.hot_push_count.load(Ordering::Relaxed);
                    let hot_ns = store_metrics.hot_push_ns.load(Ordering::Relaxed);
                    if hot_count > 0 {
                        let avg_us = (hot_ns / hot_count) / 1000;
                        metrics.hot_push_latency_us.observe(avg_us);
                    }

                    let seg_count = store_metrics.segment_flush_count.load(Ordering::Relaxed);
                    let seg_ns = store_metrics.segment_flush_ns.load(Ordering::Relaxed);
                    if seg_count > 0 {
                        let avg_us = (seg_ns / seg_count) / 1000;
                        metrics.segment_flush_latency_us.observe(avg_us);
                    }
                }
            })
            .expect("spawn gauge-updater thread")
    };

    // -----------------------------------------------------------------------
    // Join all threads (runs forever)
    // -----------------------------------------------------------------------
    let _ = metrics_server.join();
    let _ = ingest_handle.join();
    let _ = query_handle.join();
    let _ = insight_handle.join();
    let _ = replay_handle.join();
    let _ = chaos_handle.join();
    let _ = gauge_handle.join();
}
