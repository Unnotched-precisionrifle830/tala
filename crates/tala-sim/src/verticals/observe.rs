//! Observability & Diagnostics workload generator.
//!
//! Simulates the investigation journey:
//! anomaly → trace → diagnose → tune → confirm.

use rand::rngs::SmallRng;
use rand::Rng;
use tala_core::{Context, Outcome, Status};

use crate::workload::WorkloadGenerator;

const COMMANDS: &[&str] = &[
    // Metrics & alerting
    "curl -s http://localhost:9090/api/v1/query?query=http_request_duration_seconds",
    "curl -s http://localhost:9090/api/v1/query?query=node_memory_MemAvailable_bytes",
    "curl -s http://localhost:9090/api/v1/query?query=rate(http_requests_total[5m])",
    "curl -s http://localhost:3000/api/dashboards/uid/system-overview",
    "promtool check rules /etc/prometheus/rules/*.yml",
    // Tracing & profiling
    "perf record -g -p $(pgrep app) -- sleep 10",
    "perf report --stdio --sort=dso,sym | head -40",
    "perf stat -p $(pgrep app) -- sleep 5",
    "strace -c -p $(pgrep app) 2>&1 | head -30",
    "bpftrace -e 'tracepoint:syscalls:sys_enter_read { @[comm] = count(); }' -d 5",
    "flamegraph --pid $(pgrep app) --duration 10 -o /tmp/flamegraph.svg",
    // System diagnostics
    "vmstat 1 5",
    "iostat -xz 1 3",
    "sar -n DEV 1 3",
    "htop -t",
    "cat /proc/meminfo | grep -E 'MemTotal|MemFree|Cached|Buffers|SwapTotal|SwapFree'",
    "cat /proc/loadavg",
    "mpstat -P ALL 1 3",
    // Log analysis
    "journalctl -u app --since '1 hour ago' --priority=err --no-pager",
    "tail -1000 /var/log/app/access.log | awk '{print $9}' | sort | uniq -c | sort -rn",
    "grep 'slow query' /var/log/postgresql/postgresql.log | tail -20",
    "dmesg -T | grep -E 'error|warn|fail' | tail -20",
    // Tuning & confirmation
    "sysctl -w net.core.somaxconn=4096",
    "psql -c 'CREATE INDEX CONCURRENTLY idx_users_email ON users(email)'",
    "psql -c 'EXPLAIN ANALYZE SELECT * FROM orders WHERE created_at > now() - interval 1 day'",
    "redis-cli INFO memory | grep used_memory_human",
    "redis-cli SLOWLOG GET 10",
    "curl -s http://localhost:8080/debug/pprof/heap > /tmp/heap.prof",
    "go tool pprof -top /tmp/heap.prof | head -20",
];

const PROBE_COMMANDS: &[&str] = &[
    "perf flamegraph profile trace",
    "prometheus metrics query alert",
    "vmstat iostat system diagnostic",
    "psql explain analyze slow query",
];

pub struct ObserveWorkload;

impl WorkloadGenerator for ObserveWorkload {
    fn next_command(&self, rng: &mut SmallRng) -> (String, Context) {
        let idx = rng.gen_range(0..COMMANDS.len());
        let cmd = COMMANDS[idx].to_string();
        let ctx = Context {
            cwd: "/home/sre/diagnostics".to_string(),
            env_hash: 0x0b5e4e,
            session_id: rng.gen_range(1..100),
            shell: "bash".to_string(),
            user: "sre".to_string(),
        };
        (cmd, ctx)
    }

    fn simulate_outcome(&self, rng: &mut SmallRng) -> Outcome {
        let roll: f64 = rng.gen();
        let (status, exit_code) = if roll < 0.94 {
            (Status::Success, 0)
        } else if roll < 0.97 {
            (Status::Failure, rng.gen_range(1..128))
        } else {
            (Status::Partial, 0)
        };
        let latency_ns = rng.gen_range(5_000_000..15_000_000_000u64);
        Outcome {
            status,
            latency_ns,
            exit_code,
        }
    }

    fn probe_commands(&self) -> &[&str] {
        PROBE_COMMANDS
    }

    fn vertical_name(&self) -> &str {
        "observe"
    }
}
