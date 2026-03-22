//! Incident Response workload generator.
//!
//! Simulates the diagnostic journey when something breaks:
//! alert → triage → root cause → fix → verify.

use rand::rngs::SmallRng;
use rand::Rng;
use tala_core::{Context, Outcome, Status};

use crate::workload::WorkloadGenerator;

const COMMANDS: &[&str] = &[
    // Alert triage
    "ssh root@web-03.prod 'journalctl -u nginx --since \"10 min ago\" | tail -100'",
    "ssh root@web-03.prod 'dmesg | grep -i \"oom\\|kill\" | tail -20'",
    "kubectl get events -n production --sort-by=.lastTimestamp | tail -30",
    "kubectl describe pod api-gateway-7f8b9-xk2m4 -n production",
    "curl -s http://localhost:9090/api/v1/query?query=up",
    // Diagnostics
    "free -h && cat /proc/meminfo | grep -E 'MemTotal|MemFree|Cached'",
    "top -bn1 | head -20",
    "vmstat 1 5",
    "iostat -xz 1 3",
    "ss -tlnp | grep ':8080'",
    "strace -p $(pgrep api-gateway) -c -f -e trace=network 2>&1 | head -30",
    "tcpdump -i eth0 -n port 5432 -c 50 -w /tmp/capture.pcap",
    "lsof -i :5432 | head -20",
    // Log analysis
    "journalctl -u api-gateway --since '1 hour ago' --priority=err --no-pager | tail -50",
    "grep -c 'ERROR\\|FATAL' /var/log/syslog",
    "tail -500 /var/log/nginx/error.log | awk '{print $NF}' | sort | uniq -c | sort -rn | head -10",
    "kubectl logs deployment/api-gateway -n production --tail=200 | grep -i exception",
    "journalctl -u postgresql --since '30 min ago' | grep -E 'FATAL|PANIC'",
    // Fix & remediation
    "systemctl restart api-gateway.service",
    "kubectl rollout restart deployment/api-gateway -n production",
    "kubectl scale deployment/api-gateway -n production --replicas=4",
    "sysctl -w vm.overcommit_memory=1",
    "echo 1 > /proc/sys/vm/drop_caches",
    "iptables -A INPUT -s 10.0.0.0/8 -p tcp --dport 5432 -j ACCEPT",
    // Verification
    "curl -s http://localhost:8080/healthz",
    "kubectl rollout status deployment/api-gateway -n production --timeout=120s",
    "ab -n 100 -c 10 http://localhost:8080/api/v1/status",
    "watch -n2 'kubectl get pods -n production | grep api-gateway'",
    "uptime && w",
];

const PROBE_COMMANDS: &[&str] = &[
    "journalctl dmesg oom kill diagnostic",
    "kubectl logs pod restart troubleshoot",
    "strace tcpdump network trace",
    "systemctl restart remediation fix",
];

pub struct IncidentWorkload;

impl WorkloadGenerator for IncidentWorkload {
    fn next_command(&self, rng: &mut SmallRng) -> (String, Context) {
        let idx = rng.gen_range(0..COMMANDS.len());
        let cmd = COMMANDS[idx].to_string();
        let ctx = Context {
            cwd: "/root".to_string(),
            env_hash: 0x1c1de7,
            session_id: rng.gen_range(1..100),
            shell: "bash".to_string(),
            user: "oncall".to_string(),
        };
        (cmd, ctx)
    }

    fn simulate_outcome(&self, rng: &mut SmallRng) -> Outcome {
        let roll: f64 = rng.gen();
        // Incident commands have higher failure rate — diagnostics surface errors
        let (status, exit_code) = if roll < 0.85 {
            (Status::Success, 0)
        } else if roll < 0.93 {
            (Status::Failure, rng.gen_range(1..128))
        } else {
            (Status::Partial, 0)
        };
        let latency_ns = rng.gen_range(10_000_000..8_000_000_000u64);
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
        "incident"
    }
}
