//! Continuous Deployment workload generator.
//!
//! Simulates the release lifecycle:
//! build → test → stage → canary → rollout.

use rand::rngs::SmallRng;
use rand::Rng;
use tala_core::{Context, Outcome, Status};

use crate::workload::WorkloadGenerator;

const COMMANDS: &[&str] = &[
    // Build
    "git log --oneline -5",
    "git diff --stat HEAD~1",
    "cargo build --release 2>&1 | tail -5",
    "cargo test --workspace 2>&1 | tail -20",
    "cargo clippy --workspace -- -D warnings",
    "docker build -t app:$(git rev-parse --short HEAD) .",
    "docker push registry.example.com/app:$(git rev-parse --short HEAD)",
    // Stage
    "ansible-playbook -i inventory/staging deploy.yml --check",
    "ansible-playbook -i inventory/staging deploy.yml",
    "terraform plan -out=tfplan -var-file=staging.tfvars",
    "terraform apply tfplan",
    "kubectl apply -f k8s/staging/ --dry-run=server",
    "kubectl apply -f k8s/staging/",
    // Canary
    "kubectl set image deployment/app app=registry.example.com/app:abc123 -n canary",
    "kubectl rollout status deployment/app -n canary --timeout=120s",
    "curl -s http://canary.example.com/healthz",
    "curl -s http://canary.example.com/api/v1/readiness",
    "ab -n 1000 -c 50 http://canary.example.com/api/v1/ping",
    // Rollout
    "kubectl set image deployment/app app=registry.example.com/app:abc123 -n production",
    "kubectl rollout status deployment/app -n production --timeout=300s",
    "kubectl get pods -n production -l app=app -o wide",
    // Rollback
    "kubectl rollout undo deployment/app -n production",
    "kubectl rollout history deployment/app -n production",
    "git revert --no-edit HEAD",
    // Verification
    "curl -s http://localhost:8080/healthz",
    "curl -s http://localhost:8080/api/v1/version",
    "helm list -n production",
    "kubectl top pods -n production --sort-by=cpu",
];

const PROBE_COMMANDS: &[&str] = &[
    "cargo build test deploy release",
    "kubectl rollout deployment canary",
    "docker build push registry image",
    "terraform plan apply infrastructure",
];

pub struct DeployWorkload;

impl WorkloadGenerator for DeployWorkload {
    fn next_command(&self, rng: &mut SmallRng) -> (String, Context) {
        let idx = rng.gen_range(0..COMMANDS.len());
        let cmd = COMMANDS[idx].to_string();
        let ctx = Context {
            cwd: "/home/deploy/app".to_string(),
            env_hash: 0xde910d,
            session_id: rng.gen_range(1..100),
            shell: "bash".to_string(),
            user: "deploy".to_string(),
        };
        (cmd, ctx)
    }

    fn simulate_outcome(&self, rng: &mut SmallRng) -> Outcome {
        let roll: f64 = rng.gen();
        let (status, exit_code) = if roll < 0.93 {
            (Status::Success, 0)
        } else if roll < 0.97 {
            (Status::Failure, rng.gen_range(1..128))
        } else {
            (Status::Partial, 0)
        };
        let latency_ns = rng.gen_range(50_000_000..30_000_000_000u64);
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
        "deploy"
    }
}
