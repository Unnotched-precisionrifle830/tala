//! System Provisioning workload generator.
//!
//! Simulates setting up and configuring machines:
//! install → configure → harden → test → activate.

use rand::rngs::SmallRng;
use rand::Rng;
use tala_core::{Context, Outcome, Status};

use crate::workload::WorkloadGenerator;

const COMMANDS: &[&str] = &[
    // Package management
    "apt-get update",
    "apt-get install -y nginx postgresql-client redis-tools htop curl jq",
    "apt-get upgrade -y --with-new-pkgs",
    "dpkg -l | grep -c '^ii'",
    "snap install --classic certbot",
    // Nginx configuration
    "cp /etc/nginx/nginx.conf /etc/nginx/nginx.conf.bak",
    "vim /etc/nginx/sites-available/app.conf",
    "ln -sf /etc/nginx/sites-available/app.conf /etc/nginx/sites-enabled/",
    "nginx -t",
    "systemctl enable --now nginx",
    "systemctl reload nginx",
    // TLS & certificates
    "openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout /etc/ssl/private/app.key -out /etc/ssl/certs/app.crt",
    "certbot certonly --nginx -d app.example.com --non-interactive --agree-tos",
    "openssl x509 -in /etc/ssl/certs/app.crt -noout -enddate",
    // Kernel tuning
    "sysctl -w vm.swappiness=10",
    "sysctl -w net.core.somaxconn=4096",
    "sysctl -w net.ipv4.tcp_max_syn_backlog=4096",
    "echo 'vm.swappiness=10' >> /etc/sysctl.d/99-tala.conf",
    "sysctl -p /etc/sysctl.d/99-tala.conf",
    "ulimit -n 65535",
    // User & permissions
    "useradd -m -s /bin/bash -G sudo deploy",
    "chmod 750 /opt/app",
    "chown -R deploy:deploy /opt/app",
    "mkdir -p /opt/app/{bin,conf,data,logs}",
    "ssh-keygen -t ed25519 -f /home/deploy/.ssh/id_ed25519 -N ''",
    // Systemd services
    "cp /opt/app/conf/app.service /etc/systemd/system/",
    "systemctl daemon-reload",
    "systemctl enable --now app.service",
    "systemctl status app.service",
    // Verification
    "curl -s http://localhost:8080/healthz",
    "ss -tlnp | grep -E '80|443|8080'",
    "df -h /opt/app /var/log",
    "lsblk -o NAME,SIZE,FSTYPE,MOUNTPOINT",
    "hostnamectl set-hostname app-01.prod",
];

const PROBE_COMMANDS: &[&str] = &[
    "apt-get install package provision",
    "nginx systemctl configure service",
    "openssl certbot tls certificate",
    "sysctl kernel tune performance",
];

pub struct ProvisionWorkload;

impl WorkloadGenerator for ProvisionWorkload {
    fn next_command(&self, rng: &mut SmallRng) -> (String, Context) {
        let idx = rng.gen_range(0..COMMANDS.len());
        let cmd = COMMANDS[idx].to_string();
        let ctx = Context {
            cwd: "/opt/app".to_string(),
            env_hash: 0x940510,
            session_id: rng.gen_range(1..100),
            shell: "bash".to_string(),
            user: "root".to_string(),
        };
        (cmd, ctx)
    }

    fn simulate_outcome(&self, rng: &mut SmallRng) -> Outcome {
        let roll: f64 = rng.gen();
        let (status, exit_code) = if roll < 0.92 {
            (Status::Success, 0)
        } else if roll < 0.96 {
            (Status::Failure, rng.gen_range(1..128))
        } else {
            (Status::Partial, 0)
        };
        let latency_ns = rng.gen_range(100_000_000..20_000_000_000u64);
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
        "provision"
    }
}
