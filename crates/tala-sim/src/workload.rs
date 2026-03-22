//! Workload generator trait and factory.

use rand::rngs::SmallRng;
use tala_core::{Context, Outcome};

/// Trait for generating vertical-specific workloads.
pub trait WorkloadGenerator: Send + Sync {
    /// Generate the next command string and execution context.
    fn next_command(&self, rng: &mut SmallRng) -> (String, Context);
    /// Simulate an execution outcome for the most recently generated command.
    fn simulate_outcome(&self, rng: &mut SmallRng) -> Outcome;
    /// Return a set of representative probe commands for semantic queries.
    fn probe_commands(&self) -> &[&str];
    /// Return the vertical name.
    fn vertical_name(&self) -> &str;
}

/// Create a workload generator for the given domain name.
///
/// Supported domains: "incident", "deploy", "observe", "provision".
/// Legacy names ("medical", "financial", "ecommerce", "gaming") are mapped for compatibility.
pub fn create_generator(vertical: &str) -> Box<dyn WorkloadGenerator> {
    match vertical {
        "incident" | "medical" => Box::new(crate::verticals::incident::IncidentWorkload),
        "deploy" | "financial" => Box::new(crate::verticals::deploy::DeployWorkload),
        "observe" | "ecommerce" => Box::new(crate::verticals::observe::ObserveWorkload),
        "provision" | "gaming" => Box::new(crate::verticals::provision::ProvisionWorkload),
        other => {
            eprintln!("[tala-sim] unknown domain '{}', defaulting to incident", other);
            Box::new(crate::verticals::incident::IncidentWorkload)
        }
    }
}
