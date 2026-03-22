//! Chaos engine with periodic fault injection.

use rand::rngs::SmallRng;
use rand::Rng;

/// A chaos event that can be injected into the system.
pub enum ChaosEvent {
    /// Force the next ingest to be treated as a failure.
    ForcedFailure,
    /// Simulate a latency spike by multiplying sleep duration.
    LatencySpike { multiplier: u32 },
    /// Rapidly ingest `count` copies of the same command (retry storm).
    RetryStorm { count: usize },
    /// Attempt an ingest that is expected to fail.
    IngestFailure,
    /// Query with a degenerate (zero) vector.
    DegenerateQuery,
    /// Run insights with an unusually large k.
    InsightStress { k_clusters: usize },
}

impl ChaosEvent {
    /// Human-readable name for logging.
    pub fn name(&self) -> &'static str {
        match self {
            ChaosEvent::ForcedFailure => "ForcedFailure",
            ChaosEvent::LatencySpike { .. } => "LatencySpike",
            ChaosEvent::RetryStorm { .. } => "RetryStorm",
            ChaosEvent::IngestFailure => "IngestFailure",
            ChaosEvent::DegenerateQuery => "DegenerateQuery",
            ChaosEvent::InsightStress { .. } => "InsightStress",
        }
    }
}

/// Engine that probabilistically triggers chaos events.
pub struct ChaosEngine {
    /// Minimum interval between triggers (seconds). Used for sleep in the chaos loop.
    pub interval_s: u64,
    /// Probability [0.0, 1.0] that a trigger check fires.
    probability: f64,
}

impl ChaosEngine {
    pub fn new(interval_s: u64, probability: f64) -> Self {
        Self {
            interval_s,
            probability: probability.clamp(0.0, 1.0),
        }
    }

    /// Roll the dice and maybe return a chaos event.
    ///
    /// Weighted selection:
    ///   ForcedFailure   30%
    ///   LatencySpike    25%
    ///   RetryStorm      20%
    ///   IngestFailure   10%
    ///   DegenerateQuery 10%
    ///   InsightStress    5%
    pub fn maybe_trigger(&self, rng: &mut SmallRng) -> Option<ChaosEvent> {
        let roll: f64 = rng.gen();
        if roll >= self.probability {
            return None;
        }

        let pick: f64 = rng.gen();
        let event = if pick < 0.30 {
            ChaosEvent::ForcedFailure
        } else if pick < 0.55 {
            ChaosEvent::LatencySpike {
                multiplier: rng.gen_range(2..10),
            }
        } else if pick < 0.75 {
            ChaosEvent::RetryStorm {
                count: rng.gen_range(3..15),
            }
        } else if pick < 0.85 {
            ChaosEvent::IngestFailure
        } else if pick < 0.95 {
            ChaosEvent::DegenerateQuery
        } else {
            ChaosEvent::InsightStress { k_clusters: 50 }
        };

        Some(event)
    }
}
