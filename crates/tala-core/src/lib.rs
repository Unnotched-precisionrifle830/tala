//! TALA Core — Foundation types and traits for the intent-native narrative execution layer.

pub use uuid::Uuid;

/// Unique identifier for an intent node.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct IntentId(pub Uuid);

impl IntentId {
    pub fn random() -> Self {
        Self(Uuid::new_v4())
    }
    pub fn as_bytes(&self) -> &[u8; 16] {
        self.0.as_bytes()
    }
}

impl Default for IntentId {
    fn default() -> Self {
        Self::random()
    }
}

/// Segment identifier (monotonic).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SegmentId(pub u64);

/// Edge relation type.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RelationType {
    Causal = 0,
    Temporal = 1,
    Dependency = 2,
    Retry = 3,
    Branch = 4,
}

/// Outcome status.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    Pending = 0,
    Success = 1,
    Failure = 2,
    Partial = 3,
}

/// Outcome of an intent execution.
#[derive(Clone, Debug)]
pub struct Outcome {
    pub status: Status,
    pub latency_ns: u64,
    pub exit_code: i32,
}

/// A fully resolved intent node.
#[derive(Clone, Debug)]
pub struct Intent {
    pub id: IntentId,
    pub timestamp: u64,
    pub raw_command: String,
    pub embedding: Vec<f32>,
    pub context_hash: u64,
    pub parent_ids: Vec<IntentId>,
    pub outcome: Option<Outcome>,
    pub confidence: f32,
}

/// Edge between two intent nodes.
#[derive(Clone, Debug)]
pub struct Edge {
    pub from: IntentId,
    pub to: IntentId,
    pub relation: RelationType,
    pub weight: f32,
}

/// Time range for temporal queries (inclusive start, exclusive end, nanosecond epoch).
#[derive(Clone, Copy, Debug)]
pub struct TimeRange {
    pub start: u64,
    pub end: u64,
}

/// Execution context captured at intent time.
#[derive(Clone, Debug, Default)]
pub struct Context {
    pub cwd: String,
    pub env_hash: u64,
    pub session_id: u64,
    pub shell: String,
    pub user: String,
}

/// A step in a replay plan.
#[derive(Clone, Debug)]
pub struct ReplayStep {
    pub intent_id: IntentId,
    pub command: String,
    pub deps: Vec<IntentId>,
}

/// Result of a replay execution.
#[derive(Clone, Debug)]
pub struct ReplayResult {
    pub step: ReplayStep,
    pub outcome: Outcome,
    pub skipped: bool,
}

/// Intent category for classification.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum IntentCategory {
    Build,
    Deploy,
    Debug,
    Configure,
    Query,
    Navigate,
    Other(String),
}

/// Insight from narrative analysis.
#[derive(Clone, Debug)]
pub struct Insight {
    pub kind: InsightKind,
    pub description: String,
    pub intent_ids: Vec<IntentId>,
    pub confidence: f32,
}

/// Classification of insights.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InsightKind {
    RecurringPattern,
    FailureCluster,
    Prediction,
    Summary,
}

/// Unified error type.
#[derive(Debug, thiserror::Error)]
pub enum TalaError {
    #[error("segment not found: {0:?}")]
    SegmentNotFound(SegmentId),
    #[error("segment corrupted: {0}")]
    SegmentCorrupted(String),
    #[error("node not found: {0:?}")]
    NodeNotFound(IntentId),
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },
    #[error("extraction failed: {0}")]
    ExtractionFailed(String),
    #[error("cycle detected in narrative graph")]
    CycleDetected,
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

/// Core storage abstraction. Implemented by tala-store::StorageEngine.
pub trait IntentStore: Send + Sync {
    fn insert(&self, intent: Intent) -> Result<IntentId, TalaError>;
    fn get(&self, id: IntentId) -> Result<Option<Intent>, TalaError>;
    fn query_semantic(&self, embedding: &[f32], k: usize) -> Result<Vec<(IntentId, f32)>, TalaError>;
    fn query_temporal(&self, range: TimeRange) -> Result<Vec<Intent>, TalaError>;
    fn attach_outcome(&self, id: IntentId, outcome: Outcome) -> Result<(), TalaError>;
}

/// Intent extraction from raw input. Implemented by tala-intent.
pub trait IntentExtractor: Send + Sync {
    fn extract(&self, raw: &str, context: &Context) -> Result<Intent, TalaError>;
}
