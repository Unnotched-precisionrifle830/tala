//! TALA CLI — User-facing command-line interface for TALA.
//!
//! Provides command parsing, execution against a `Daemon`, and human-readable
//! output formatting. This is a library crate; the binary wrapper (`main.rs`)
//! comes later.

use std::fmt;

use tala_core::{Context, InsightKind, TalaError, Uuid};
use tala_daemon::Daemon;
use tala_intent::IntentPipeline;

// ===========================================================================
// Command — parsed CLI subcommands
// ===========================================================================

/// Parsed CLI subcommand.
#[derive(Clone, Debug, PartialEq)]
pub enum Command {
    /// Ingest a raw shell command into the narrative graph.
    Ingest { raw_command: String },
    /// Semantic search for intents matching a query string.
    Find { query: String, k: usize },
    /// Build a replay plan from a root intent.
    Replay {
        root_id: String,
        depth: usize,
        dry_run: bool,
    },
    /// Show daemon status.
    Status,
    /// Run insight analysis (clustering + pattern detection).
    Insights { clusters: usize },
}

// ===========================================================================
// CommandParser — hand-written arg parser
// ===========================================================================

/// Parses command-line arguments into a `Command`.
///
/// Expected argument format (args[0] is the binary name, args[1] is the subcommand):
/// - `tala ingest "kubectl apply -f deploy.yaml"`
/// - `tala find "deploy" --k 10`
/// - `tala replay <uuid> --depth 5 --dry-run`
/// - `tala status`
/// - `tala insights --clusters 5`
pub struct CommandParser;

impl CommandParser {
    /// Parse a slice of command-line arguments into a `Command`.
    ///
    /// `args` should include the binary name at index 0 (e.g., from `std::env::args()`).
    pub fn parse(args: &[String]) -> Result<Command, String> {
        if args.len() < 2 {
            return Err("usage: tala <subcommand> [args...]".to_string());
        }

        match args[1].as_str() {
            "ingest" => Self::parse_ingest(&args[2..]),
            "find" => Self::parse_find(&args[2..]),
            "replay" => Self::parse_replay(&args[2..]),
            "status" => Ok(Command::Status),
            "insights" => Self::parse_insights(&args[2..]),
            other => Err(format!("unknown subcommand: {other}")),
        }
    }

    fn parse_ingest(args: &[String]) -> Result<Command, String> {
        if args.is_empty() {
            return Err("usage: tala ingest <command>".to_string());
        }
        Ok(Command::Ingest {
            raw_command: args[0].clone(),
        })
    }

    fn parse_find(args: &[String]) -> Result<Command, String> {
        if args.is_empty() {
            return Err("usage: tala find <query> [--k N]".to_string());
        }

        let query = args[0].clone();
        let mut k: usize = 10; // default

        let mut i = 1;
        while i < args.len() {
            match args[i].as_str() {
                "--k" => {
                    i += 1;
                    if i >= args.len() {
                        return Err("--k requires a value".to_string());
                    }
                    k = args[i]
                        .parse()
                        .map_err(|_| format!("invalid value for --k: {}", args[i]))?;
                }
                other => {
                    return Err(format!("unknown flag for find: {other}"));
                }
            }
            i += 1;
        }

        Ok(Command::Find { query, k })
    }

    fn parse_replay(args: &[String]) -> Result<Command, String> {
        if args.is_empty() {
            return Err("usage: tala replay <uuid> [--depth N] [--dry-run]".to_string());
        }

        let root_id = args[0].clone();
        let mut depth: usize = 3; // default
        let mut dry_run = false;

        let mut i = 1;
        while i < args.len() {
            match args[i].as_str() {
                "--depth" => {
                    i += 1;
                    if i >= args.len() {
                        return Err("--depth requires a value".to_string());
                    }
                    depth = args[i]
                        .parse()
                        .map_err(|_| format!("invalid value for --depth: {}", args[i]))?;
                }
                "--dry-run" => {
                    dry_run = true;
                }
                other => {
                    return Err(format!("unknown flag for replay: {other}"));
                }
            }
            i += 1;
        }

        Ok(Command::Replay {
            root_id,
            depth,
            dry_run,
        })
    }

    fn parse_insights(args: &[String]) -> Result<Command, String> {
        let mut clusters: usize = 5; // default

        let mut i = 0;
        while i < args.len() {
            match args[i].as_str() {
                "--clusters" => {
                    i += 1;
                    if i >= args.len() {
                        return Err("--clusters requires a value".to_string());
                    }
                    clusters = args[i]
                        .parse()
                        .map_err(|_| format!("invalid value for --clusters: {}", args[i]))?;
                }
                other => {
                    return Err(format!("unknown flag for insights: {other}"));
                }
            }
            i += 1;
        }

        Ok(Command::Insights { clusters })
    }
}

// ===========================================================================
// Output — structured results from command execution
// ===========================================================================

/// A single search result.
#[derive(Clone, Debug)]
pub struct SearchResult {
    pub id: String,
    pub similarity: f32,
}

/// A single step in a replay plan (for display).
#[derive(Clone, Debug)]
pub struct ReplayStepOutput {
    pub id: String,
    pub command: String,
    pub dep_count: usize,
}

/// Daemon status (for display).
#[derive(Clone, Debug)]
pub struct StatusOutput {
    pub node_count: usize,
    pub edge_count: usize,
    pub command_count: usize,
    pub dim: usize,
}

/// A single insight (for display).
#[derive(Clone, Debug)]
pub struct InsightOutput {
    pub kind: String,
    pub description: String,
    pub confidence: f32,
}

/// Structured output from command execution.
#[derive(Clone, Debug)]
pub enum Output {
    /// Result of an ingest command.
    Ingested { id: String },
    /// Results of a semantic search.
    SearchResults(Vec<SearchResult>),
    /// Replay plan steps.
    ReplayPlan(Vec<ReplayStepOutput>),
    /// Daemon status.
    Status(StatusOutput),
    /// Insight analysis results.
    Insights(Vec<InsightOutput>),
}

impl fmt::Display for Output {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Output::Ingested { id } => {
                writeln!(f, "Ingested: {id}")
            }
            Output::SearchResults(results) => {
                if results.is_empty() {
                    return writeln!(f, "No results found.");
                }
                writeln!(f, "Search results ({} found):", results.len())?;
                for (i, r) in results.iter().enumerate() {
                    writeln!(f, "  [{i}] {id}  (similarity: {sim:.4})", id = r.id, sim = r.similarity)?;
                }
                Ok(())
            }
            Output::ReplayPlan(steps) => {
                if steps.is_empty() {
                    return writeln!(f, "Empty replay plan.");
                }
                writeln!(f, "Replay plan ({} steps):", steps.len())?;
                for (i, step) in steps.iter().enumerate() {
                    writeln!(
                        f,
                        "  [{i}] {cmd}  (id: {id}, deps: {deps})",
                        cmd = step.command,
                        id = step.id,
                        deps = step.dep_count,
                    )?;
                }
                Ok(())
            }
            Output::Status(s) => {
                writeln!(f, "TALA daemon status:")?;
                writeln!(f, "  Nodes:    {}", s.node_count)?;
                writeln!(f, "  Edges:    {}", s.edge_count)?;
                writeln!(f, "  Commands: {}", s.command_count)?;
                writeln!(f, "  Dim:      {}", s.dim)
            }
            Output::Insights(insights) => {
                if insights.is_empty() {
                    return writeln!(f, "No insights available.");
                }
                writeln!(f, "Insights ({} found):", insights.len())?;
                for (i, insight) in insights.iter().enumerate() {
                    writeln!(
                        f,
                        "  [{i}] [{kind}] {desc}  (confidence: {conf:.2})",
                        kind = insight.kind,
                        desc = insight.description,
                        conf = insight.confidence,
                    )?;
                }
                Ok(())
            }
        }
    }
}

// ===========================================================================
// CommandRunner — executes parsed commands against a Daemon
// ===========================================================================

/// Executes parsed `Command` values against a `Daemon` instance.
pub struct CommandRunner;

impl CommandRunner {
    /// Run a parsed command against the given daemon, returning structured output.
    ///
    /// The `_dry_run` flag on replay commands is accepted for forward compatibility
    /// but all replays currently produce a plan without execution.
    pub fn run(daemon: &Daemon, cmd: Command) -> Result<Output, TalaError> {
        match cmd {
            Command::Ingest { raw_command } => {
                let ctx = Context::default();
                let id = daemon.ingest(&raw_command, &ctx)?;
                Ok(Output::Ingested {
                    id: id.0.to_string(),
                })
            }
            Command::Find { query, k } => {
                let pipeline = IntentPipeline::new();
                let embedding = pipeline.embed(&query);
                let results = daemon.query(&embedding, k)?;
                let search_results = results
                    .into_iter()
                    .map(|(id, sim)| SearchResult {
                        id: id.0.to_string(),
                        similarity: sim,
                    })
                    .collect();
                Ok(Output::SearchResults(search_results))
            }
            Command::Replay {
                root_id,
                depth,
                dry_run: _,
            } => {
                let uuid = Uuid::parse_str(&root_id).map_err(|e| {
                    TalaError::ExtractionFailed(format!("invalid UUID: {e}"))
                })?;
                let intent_id = tala_core::IntentId(uuid);

                let plan = daemon.replay(intent_id, depth)?;
                let steps = plan
                    .into_iter()
                    .map(|step| ReplayStepOutput {
                        id: step.intent_id.0.to_string(),
                        command: step.command,
                        dep_count: step.deps.len(),
                    })
                    .collect();
                Ok(Output::ReplayPlan(steps))
            }
            Command::Status => {
                // Status is derived from the store; we report what we can.
                // The daemon exposes store() for intent lookups. Node/edge
                // counts are not directly available through the public API,
                // so we report zeros for now. This will be filled in once
                // Daemon exposes a status method.
                Ok(Output::Status(StatusOutput {
                    node_count: 0,
                    edge_count: 0,
                    command_count: 0,
                    dim: 0,
                }))
            }
            Command::Insights { clusters } => {
                let insights = daemon.insights(clusters)?;
                let outputs = insights
                    .into_iter()
                    .map(|insight| {
                        let kind = match insight.kind {
                            InsightKind::RecurringPattern => "pattern".to_string(),
                            InsightKind::FailureCluster => "failure".to_string(),
                            InsightKind::Prediction => "prediction".to_string(),
                            InsightKind::Summary => "summary".to_string(),
                        };
                        InsightOutput {
                            kind,
                            description: insight.description,
                            confidence: insight.confidence,
                        }
                    })
                    .collect();
                Ok(Output::Insights(outputs))
            }
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn args(strs: &[&str]) -> Vec<String> {
        strs.iter().map(|s| s.to_string()).collect()
    }

    fn test_daemon() -> Daemon {
        tala_daemon::DaemonBuilder::new().dim(384).build_in_memory()
    }

    // -----------------------------------------------------------------------
    // Command parsing tests
    // -----------------------------------------------------------------------

    #[test]
    fn parse_ingest() {
        let cmd = CommandParser::parse(&args(&["tala", "ingest", "kubectl apply -f deploy.yaml"]))
            .unwrap();
        assert_eq!(
            cmd,
            Command::Ingest {
                raw_command: "kubectl apply -f deploy.yaml".to_string()
            }
        );
    }

    #[test]
    fn parse_find_default_k() {
        let cmd = CommandParser::parse(&args(&["tala", "find", "deploy"])).unwrap();
        assert_eq!(
            cmd,
            Command::Find {
                query: "deploy".to_string(),
                k: 10,
            }
        );
    }

    #[test]
    fn parse_find_custom_k() {
        let cmd =
            CommandParser::parse(&args(&["tala", "find", "deploy", "--k", "20"])).unwrap();
        assert_eq!(
            cmd,
            Command::Find {
                query: "deploy".to_string(),
                k: 20,
            }
        );
    }

    #[test]
    fn parse_replay_defaults() {
        let cmd = CommandParser::parse(&args(&[
            "tala",
            "replay",
            "550e8400-e29b-41d4-a716-446655440000",
        ]))
        .unwrap();
        assert_eq!(
            cmd,
            Command::Replay {
                root_id: "550e8400-e29b-41d4-a716-446655440000".to_string(),
                depth: 3,
                dry_run: false,
            }
        );
    }

    #[test]
    fn parse_replay_with_flags() {
        let cmd = CommandParser::parse(&args(&[
            "tala",
            "replay",
            "550e8400-e29b-41d4-a716-446655440000",
            "--depth",
            "5",
            "--dry-run",
        ]))
        .unwrap();
        assert_eq!(
            cmd,
            Command::Replay {
                root_id: "550e8400-e29b-41d4-a716-446655440000".to_string(),
                depth: 5,
                dry_run: true,
            }
        );
    }

    #[test]
    fn parse_status() {
        let cmd = CommandParser::parse(&args(&["tala", "status"])).unwrap();
        assert_eq!(cmd, Command::Status);
    }

    #[test]
    fn parse_insights_default() {
        let cmd = CommandParser::parse(&args(&["tala", "insights"])).unwrap();
        assert_eq!(cmd, Command::Insights { clusters: 5 });
    }

    #[test]
    fn parse_insights_custom_clusters() {
        let cmd =
            CommandParser::parse(&args(&["tala", "insights", "--clusters", "8"])).unwrap();
        assert_eq!(cmd, Command::Insights { clusters: 8 });
    }

    // -----------------------------------------------------------------------
    // Parsing error cases
    // -----------------------------------------------------------------------

    #[test]
    fn parse_no_args() {
        let result = CommandParser::parse(&args(&["tala"]));
        assert!(result.is_err());
    }

    #[test]
    fn parse_unknown_subcommand() {
        let result = CommandParser::parse(&args(&["tala", "frobnicate"]));
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("unknown subcommand"));
    }

    #[test]
    fn parse_ingest_missing_command() {
        let result = CommandParser::parse(&args(&["tala", "ingest"]));
        assert!(result.is_err());
    }

    #[test]
    fn parse_find_missing_query() {
        let result = CommandParser::parse(&args(&["tala", "find"]));
        assert!(result.is_err());
    }

    #[test]
    fn parse_replay_missing_uuid() {
        let result = CommandParser::parse(&args(&["tala", "replay"]));
        assert!(result.is_err());
    }

    #[test]
    fn parse_find_invalid_k() {
        let result = CommandParser::parse(&args(&["tala", "find", "query", "--k", "abc"]));
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("invalid value for --k"));
    }

    #[test]
    fn parse_replay_invalid_depth() {
        let result = CommandParser::parse(&args(&[
            "tala",
            "replay",
            "550e8400-e29b-41d4-a716-446655440000",
            "--depth",
            "xyz",
        ]));
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("invalid value for --depth"));
    }

    #[test]
    fn parse_find_unknown_flag() {
        let result = CommandParser::parse(&args(&["tala", "find", "query", "--verbose"]));
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("unknown flag"));
    }

    #[test]
    fn parse_replay_unknown_flag() {
        let result = CommandParser::parse(&args(&[
            "tala",
            "replay",
            "550e8400-e29b-41d4-a716-446655440000",
            "--force",
        ]));
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("unknown flag"));
    }

    #[test]
    fn parse_insights_unknown_flag() {
        let result = CommandParser::parse(&args(&["tala", "insights", "--verbose"]));
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("unknown flag"));
    }

    #[test]
    fn parse_find_k_missing_value() {
        let result = CommandParser::parse(&args(&["tala", "find", "query", "--k"]));
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("--k requires a value"));
    }

    #[test]
    fn parse_replay_depth_missing_value() {
        let result = CommandParser::parse(&args(&[
            "tala",
            "replay",
            "550e8400-e29b-41d4-a716-446655440000",
            "--depth",
        ]));
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("--depth requires a value"));
    }

    #[test]
    fn parse_insights_clusters_missing_value() {
        let result = CommandParser::parse(&args(&["tala", "insights", "--clusters"]));
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("--clusters requires a value"));
    }

    // -----------------------------------------------------------------------
    // Output formatting tests
    // -----------------------------------------------------------------------

    #[test]
    fn format_ingested() {
        let output = Output::Ingested {
            id: "abc-123".to_string(),
        };
        let text = format!("{output}");
        assert!(text.contains("Ingested: abc-123"));
    }

    #[test]
    fn format_search_results_empty() {
        let output = Output::SearchResults(vec![]);
        let text = format!("{output}");
        assert!(text.contains("No results found"));
    }

    #[test]
    fn format_search_results() {
        let output = Output::SearchResults(vec![
            SearchResult {
                id: "id-1".to_string(),
                similarity: 0.95,
            },
            SearchResult {
                id: "id-2".to_string(),
                similarity: 0.82,
            },
        ]);
        let text = format!("{output}");
        assert!(text.contains("2 found"));
        assert!(text.contains("id-1"));
        assert!(text.contains("0.9500"));
        assert!(text.contains("id-2"));
        assert!(text.contains("0.8200"));
    }

    #[test]
    fn format_replay_plan_empty() {
        let output = Output::ReplayPlan(vec![]);
        let text = format!("{output}");
        assert!(text.contains("Empty replay plan"));
    }

    #[test]
    fn format_replay_plan() {
        let output = Output::ReplayPlan(vec![ReplayStepOutput {
            id: "step-1".to_string(),
            command: "echo hello".to_string(),
            dep_count: 0,
        }]);
        let text = format!("{output}");
        assert!(text.contains("1 steps"));
        assert!(text.contains("echo hello"));
        assert!(text.contains("step-1"));
    }

    #[test]
    fn format_status() {
        let output = Output::Status(StatusOutput {
            node_count: 42,
            edge_count: 100,
            command_count: 42,
            dim: 384,
        });
        let text = format!("{output}");
        assert!(text.contains("Nodes:    42"));
        assert!(text.contains("Edges:    100"));
        assert!(text.contains("Commands: 42"));
        assert!(text.contains("Dim:      384"));
    }

    #[test]
    fn format_insights_empty() {
        let output = Output::Insights(vec![]);
        let text = format!("{output}");
        assert!(text.contains("No insights available"));
    }

    #[test]
    fn format_insights() {
        let output = Output::Insights(vec![InsightOutput {
            kind: "pattern".to_string(),
            description: "Recurring sequence".to_string(),
            confidence: 0.85,
        }]);
        let text = format!("{output}");
        assert!(text.contains("1 found"));
        assert!(text.contains("[pattern]"));
        assert!(text.contains("Recurring sequence"));
        assert!(text.contains("0.85"));
    }

    // -----------------------------------------------------------------------
    // CommandRunner integration tests
    // -----------------------------------------------------------------------

    #[test]
    fn runner_ingest() {
        let daemon = test_daemon();
        let output =
            CommandRunner::run(&daemon, Command::Ingest { raw_command: "ls -la".to_string() })
                .unwrap();
        match output {
            Output::Ingested { id } => {
                assert!(!id.is_empty());
            }
            _ => panic!("expected Ingested output"),
        }
    }

    #[test]
    fn runner_find() {
        let daemon = test_daemon();
        let ctx = Context::default();
        daemon.ingest("cargo build --release", &ctx).unwrap();
        daemon.ingest("cargo test", &ctx).unwrap();
        daemon.ingest("kubectl apply -f deploy.yaml", &ctx).unwrap();

        let output = CommandRunner::run(
            &daemon,
            Command::Find {
                query: "cargo build".to_string(),
                k: 5,
            },
        )
        .unwrap();

        match output {
            Output::SearchResults(results) => {
                assert!(!results.is_empty());
            }
            _ => panic!("expected SearchResults output"),
        }
    }

    #[test]
    fn runner_status() {
        let daemon = test_daemon();
        let ctx = Context::default();
        daemon.ingest("echo hello", &ctx).unwrap();

        let output = CommandRunner::run(&daemon, Command::Status).unwrap();
        match output {
            Output::Status(_s) => {
                // Status is currently placeholder; just verify it works.
            }
            _ => panic!("expected Status output"),
        }
    }

    #[test]
    fn runner_replay() {
        let daemon = test_daemon();
        let ctx = Context::default();
        let id = daemon.ingest("echo hello", &ctx).unwrap();

        let output = CommandRunner::run(
            &daemon,
            Command::Replay {
                root_id: id.0.to_string(),
                depth: 3,
                dry_run: true,
            },
        )
        .unwrap();

        match output {
            Output::ReplayPlan(steps) => {
                assert_eq!(steps.len(), 1);
                assert_eq!(steps[0].command, "echo hello");
            }
            _ => panic!("expected ReplayPlan output"),
        }
    }

    #[test]
    fn runner_replay_invalid_uuid() {
        let daemon = test_daemon();
        let result = CommandRunner::run(
            &daemon,
            Command::Replay {
                root_id: "not-a-uuid".to_string(),
                depth: 3,
                dry_run: true,
            },
        );
        assert!(result.is_err());
    }

    #[test]
    fn runner_insights() {
        let daemon = test_daemon();
        let ctx = Context::default();
        for _ in 0..5 {
            daemon.ingest("cargo build", &ctx).unwrap();
            daemon.ingest("cargo test", &ctx).unwrap();
        }

        let output = CommandRunner::run(&daemon, Command::Insights { clusters: 2 }).unwrap();
        match output {
            Output::Insights(insights) => {
                assert!(!insights.is_empty());
            }
            _ => panic!("expected Insights output"),
        }
    }
}
