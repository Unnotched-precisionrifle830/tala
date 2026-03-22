//! TALA Weave — Adaptive replay engine for narrative graph replay.
//!
//! Takes a narrative graph (or subgraph) and produces a topologically sorted
//! replay plan, applies environment transforms, checks idempotency, and
//! orchestrates execution through a caller-supplied executor closure.

use std::collections::{HashMap, HashSet, VecDeque};

use tala_core::{IntentId, Outcome, ReplayResult, ReplayStep, Status, TalaError};
use tala_graph::NarrativeGraph;

// ---------------------------------------------------------------------------
// Planner — topological sort via Kahn's algorithm
// ---------------------------------------------------------------------------

/// Builds a topologically sorted replay plan from the given intent IDs and graph edges.
///
/// Uses Kahn's algorithm. Returns `TalaError::CycleDetected` if the subgraph
/// induced by `intent_ids` contains a cycle.
pub fn build_plan(
    graph: &NarrativeGraph,
    intent_ids: &[IntentId],
    commands: &HashMap<IntentId, String>,
) -> Result<Vec<ReplayStep>, TalaError> {
    let id_set: HashSet<IntentId> = intent_ids.iter().copied().collect();

    // Build in-degree map and adjacency restricted to the requested subgraph.
    let mut in_degree: HashMap<IntentId, usize> = HashMap::new();
    let mut adj: HashMap<IntentId, Vec<IntentId>> = HashMap::new();

    for &id in &id_set {
        in_degree.entry(id).or_insert(0);
        adj.entry(id).or_default();
    }

    for &id in &id_set {
        for &(successor, _, _) in graph.successors(id) {
            if id_set.contains(&successor) {
                *in_degree.entry(successor).or_insert(0) += 1;
                adj.entry(id).or_default().push(successor);
            }
        }
    }

    // Kahn's algorithm
    let mut queue: VecDeque<IntentId> = VecDeque::new();
    for (&id, &deg) in &in_degree {
        if deg == 0 {
            queue.push_back(id);
        }
    }

    let mut sorted: Vec<IntentId> = Vec::with_capacity(id_set.len());

    while let Some(node) = queue.pop_front() {
        sorted.push(node);
        if let Some(neighbors) = adj.get(&node) {
            for &neighbor in neighbors {
                if let Some(deg) = in_degree.get_mut(&neighbor) {
                    *deg -= 1;
                    if *deg == 0 {
                        queue.push_back(neighbor);
                    }
                }
            }
        }
    }

    if sorted.len() != id_set.len() {
        return Err(TalaError::CycleDetected);
    }

    // Convert sorted IDs into ReplaySteps.
    let steps = sorted
        .into_iter()
        .map(|id| {
            let command = commands
                .get(&id)
                .cloned()
                .unwrap_or_default();

            // Dependencies: predecessors within the subgraph.
            let deps: Vec<IntentId> = graph
                .predecessors(id)
                .iter()
                .filter(|(pred, _, _)| id_set.contains(pred))
                .map(|(pred, _, _)| *pred)
                .collect();

            ReplayStep {
                intent_id: id,
                command,
                deps,
            }
        })
        .collect();

    Ok(steps)
}

// ---------------------------------------------------------------------------
// Transform — variable substitution in commands
// ---------------------------------------------------------------------------

/// Replaces `${VAR}` patterns in `input` with values from `vars`.
/// Unknown variables are left as-is.
pub fn substitute_vars(input: &str, vars: &HashMap<String, String>) -> String {
    let mut result = String::with_capacity(input.len());
    let bytes = input.as_bytes();
    let len = bytes.len();
    let mut i = 0;

    while i < len {
        if i + 1 < len && bytes[i] == b'$' && bytes[i + 1] == b'{' {
            // Scan for closing brace.
            if let Some(close) = input[i + 2..].find('}') {
                let var_name = &input[i + 2..i + 2 + close];
                if let Some(value) = vars.get(var_name) {
                    result.push_str(value);
                } else {
                    // Unknown variable — leave as-is.
                    result.push_str(&input[i..i + 2 + close + 1]);
                }
                i += 2 + close + 1;
                continue;
            }
        }
        // SAFETY: bytes[i] is valid UTF-8 since `input` is a &str and we index
        // byte-by-byte only within ASCII-safe checks ($ and {). For non-ASCII
        // we fall through here and push the full char.
        let ch = input[i..].chars().next().unwrap_or('\0');
        result.push(ch);
        i += ch.len_utf8();
    }

    result
}

// ---------------------------------------------------------------------------
// Idempotency — skip already-completed steps
// ---------------------------------------------------------------------------

/// Filters out steps whose `intent_id` appears in `completed`.
pub fn filter_completed(steps: Vec<ReplayStep>, completed: &HashSet<IntentId>) -> Vec<ReplayStep> {
    steps
        .into_iter()
        .filter(|s| !completed.contains(&s.intent_id))
        .collect()
}

// ---------------------------------------------------------------------------
// ReplayEngine — orchestrates the full replay pipeline
// ---------------------------------------------------------------------------

/// Executor function signature: takes a command string, returns an Outcome.
pub type Executor = Box<dyn FnMut(&str) -> Outcome>;

/// Configuration for a replay run.
pub struct ReplayConfig {
    /// Environment variable substitutions to apply.
    pub vars: HashMap<String, String>,
    /// Set of intent IDs already completed (for idempotency).
    pub completed: HashSet<IntentId>,
    /// If true, skip execution and return the plan.
    pub dry_run: bool,
}

impl Default for ReplayConfig {
    fn default() -> Self {
        Self {
            vars: HashMap::new(),
            completed: HashSet::new(),
            dry_run: true,
        }
    }
}

/// The replay engine. Builds a plan, applies transforms, checks idempotency,
/// and either dry-runs or executes each step through the provided executor.
pub struct ReplayEngine {
    config: ReplayConfig,
    executor: Option<Executor>,
}

impl ReplayEngine {
    /// Create a new replay engine with the given configuration.
    pub fn new(config: ReplayConfig) -> Self {
        Self {
            config,
            executor: None,
        }
    }

    /// Set the executor closure for live replay.
    pub fn with_executor(mut self, executor: Executor) -> Self {
        self.executor = Some(executor);
        self
    }

    /// Produce a dry-run plan: the ordered, transformed steps that *would*
    /// be executed, with completed steps removed.
    pub fn dry_run(
        &self,
        graph: &NarrativeGraph,
        intent_ids: &[IntentId],
        commands: &HashMap<IntentId, String>,
    ) -> Result<Vec<ReplayStep>, TalaError> {
        let plan = build_plan(graph, intent_ids, commands)?;
        let plan = filter_completed(plan, &self.config.completed);
        let plan = plan
            .into_iter()
            .map(|mut step| {
                step.command = substitute_vars(&step.command, &self.config.vars);
                step
            })
            .collect();
        Ok(plan)
    }

    /// Execute the replay. Returns a `ReplayResult` for every step in the plan
    /// (including skipped ones).
    ///
    /// If `dry_run` is set in the config, every step gets a synthetic
    /// `Status::Pending` outcome and `skipped = false`.
    pub fn execute(
        &mut self,
        graph: &NarrativeGraph,
        intent_ids: &[IntentId],
        commands: &HashMap<IntentId, String>,
    ) -> Result<Vec<ReplayResult>, TalaError> {
        let plan = build_plan(graph, intent_ids, commands)?;
        let mut results: Vec<ReplayResult> = Vec::with_capacity(plan.len());

        for mut step in plan {
            // Idempotency check.
            if self.config.completed.contains(&step.intent_id) {
                results.push(ReplayResult {
                    step,
                    outcome: Outcome {
                        status: Status::Success,
                        latency_ns: 0,
                        exit_code: 0,
                    },
                    skipped: true,
                });
                continue;
            }

            // Apply variable substitution.
            step.command = substitute_vars(&step.command, &self.config.vars);

            if self.config.dry_run {
                results.push(ReplayResult {
                    step,
                    outcome: Outcome {
                        status: Status::Pending,
                        latency_ns: 0,
                        exit_code: 0,
                    },
                    skipped: false,
                });
            } else if let Some(ref mut exec) = self.executor {
                let outcome = exec(&step.command);
                results.push(ReplayResult {
                    step,
                    outcome,
                    skipped: false,
                });
            } else {
                // No executor and not dry-run — treat as dry-run.
                results.push(ReplayResult {
                    step,
                    outcome: Outcome {
                        status: Status::Pending,
                        latency_ns: 0,
                        exit_code: 0,
                    },
                    skipped: false,
                });
            }
        }

        Ok(results)
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a linear graph A -> B -> C and return (graph, ids, commands).
    fn linear_graph() -> (NarrativeGraph, Vec<IntentId>, HashMap<IntentId, String>) {
        let mut graph = NarrativeGraph::new();
        let a = IntentId::random();
        let b = IntentId::random();
        let c = IntentId::random();

        graph.insert_node(a, 1, 1.0);
        graph.insert_node(b, 2, 1.0);
        graph.insert_node(c, 3, 1.0);

        graph.add_edge(a, b, tala_core::RelationType::Causal, 1.0);
        graph.add_edge(b, c, tala_core::RelationType::Causal, 1.0);

        let mut commands = HashMap::new();
        commands.insert(a, "echo A".to_string());
        commands.insert(b, "echo B".to_string());
        commands.insert(c, "echo C".to_string());

        (graph, vec![a, b, c], commands)
    }

    /// Helper: build a diamond graph A -> B, A -> C, B -> D, C -> D.
    fn diamond_graph() -> (NarrativeGraph, Vec<IntentId>, HashMap<IntentId, String>) {
        let mut graph = NarrativeGraph::new();
        let a = IntentId::random();
        let b = IntentId::random();
        let c = IntentId::random();
        let d = IntentId::random();

        graph.insert_node(a, 1, 1.0);
        graph.insert_node(b, 2, 1.0);
        graph.insert_node(c, 2, 1.0);
        graph.insert_node(d, 3, 1.0);

        graph.add_edge(a, b, tala_core::RelationType::Causal, 1.0);
        graph.add_edge(a, c, tala_core::RelationType::Causal, 1.0);
        graph.add_edge(b, d, tala_core::RelationType::Causal, 1.0);
        graph.add_edge(c, d, tala_core::RelationType::Causal, 1.0);

        let mut commands = HashMap::new();
        commands.insert(a, "echo A".to_string());
        commands.insert(b, "echo B".to_string());
        commands.insert(c, "echo C".to_string());
        commands.insert(d, "echo D".to_string());

        (graph, vec![a, b, c, d], commands)
    }

    // -- Topological sort tests --

    #[test]
    fn test_linear_topo_sort() {
        let (graph, ids, commands) = linear_graph();
        let plan = build_plan(&graph, &ids, &commands).unwrap();

        assert_eq!(plan.len(), 3);
        // A must come before B, B before C.
        let positions: HashMap<IntentId, usize> = plan
            .iter()
            .enumerate()
            .map(|(i, s)| (s.intent_id, i))
            .collect();

        assert!(positions[&ids[0]] < positions[&ids[1]]);
        assert!(positions[&ids[1]] < positions[&ids[2]]);
    }

    #[test]
    fn test_diamond_topo_sort() {
        let (graph, ids, commands) = diamond_graph();
        let plan = build_plan(&graph, &ids, &commands).unwrap();

        assert_eq!(plan.len(), 4);
        let positions: HashMap<IntentId, usize> = plan
            .iter()
            .enumerate()
            .map(|(i, s)| (s.intent_id, i))
            .collect();

        // A before B, A before C, B before D, C before D.
        assert!(positions[&ids[0]] < positions[&ids[1]]);
        assert!(positions[&ids[0]] < positions[&ids[2]]);
        assert!(positions[&ids[1]] < positions[&ids[3]]);
        assert!(positions[&ids[2]] < positions[&ids[3]]);
    }

    #[test]
    fn test_cycle_detection() {
        let mut graph = NarrativeGraph::new();
        let a = IntentId::random();
        let b = IntentId::random();

        graph.insert_node(a, 1, 1.0);
        graph.insert_node(b, 2, 1.0);

        // A -> B -> A: cycle
        graph.add_edge(a, b, tala_core::RelationType::Causal, 1.0);
        graph.add_edge(b, a, tala_core::RelationType::Causal, 1.0);

        let mut commands = HashMap::new();
        commands.insert(a, "echo A".to_string());
        commands.insert(b, "echo B".to_string());

        let result = build_plan(&graph, &[a, b], &commands);
        assert!(matches!(result, Err(TalaError::CycleDetected)));
    }

    #[test]
    fn test_single_node_plan() {
        let mut graph = NarrativeGraph::new();
        let a = IntentId::random();
        graph.insert_node(a, 1, 1.0);

        let mut commands = HashMap::new();
        commands.insert(a, "echo hello".to_string());

        let plan = build_plan(&graph, &[a], &commands).unwrap();
        assert_eq!(plan.len(), 1);
        assert_eq!(plan[0].intent_id, a);
        assert_eq!(plan[0].command, "echo hello");
        assert!(plan[0].deps.is_empty());
    }

    // -- Transform tests --

    #[test]
    fn test_substitute_simple() {
        let mut vars = HashMap::new();
        vars.insert("HOME".to_string(), "/home/user".to_string());
        vars.insert("ENV".to_string(), "prod".to_string());

        let result = substitute_vars("cd ${HOME} && deploy --env=${ENV}", &vars);
        assert_eq!(result, "cd /home/user && deploy --env=prod");
    }

    #[test]
    fn test_substitute_unknown_var_left_intact() {
        let vars = HashMap::new();
        let result = substitute_vars("echo ${UNKNOWN}", &vars);
        assert_eq!(result, "echo ${UNKNOWN}");
    }

    #[test]
    fn test_substitute_no_vars() {
        let vars = HashMap::new();
        let result = substitute_vars("echo hello world", &vars);
        assert_eq!(result, "echo hello world");
    }

    #[test]
    fn test_substitute_adjacent_vars() {
        let mut vars = HashMap::new();
        vars.insert("A".to_string(), "foo".to_string());
        vars.insert("B".to_string(), "bar".to_string());

        let result = substitute_vars("${A}${B}", &vars);
        assert_eq!(result, "foobar");
    }

    #[test]
    fn test_substitute_unclosed_brace() {
        let vars = HashMap::new();
        let result = substitute_vars("echo ${UNCLOSED", &vars);
        assert_eq!(result, "echo ${UNCLOSED");
    }

    #[test]
    fn test_substitute_empty_var_name() {
        let mut vars = HashMap::new();
        vars.insert(String::new(), "empty".to_string());

        let result = substitute_vars("echo ${}", &vars);
        assert_eq!(result, "echo empty");
    }

    // -- Idempotency tests --

    #[test]
    fn test_filter_completed_removes_done_steps() {
        let a = IntentId::random();
        let b = IntentId::random();
        let c = IntentId::random();

        let steps = vec![
            ReplayStep { intent_id: a, command: "echo A".into(), deps: vec![] },
            ReplayStep { intent_id: b, command: "echo B".into(), deps: vec![a] },
            ReplayStep { intent_id: c, command: "echo C".into(), deps: vec![b] },
        ];

        let mut completed = HashSet::new();
        completed.insert(a);
        completed.insert(b);

        let remaining = filter_completed(steps, &completed);
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].intent_id, c);
    }

    #[test]
    fn test_filter_completed_none_done() {
        let a = IntentId::random();
        let steps = vec![
            ReplayStep { intent_id: a, command: "echo A".into(), deps: vec![] },
        ];
        let completed = HashSet::new();
        let remaining = filter_completed(steps, &completed);
        assert_eq!(remaining.len(), 1);
    }

    // -- Dry-run tests --

    #[test]
    fn test_dry_run_returns_transformed_plan() {
        let (graph, ids, mut commands) = linear_graph();
        // Override command B to include a variable.
        commands.insert(ids[1], "deploy --env=${ENV}".to_string());

        let mut vars = HashMap::new();
        vars.insert("ENV".to_string(), "staging".to_string());

        let config = ReplayConfig {
            vars,
            completed: HashSet::new(),
            dry_run: true,
        };

        let engine = ReplayEngine::new(config);
        let plan = engine.dry_run(&graph, &ids, &commands).unwrap();

        assert_eq!(plan.len(), 3);
        // Find step B and verify substitution.
        let step_b = plan.iter().find(|s| s.intent_id == ids[1]).unwrap();
        assert_eq!(step_b.command, "deploy --env=staging");
    }

    #[test]
    fn test_dry_run_skips_completed() {
        let (graph, ids, commands) = linear_graph();

        let mut completed = HashSet::new();
        completed.insert(ids[0]); // A is already done.

        let config = ReplayConfig {
            vars: HashMap::new(),
            completed,
            dry_run: true,
        };

        let engine = ReplayEngine::new(config);
        let plan = engine.dry_run(&graph, &ids, &commands).unwrap();

        assert_eq!(plan.len(), 2);
        assert!(plan.iter().all(|s| s.intent_id != ids[0]));
    }

    // -- Execute tests --

    #[test]
    fn test_execute_dry_run_returns_pending() {
        let (graph, ids, commands) = linear_graph();

        let config = ReplayConfig {
            vars: HashMap::new(),
            completed: HashSet::new(),
            dry_run: true,
        };

        let mut engine = ReplayEngine::new(config);
        let results = engine.execute(&graph, &ids, &commands).unwrap();

        assert_eq!(results.len(), 3);
        for r in &results {
            assert!(!r.skipped);
            assert_eq!(r.outcome.status, Status::Pending);
        }
    }

    #[test]
    fn test_execute_with_executor() {
        let (graph, ids, commands) = linear_graph();

        let config = ReplayConfig {
            vars: HashMap::new(),
            completed: HashSet::new(),
            dry_run: false,
        };

        let executor: Executor = Box::new(|_cmd: &str| Outcome {
            status: Status::Success,
            latency_ns: 100,
            exit_code: 0,
        });

        let mut engine = ReplayEngine::new(config).with_executor(executor);
        let results = engine.execute(&graph, &ids, &commands).unwrap();

        assert_eq!(results.len(), 3);
        for r in &results {
            assert!(!r.skipped);
            assert_eq!(r.outcome.status, Status::Success);
            assert_eq!(r.outcome.exit_code, 0);
        }
    }

    #[test]
    fn test_execute_skips_completed_steps() {
        let (graph, ids, commands) = linear_graph();

        let mut completed = HashSet::new();
        completed.insert(ids[0]);

        let config = ReplayConfig {
            vars: HashMap::new(),
            completed,
            dry_run: false,
        };

        let executor: Executor = Box::new(|_cmd: &str| {
            Outcome {
                status: Status::Success,
                latency_ns: 50,
                exit_code: 0,
            }
        });

        let mut engine = ReplayEngine::new(config).with_executor(executor);
        let results = engine.execute(&graph, &ids, &commands).unwrap();

        assert_eq!(results.len(), 3);
        // First step (A) should be skipped.
        let a_result = results.iter().find(|r| r.step.intent_id == ids[0]).unwrap();
        assert!(a_result.skipped);
        assert_eq!(a_result.outcome.latency_ns, 0);

        // B and C should not be skipped.
        let b_result = results.iter().find(|r| r.step.intent_id == ids[1]).unwrap();
        assert!(!b_result.skipped);

        let c_result = results.iter().find(|r| r.step.intent_id == ids[2]).unwrap();
        assert!(!c_result.skipped);
    }

    #[test]
    fn test_execute_applies_transforms() {
        let mut graph = NarrativeGraph::new();
        let a = IntentId::random();
        graph.insert_node(a, 1, 1.0);

        let mut commands = HashMap::new();
        commands.insert(a, "ssh ${HOST} 'restart ${SVC}'".to_string());

        let mut vars = HashMap::new();
        vars.insert("HOST".to_string(), "prod-01".to_string());
        vars.insert("SVC".to_string(), "nginx".to_string());

        let config = ReplayConfig {
            vars,
            completed: HashSet::new(),
            dry_run: false,
        };

        let captured_cmds: std::sync::Arc<std::sync::Mutex<Vec<String>>> =
            std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let cmds_clone = captured_cmds.clone();

        let executor: Executor = Box::new(move |cmd: &str| {
            cmds_clone.lock().unwrap().push(cmd.to_string());
            Outcome {
                status: Status::Success,
                latency_ns: 10,
                exit_code: 0,
            }
        });

        let mut engine = ReplayEngine::new(config).with_executor(executor);
        let results = engine.execute(&graph, &[a], &commands).unwrap();

        assert_eq!(results.len(), 1);
        let executed_cmds = captured_cmds.lock().unwrap();
        assert_eq!(executed_cmds[0], "ssh prod-01 'restart nginx'");
    }
}
