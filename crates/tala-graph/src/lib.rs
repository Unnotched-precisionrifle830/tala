//! TALA Narrative Graph — in-memory graph with BFS/DFS, edge formation, and narrative extraction.

use std::collections::{HashMap, HashSet, VecDeque};
use tala_core::{IntentId, RelationType};

struct NodeData {
    timestamp: u64,
    confidence: f32,
}

/// In-memory narrative graph backed by adjacency lists.
pub struct NarrativeGraph {
    forward: HashMap<IntentId, Vec<(IntentId, RelationType, f32)>>,
    backward: HashMap<IntentId, Vec<(IntentId, RelationType, f32)>>,
    nodes: HashMap<IntentId, NodeData>,
}

impl NarrativeGraph {
    pub fn new() -> Self {
        Self {
            forward: HashMap::new(),
            backward: HashMap::new(),
            nodes: HashMap::new(),
        }
    }

    pub fn insert_node(&mut self, id: IntentId, timestamp: u64, confidence: f32) {
        self.nodes.insert(id, NodeData { timestamp, confidence });
        self.forward.entry(id).or_default();
        self.backward.entry(id).or_default();
    }

    pub fn add_edge(&mut self, from: IntentId, to: IntentId, relation: RelationType, weight: f32) {
        self.forward
            .entry(from)
            .or_default()
            .push((to, relation, weight));
        self.backward
            .entry(to)
            .or_default()
            .push((from, relation, weight));
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn edge_count(&self) -> usize {
        self.forward.values().map(|v| v.len()).sum()
    }

    /// Form edges between `new_node` and existing nodes.
    /// `similarities`: Vec of (existing_node_id, similarity_score).
    /// Connects to top-K by score with Causal relation.
    pub fn form_edges(
        &mut self,
        new_node: IntentId,
        similarities: &mut [(IntentId, f32)],
        k: usize,
    ) {
        similarities.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        for &(existing, score) in similarities.iter().take(k) {
            self.add_edge(existing, new_node, RelationType::Causal, score);
        }
    }

    /// BFS forward from `start`, up to `max_depth` hops.
    pub fn bfs_forward(&self, start: IntentId, max_depth: usize) -> Vec<IntentId> {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        let mut result = Vec::new();

        visited.insert(start);
        queue.push_back((start, 0usize));

        while let Some((node, depth)) = queue.pop_front() {
            result.push(node);
            if depth >= max_depth {
                continue;
            }
            if let Some(neighbors) = self.forward.get(&node) {
                for &(neighbor, _, _) in neighbors {
                    if visited.insert(neighbor) {
                        queue.push_back((neighbor, depth + 1));
                    }
                }
            }
        }
        result
    }

    /// BFS backward from `start` (root-cause analysis).
    pub fn bfs_backward(&self, start: IntentId, max_depth: usize) -> Vec<IntentId> {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        let mut result = Vec::new();

        visited.insert(start);
        queue.push_back((start, 0usize));

        while let Some((node, depth)) = queue.pop_front() {
            result.push(node);
            if depth >= max_depth {
                continue;
            }
            if let Some(neighbors) = self.backward.get(&node) {
                for &(neighbor, _, _) in neighbors {
                    if visited.insert(neighbor) {
                        queue.push_back((neighbor, depth + 1));
                    }
                }
            }
        }
        result
    }

    /// Extract narrative: bidirectional BFS from `root`.
    /// Returns (visited_nodes, edges).
    pub fn extract_narrative(
        &self,
        root: IntentId,
        max_depth: usize,
    ) -> (Vec<IntentId>, Vec<(IntentId, IntentId)>) {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        let mut edges = Vec::new();

        visited.insert(root);
        queue.push_back((root, 0usize));

        while let Some((node, depth)) = queue.pop_front() {
            if depth >= max_depth {
                continue;
            }
            if let Some(fwd) = self.forward.get(&node) {
                for &(neighbor, _, _) in fwd {
                    edges.push((node, neighbor));
                    if visited.insert(neighbor) {
                        queue.push_back((neighbor, depth + 1));
                    }
                }
            }
            if let Some(bwd) = self.backward.get(&node) {
                for &(neighbor, _, _) in bwd {
                    edges.push((neighbor, node));
                    if visited.insert(neighbor) {
                        queue.push_back((neighbor, depth + 1));
                    }
                }
            }
        }

        (visited.into_iter().collect(), edges)
    }

    /// Returns all node IDs in the graph.
    pub fn node_ids(&self) -> Vec<IntentId> {
        self.nodes.keys().copied().collect()
    }

    /// Returns the forward neighbors (successors) of a node.
    /// Each entry is (neighbor_id, relation_type, weight).
    pub fn successors(&self, id: IntentId) -> &[(IntentId, RelationType, f32)] {
        self.forward.get(&id).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Returns the backward neighbors (predecessors) of a node.
    /// Each entry is (neighbor_id, relation_type, weight).
    pub fn predecessors(&self, id: IntentId) -> &[(IntentId, RelationType, f32)] {
        self.backward.get(&id).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Returns true if the graph contains a node with the given ID.
    pub fn contains_node(&self, id: IntentId) -> bool {
        self.nodes.contains_key(&id)
    }
}

impl Default for NarrativeGraph {
    fn default() -> Self {
        Self::new()
    }
}
