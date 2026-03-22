//! TALA Net — Distributed networking layer for the intent-native narrative execution layer.
//!
//! Provides core distributed types (node identity, partitioning, membership),
//! a TLV message codec, and an in-process transport for testing without a real
//! network. Real QUIC transport will be added in a future phase (see spec-04).

use std::collections::HashMap;
use std::sync::{mpsc, Arc, Mutex};

use tala_core::TalaError;

// ---------------------------------------------------------------------------
// Identity types
// ---------------------------------------------------------------------------

/// Unique identifier for a node in the cluster.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NodeId(pub u64);

/// Alias for [`NodeId`] used in transport contexts.
pub type PeerId = NodeId;

/// Identifies a partition of the intent graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PartitionId(pub u32);

// ---------------------------------------------------------------------------
// Messages
// ---------------------------------------------------------------------------

/// Framed messages exchanged between nodes.
#[derive(Clone, Debug, PartialEq)]
pub enum Message {
    Ping { from: NodeId, seq: u64 },
    Pong { from: NodeId, seq: u64 },
    IntentForward { partition: PartitionId, payload: Vec<u8> },
    SegmentSync { partition: PartitionId, segment_data: Vec<u8> },
    MembershipUpdate { members: Vec<NodeId>, version: u64 },
    PartitionTableUpdate { assignments: Vec<PartitionAssignment>, version: u64 },
}

// ---------------------------------------------------------------------------
// Partition Table
// ---------------------------------------------------------------------------

/// Maps a partition to its owner and replicas.
#[derive(Clone, Debug, PartialEq)]
pub struct PartitionAssignment {
    pub partition_id: PartitionId,
    pub owner: NodeId,
    pub replicas: Vec<NodeId>,
}

/// Cluster-wide partition routing table.
#[derive(Clone, Debug)]
pub struct PartitionTable {
    pub version: u64,
    pub assignments: Vec<PartitionAssignment>,
}

impl PartitionTable {
    /// Return the owner of a given partition, if assigned.
    pub fn owner_of(&self, partition: PartitionId) -> Option<NodeId> {
        self.assignments
            .iter()
            .find(|a| a.partition_id == partition)
            .map(|a| a.owner)
    }

    /// Return all partitions owned by (or replicated on) a given node.
    pub fn partitions_for(&self, node: NodeId) -> Vec<PartitionId> {
        self.assignments
            .iter()
            .filter(|a| a.owner == node || a.replicas.contains(&node))
            .map(|a| a.partition_id)
            .collect()
    }

    /// Consistent-hash an intent id (16 raw UUID bytes) to a partition.
    ///
    /// Uses FNV-1a over the id bytes, then reduces modulo `num_partitions`.
    /// Returns `PartitionId(0)` if `num_partitions` is zero.
    pub fn partition_for_intent(id_bytes: &[u8; 16], num_partitions: u32) -> PartitionId {
        if num_partitions == 0 {
            return PartitionId(0);
        }
        let hash = fnv1a(id_bytes);
        PartitionId((hash % num_partitions as u64) as u32)
    }
}

// ---------------------------------------------------------------------------
// Membership
// ---------------------------------------------------------------------------

/// Liveness state of a cluster member (SWIM protocol model).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemberState {
    Alive,
    Suspect,
    Dead,
}

/// Cluster membership list with versioned state.
#[derive(Clone, Debug)]
pub struct MembershipList {
    pub members: Vec<(NodeId, MemberState)>,
    pub version: u64,
}

impl MembershipList {
    /// Create an empty membership list.
    pub fn new() -> Self {
        Self {
            members: Vec::new(),
            version: 0,
        }
    }

    /// Add a member as `Alive`. Bumps the version. If the node already
    /// exists, its state is set back to `Alive`.
    pub fn add_member(&mut self, node: NodeId) {
        if let Some(entry) = self.members.iter_mut().find(|(id, _)| *id == node) {
            entry.1 = MemberState::Alive;
        } else {
            self.members.push((node, MemberState::Alive));
        }
        self.version += 1;
    }

    /// Transition a member to `Suspect`. Bumps the version.
    /// No-op if the node is not present.
    pub fn mark_suspect(&mut self, node: NodeId) {
        if let Some(entry) = self.members.iter_mut().find(|(id, _)| *id == node) {
            entry.1 = MemberState::Suspect;
            self.version += 1;
        }
    }

    /// Transition a member to `Dead`. Bumps the version.
    /// No-op if the node is not present.
    pub fn mark_dead(&mut self, node: NodeId) {
        if let Some(entry) = self.members.iter_mut().find(|(id, _)| *id == node) {
            entry.1 = MemberState::Dead;
            self.version += 1;
        }
    }

    /// Return the set of members whose state is `Alive`.
    pub fn alive_members(&self) -> Vec<NodeId> {
        self.members
            .iter()
            .filter(|(_, state)| *state == MemberState::Alive)
            .map(|(id, _)| *id)
            .collect()
    }
}

impl Default for MembershipList {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Codec — TLV message serialization
// ---------------------------------------------------------------------------

// Message type tags (first byte of encoded frame).
const TAG_PING: u8 = 0x01;
const TAG_PONG: u8 = 0x02;
const TAG_INTENT_FORWARD: u8 = 0x03;
const TAG_SEGMENT_SYNC: u8 = 0x04;
const TAG_MEMBERSHIP_UPDATE: u8 = 0x05;
const TAG_PARTITION_TABLE_UPDATE: u8 = 0x06;

/// Encode a [`Message`] into a byte buffer using TLV framing.
///
/// Wire layout: `[tag:1][length:4 LE][payload:length]`
pub fn encode(msg: &Message) -> Vec<u8> {
    let mut payload = Vec::new();
    let tag = match msg {
        Message::Ping { from, seq } => {
            payload.extend_from_slice(&from.0.to_le_bytes());
            payload.extend_from_slice(&seq.to_le_bytes());
            TAG_PING
        }
        Message::Pong { from, seq } => {
            payload.extend_from_slice(&from.0.to_le_bytes());
            payload.extend_from_slice(&seq.to_le_bytes());
            TAG_PONG
        }
        Message::IntentForward { partition, payload: data } => {
            payload.extend_from_slice(&partition.0.to_le_bytes());
            payload.extend_from_slice(data);
            TAG_INTENT_FORWARD
        }
        Message::SegmentSync { partition, segment_data } => {
            payload.extend_from_slice(&partition.0.to_le_bytes());
            payload.extend_from_slice(segment_data);
            TAG_SEGMENT_SYNC
        }
        Message::MembershipUpdate { members, version } => {
            payload.extend_from_slice(&(*version).to_le_bytes());
            payload.extend_from_slice(&(members.len() as u32).to_le_bytes());
            for m in members {
                payload.extend_from_slice(&m.0.to_le_bytes());
            }
            TAG_MEMBERSHIP_UPDATE
        }
        Message::PartitionTableUpdate { assignments, version } => {
            payload.extend_from_slice(&(*version).to_le_bytes());
            payload.extend_from_slice(&(assignments.len() as u32).to_le_bytes());
            for a in assignments {
                payload.extend_from_slice(&a.partition_id.0.to_le_bytes());
                payload.extend_from_slice(&a.owner.0.to_le_bytes());
                payload.extend_from_slice(&(a.replicas.len() as u32).to_le_bytes());
                for r in &a.replicas {
                    payload.extend_from_slice(&r.0.to_le_bytes());
                }
            }
            TAG_PARTITION_TABLE_UPDATE
        }
    };

    let mut buf = Vec::with_capacity(1 + 4 + payload.len());
    buf.push(tag);
    buf.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    buf.extend_from_slice(&payload);
    buf
}

/// Decode a [`Message`] from a TLV-encoded byte buffer.
pub fn decode(data: &[u8]) -> Result<Message, TalaError> {
    if data.len() < 5 {
        return Err(TalaError::SegmentCorrupted(
            "message too short for TLV header".into(),
        ));
    }

    let tag = data[0];
    let length = u32::from_le_bytes(
        data[1..5]
            .try_into()
            .map_err(|_| TalaError::SegmentCorrupted("invalid length bytes".into()))?,
    ) as usize;

    if data.len() < 5 + length {
        return Err(TalaError::SegmentCorrupted(format!(
            "message truncated: expected {} payload bytes, got {}",
            length,
            data.len() - 5
        )));
    }

    let payload = &data[5..5 + length];

    match tag {
        TAG_PING => {
            check_len(payload, 16, "Ping")?;
            Ok(Message::Ping {
                from: NodeId(u64::from_le_bytes(payload[0..8].try_into().map_err(conv_err)?)),
                seq: u64::from_le_bytes(payload[8..16].try_into().map_err(conv_err)?),
            })
        }
        TAG_PONG => {
            check_len(payload, 16, "Pong")?;
            Ok(Message::Pong {
                from: NodeId(u64::from_le_bytes(payload[0..8].try_into().map_err(conv_err)?)),
                seq: u64::from_le_bytes(payload[8..16].try_into().map_err(conv_err)?),
            })
        }
        TAG_INTENT_FORWARD => {
            check_min_len(payload, 4, "IntentForward")?;
            let partition = PartitionId(u32::from_le_bytes(
                payload[0..4].try_into().map_err(conv_err)?,
            ));
            let body = payload[4..].to_vec();
            Ok(Message::IntentForward {
                partition,
                payload: body,
            })
        }
        TAG_SEGMENT_SYNC => {
            check_min_len(payload, 4, "SegmentSync")?;
            let partition = PartitionId(u32::from_le_bytes(
                payload[0..4].try_into().map_err(conv_err)?,
            ));
            let body = payload[4..].to_vec();
            Ok(Message::SegmentSync {
                partition,
                segment_data: body,
            })
        }
        TAG_MEMBERSHIP_UPDATE => {
            check_min_len(payload, 12, "MembershipUpdate")?;
            let version = u64::from_le_bytes(payload[0..8].try_into().map_err(conv_err)?);
            let count = u32::from_le_bytes(payload[8..12].try_into().map_err(conv_err)?) as usize;
            check_len(payload, 12 + count * 8, "MembershipUpdate members")?;
            let mut members = Vec::with_capacity(count);
            for i in 0..count {
                let off = 12 + i * 8;
                members.push(NodeId(u64::from_le_bytes(
                    payload[off..off + 8].try_into().map_err(conv_err)?,
                )));
            }
            Ok(Message::MembershipUpdate { members, version })
        }
        TAG_PARTITION_TABLE_UPDATE => {
            check_min_len(payload, 12, "PartitionTableUpdate")?;
            let version = u64::from_le_bytes(payload[0..8].try_into().map_err(conv_err)?);
            let count = u32::from_le_bytes(payload[8..12].try_into().map_err(conv_err)?) as usize;
            let mut assignments = Vec::with_capacity(count);
            let mut cursor = 12;
            for _ in 0..count {
                check_min_len(payload, cursor + 16, "PartitionTableUpdate assignment")?;
                let partition_id = PartitionId(u32::from_le_bytes(
                    payload[cursor..cursor + 4].try_into().map_err(conv_err)?,
                ));
                cursor += 4;
                let owner = NodeId(u64::from_le_bytes(
                    payload[cursor..cursor + 8].try_into().map_err(conv_err)?,
                ));
                cursor += 8;
                let replica_count = u32::from_le_bytes(
                    payload[cursor..cursor + 4].try_into().map_err(conv_err)?,
                ) as usize;
                cursor += 4;
                check_min_len(
                    payload,
                    cursor + replica_count * 8,
                    "PartitionTableUpdate replicas",
                )?;
                let mut replicas = Vec::with_capacity(replica_count);
                for _ in 0..replica_count {
                    replicas.push(NodeId(u64::from_le_bytes(
                        payload[cursor..cursor + 8].try_into().map_err(conv_err)?,
                    )));
                    cursor += 8;
                }
                assignments.push(PartitionAssignment {
                    partition_id,
                    owner,
                    replicas,
                });
            }
            Ok(Message::PartitionTableUpdate {
                assignments,
                version,
            })
        }
        unknown => Err(TalaError::SegmentCorrupted(format!(
            "unknown message tag: 0x{:02x}",
            unknown
        ))),
    }
}

fn check_len(payload: &[u8], expected: usize, context: &str) -> Result<(), TalaError> {
    if payload.len() < expected {
        return Err(TalaError::SegmentCorrupted(format!(
            "{}: expected at least {} bytes, got {}",
            context,
            expected,
            payload.len()
        )));
    }
    Ok(())
}

fn check_min_len(payload: &[u8], min: usize, context: &str) -> Result<(), TalaError> {
    check_len(payload, min, context)
}

fn conv_err(_: std::array::TryFromSliceError) -> TalaError {
    TalaError::SegmentCorrupted("byte conversion failed".into())
}

// ---------------------------------------------------------------------------
// FNV-1a hash (matches tala-wire's internal hash for consistency)
// ---------------------------------------------------------------------------

fn fnv1a(data: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &byte in data {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

// ---------------------------------------------------------------------------
// In-Process Transport — channel-based transport for testing
// ---------------------------------------------------------------------------

/// A simulated network of in-process transports connected via mpsc channels.
pub struct InProcessNetwork {
    senders: Arc<Mutex<HashMap<u64, mpsc::Sender<(NodeId, Message)>>>>,
}

impl InProcessNetwork {
    /// Create a new empty in-process network.
    pub fn new() -> Self {
        Self {
            senders: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Register a node and return its transport handle.
    pub fn add_node(&self, id: NodeId) -> InProcessTransport {
        let (tx, rx) = mpsc::channel();
        {
            let mut senders = self.senders.lock().unwrap_or_else(|e| e.into_inner());
            senders.insert(id.0, tx);
        }
        InProcessTransport {
            id,
            senders: Arc::clone(&self.senders),
            rx,
        }
    }
}

impl Default for InProcessNetwork {
    fn default() -> Self {
        Self::new()
    }
}

/// A single node's view of the in-process network.
pub struct InProcessTransport {
    id: NodeId,
    senders: Arc<Mutex<HashMap<u64, mpsc::Sender<(NodeId, Message)>>>>,
    rx: mpsc::Receiver<(NodeId, Message)>,
}

impl InProcessTransport {
    /// Send a message to a specific peer. Silently drops if the peer is not
    /// registered (mirrors real-world UDP-like fire-and-forget semantics).
    pub fn send(&self, to: NodeId, msg: Message) {
        let senders = self.senders.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(tx) = senders.get(&to.0) {
            // Ignore send errors — the receiver may have been dropped.
            let _ = tx.send((self.id, msg));
        }
    }

    /// Try to receive the next message. Returns `None` if no message is
    /// available (non-blocking).
    pub fn recv(&self) -> Option<(NodeId, Message)> {
        self.rx.try_recv().ok()
    }

    /// Broadcast a message to all registered peers except self.
    pub fn broadcast(&self, msg: Message) {
        let senders = self.senders.lock().unwrap_or_else(|e| e.into_inner());
        for (&peer_id, tx) in senders.iter() {
            if peer_id != self.id.0 {
                let _ = tx.send((self.id, msg.clone()));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Partition Table --

    #[test]
    fn partition_table_owner_of() {
        let table = PartitionTable {
            version: 1,
            assignments: vec![
                PartitionAssignment {
                    partition_id: PartitionId(0),
                    owner: NodeId(10),
                    replicas: vec![NodeId(11), NodeId(12)],
                },
                PartitionAssignment {
                    partition_id: PartitionId(1),
                    owner: NodeId(11),
                    replicas: vec![NodeId(10)],
                },
            ],
        };

        assert_eq!(table.owner_of(PartitionId(0)), Some(NodeId(10)));
        assert_eq!(table.owner_of(PartitionId(1)), Some(NodeId(11)));
        assert_eq!(table.owner_of(PartitionId(99)), None);
    }

    #[test]
    fn partition_table_partitions_for() {
        let table = PartitionTable {
            version: 1,
            assignments: vec![
                PartitionAssignment {
                    partition_id: PartitionId(0),
                    owner: NodeId(10),
                    replicas: vec![NodeId(11)],
                },
                PartitionAssignment {
                    partition_id: PartitionId(1),
                    owner: NodeId(11),
                    replicas: vec![NodeId(10)],
                },
                PartitionAssignment {
                    partition_id: PartitionId(2),
                    owner: NodeId(12),
                    replicas: vec![],
                },
            ],
        };

        let mut p10 = table.partitions_for(NodeId(10));
        p10.sort_by_key(|p| p.0);
        assert_eq!(p10, vec![PartitionId(0), PartitionId(1)]);

        let p12 = table.partitions_for(NodeId(12));
        assert_eq!(p12, vec![PartitionId(2)]);

        let p99 = table.partitions_for(NodeId(99));
        assert!(p99.is_empty());
    }

    #[test]
    fn partition_for_intent_deterministic() {
        let id_a: [u8; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let id_b: [u8; 16] = [16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1];
        let num = 64;

        let pa1 = PartitionTable::partition_for_intent(&id_a, num);
        let pa2 = PartitionTable::partition_for_intent(&id_a, num);
        assert_eq!(pa1, pa2, "same input must yield same partition");

        // Both must be in range.
        assert!(pa1.0 < num);

        let pb = PartitionTable::partition_for_intent(&id_b, num);
        assert!(pb.0 < num);
        // Different inputs *may* differ — not guaranteed, but extremely likely
        // with these particular values.
    }

    #[test]
    fn partition_for_intent_zero_partitions() {
        let id: [u8; 16] = [0; 16];
        let p = PartitionTable::partition_for_intent(&id, 0);
        assert_eq!(p, PartitionId(0));
    }

    // -- Membership --

    #[test]
    fn membership_lifecycle() {
        let mut ml = MembershipList::new();
        assert!(ml.alive_members().is_empty());
        assert_eq!(ml.version, 0);

        ml.add_member(NodeId(1));
        ml.add_member(NodeId(2));
        ml.add_member(NodeId(3));
        assert_eq!(ml.alive_members().len(), 3);
        assert_eq!(ml.version, 3);

        ml.mark_suspect(NodeId(2));
        assert_eq!(ml.version, 4);
        assert_eq!(ml.alive_members(), vec![NodeId(1), NodeId(3)]);

        ml.mark_dead(NodeId(2));
        assert_eq!(ml.version, 5);
        assert_eq!(ml.alive_members(), vec![NodeId(1), NodeId(3)]);

        // Re-add a dead node → back to Alive.
        ml.add_member(NodeId(2));
        assert_eq!(ml.version, 6);
        assert_eq!(ml.alive_members().len(), 3);
    }

    #[test]
    fn membership_no_op_for_unknown_node() {
        let mut ml = MembershipList::new();
        ml.add_member(NodeId(1));
        let v = ml.version;
        ml.mark_suspect(NodeId(99));
        assert_eq!(ml.version, v, "version should not bump for unknown node");
        ml.mark_dead(NodeId(99));
        assert_eq!(ml.version, v);
    }

    // -- Codec roundtrip --

    fn roundtrip(msg: &Message) {
        let bytes = encode(msg);
        let decoded = decode(&bytes).expect("decode must succeed");
        assert_eq!(&decoded, msg);
    }

    #[test]
    fn codec_ping_pong() {
        roundtrip(&Message::Ping {
            from: NodeId(42),
            seq: 1,
        });
        roundtrip(&Message::Pong {
            from: NodeId(7),
            seq: 9999,
        });
    }

    #[test]
    fn codec_intent_forward() {
        roundtrip(&Message::IntentForward {
            partition: PartitionId(3),
            payload: vec![0xDE, 0xAD, 0xBE, 0xEF],
        });
        // Empty payload.
        roundtrip(&Message::IntentForward {
            partition: PartitionId(0),
            payload: vec![],
        });
    }

    #[test]
    fn codec_segment_sync() {
        roundtrip(&Message::SegmentSync {
            partition: PartitionId(10),
            segment_data: vec![1, 2, 3, 4, 5],
        });
    }

    #[test]
    fn codec_membership_update() {
        roundtrip(&Message::MembershipUpdate {
            members: vec![NodeId(1), NodeId(2), NodeId(3)],
            version: 42,
        });
        // Empty members.
        roundtrip(&Message::MembershipUpdate {
            members: vec![],
            version: 0,
        });
    }

    #[test]
    fn codec_partition_table_update() {
        roundtrip(&Message::PartitionTableUpdate {
            assignments: vec![
                PartitionAssignment {
                    partition_id: PartitionId(0),
                    owner: NodeId(10),
                    replicas: vec![NodeId(11), NodeId(12)],
                },
                PartitionAssignment {
                    partition_id: PartitionId(1),
                    owner: NodeId(11),
                    replicas: vec![],
                },
            ],
            version: 7,
        });
    }

    #[test]
    fn codec_rejects_truncated() {
        assert!(decode(&[]).is_err());
        assert!(decode(&[0x01, 0x00]).is_err());
        // Valid header but payload too short.
        let mut buf = encode(&Message::Ping {
            from: NodeId(1),
            seq: 1,
        });
        buf.truncate(buf.len() - 4);
        assert!(decode(&buf).is_err());
    }

    #[test]
    fn codec_rejects_unknown_tag() {
        let data = [0xFF, 0x00, 0x00, 0x00, 0x00];
        assert!(decode(&data).is_err());
    }

    // -- In-process transport --

    #[test]
    fn transport_send_recv() {
        let net = InProcessNetwork::new();
        let t1 = net.add_node(NodeId(1));
        let t2 = net.add_node(NodeId(2));

        let msg = Message::Ping {
            from: NodeId(1),
            seq: 100,
        };
        t1.send(NodeId(2), msg.clone());

        let (from, received) = t2.recv().expect("should receive message");
        assert_eq!(from, NodeId(1));
        assert_eq!(received, msg);

        // No more messages.
        assert!(t2.recv().is_none());
    }

    #[test]
    fn transport_send_to_unknown_peer() {
        let net = InProcessNetwork::new();
        let t1 = net.add_node(NodeId(1));

        // Sending to non-existent peer should not panic.
        t1.send(
            NodeId(99),
            Message::Ping {
                from: NodeId(1),
                seq: 0,
            },
        );
    }

    #[test]
    fn transport_broadcast() {
        let net = InProcessNetwork::new();
        let t1 = net.add_node(NodeId(1));
        let t2 = net.add_node(NodeId(2));
        let t3 = net.add_node(NodeId(3));

        let msg = Message::Pong {
            from: NodeId(1),
            seq: 42,
        };
        t1.broadcast(msg.clone());

        // Both t2 and t3 should receive it.
        let (from2, m2) = t2.recv().expect("t2 should receive broadcast");
        assert_eq!(from2, NodeId(1));
        assert_eq!(m2, msg);

        let (from3, m3) = t3.recv().expect("t3 should receive broadcast");
        assert_eq!(from3, NodeId(1));
        assert_eq!(m3, msg);

        // t1 should NOT receive its own broadcast.
        assert!(t1.recv().is_none());
    }

    #[test]
    fn transport_multiple_messages_ordered() {
        let net = InProcessNetwork::new();
        let t1 = net.add_node(NodeId(1));
        let t2 = net.add_node(NodeId(2));

        for seq in 0..10 {
            t1.send(
                NodeId(2),
                Message::Ping {
                    from: NodeId(1),
                    seq,
                },
            );
        }

        for expected_seq in 0..10 {
            let (_, msg) = t2.recv().expect("should receive message");
            match msg {
                Message::Ping { seq, .. } => assert_eq!(seq, expected_seq),
                other => panic!("unexpected message: {:?}", other),
            }
        }
        assert!(t2.recv().is_none());
    }
}
