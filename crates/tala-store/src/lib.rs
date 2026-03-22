//! TALA Storage Engine — WAL, hot buffer, segment flushing, and semantic query.

use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Instant;

use tala_core::{Intent, IntentId, IntentStore, Outcome, Status, TalaError, TimeRange};
use tala_embed::{cosine_similarity, HnswIndex};
use tala_wire::SegmentWriter;

// ===========================================================================
// Lock and Storage Metrics
// ===========================================================================

/// Per-lock statistics tracked via atomics. All times in nanoseconds.
pub struct LockStats {
    /// Total number of lock acquisitions.
    pub acquisitions: AtomicU64,
    /// Number of acquisitions where wait exceeded 1us (contention proxy).
    pub contentions: AtomicU64,
    /// Cumulative time spent waiting to acquire the lock (nanoseconds).
    pub total_wait_ns: AtomicU64,
    /// Cumulative time spent holding the lock (nanoseconds).
    pub total_hold_ns: AtomicU64,
    /// Worst-case wait time observed (nanoseconds).
    pub max_wait_ns: AtomicU64,
    /// Worst-case hold time observed (nanoseconds).
    pub max_hold_ns: AtomicU64,
}

impl LockStats {
    pub fn new() -> Self {
        Self {
            acquisitions: AtomicU64::new(0),
            contentions: AtomicU64::new(0),
            total_wait_ns: AtomicU64::new(0),
            total_hold_ns: AtomicU64::new(0),
            max_wait_ns: AtomicU64::new(0),
            max_hold_ns: AtomicU64::new(0),
        }
    }

    /// Record the wait time for a lock acquisition.
    pub fn record_acquisition(&self, wait_ns: u64) {
        self.acquisitions.fetch_add(1, Relaxed);
        self.total_wait_ns.fetch_add(wait_ns, Relaxed);
        // Update max_wait_ns using CAS loop
        let mut current = self.max_wait_ns.load(Relaxed);
        while wait_ns > current {
            match self.max_wait_ns.compare_exchange_weak(current, wait_ns, Relaxed, Relaxed) {
                Ok(_) => break,
                Err(c) => current = c,
            }
        }
        if wait_ns > 1_000 {
            // >1us counts as contention
            self.contentions.fetch_add(1, Relaxed);
        }
    }

    /// Record the hold duration when a lock is released.
    pub fn record_release(&self, hold_ns: u64) {
        self.total_hold_ns.fetch_add(hold_ns, Relaxed);
        let mut current = self.max_hold_ns.load(Relaxed);
        while hold_ns > current {
            match self.max_hold_ns.compare_exchange_weak(current, hold_ns, Relaxed, Relaxed) {
                Ok(_) => break,
                Err(c) => current = c,
            }
        }
    }
}

impl Default for LockStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Comprehensive storage engine metrics.
pub struct StorageMetrics {
    // --- Per-lock stats ---
    pub intents_lock: LockStats,
    pub hnsw_lock: LockStats,
    pub index_map_lock: LockStats,
    pub wal_lock: LockStats,
    pub hot_lock: LockStats,

    // --- Pipeline sub-operation timing (cumulative nanoseconds) ---
    pub wal_append_ns: AtomicU64,
    pub wal_append_count: AtomicU64,
    pub hnsw_insert_ns: AtomicU64,
    pub hnsw_insert_count: AtomicU64,
    pub hot_push_ns: AtomicU64,
    pub hot_push_count: AtomicU64,
    pub segment_flush_ns: AtomicU64,
    pub segment_flush_count: AtomicU64,
    pub edge_search_ns: AtomicU64,
    pub edge_search_count: AtomicU64,

    // --- HNSW internals ---
    pub hnsw_search_visited: AtomicU64,
    pub hnsw_search_count: AtomicU64,

    // --- Store state ---
    pub hot_buffer_len: AtomicU64,
    pub hot_buffer_capacity: AtomicU64,
    pub wal_entry_count: AtomicU64,
    pub total_bytes_flushed: AtomicU64,
    pub segments_flushed_count: AtomicU64,
}

impl StorageMetrics {
    pub fn new() -> Self {
        Self {
            intents_lock: LockStats::new(),
            hnsw_lock: LockStats::new(),
            index_map_lock: LockStats::new(),
            wal_lock: LockStats::new(),
            hot_lock: LockStats::new(),

            wal_append_ns: AtomicU64::new(0),
            wal_append_count: AtomicU64::new(0),
            hnsw_insert_ns: AtomicU64::new(0),
            hnsw_insert_count: AtomicU64::new(0),
            hot_push_ns: AtomicU64::new(0),
            hot_push_count: AtomicU64::new(0),
            segment_flush_ns: AtomicU64::new(0),
            segment_flush_count: AtomicU64::new(0),
            edge_search_ns: AtomicU64::new(0),
            edge_search_count: AtomicU64::new(0),

            hnsw_search_visited: AtomicU64::new(0),
            hnsw_search_count: AtomicU64::new(0),

            hot_buffer_len: AtomicU64::new(0),
            hot_buffer_capacity: AtomicU64::new(0),
            wal_entry_count: AtomicU64::new(0),
            total_bytes_flushed: AtomicU64::new(0),
            segments_flushed_count: AtomicU64::new(0),
        }
    }
}

impl Default for StorageMetrics {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// Write-Ahead Log
// ===========================================================================

pub struct Wal {
    writer: BufWriter<File>,
    path: PathBuf,
    entry_count: u64,
}

impl Wal {
    pub fn create(path: impl AsRef<Path>) -> io::Result<Self> {
        let file = File::create(path.as_ref())?;
        Ok(Self {
            writer: BufWriter::with_capacity(64 * 1024, file),
            path: path.as_ref().to_path_buf(),
            entry_count: 0,
        })
    }

    /// Append a single intent entry to the WAL.
    /// Format: [4B payload_len][16B id][8B ts][4B embed_len][embed..][4B cmd_len][cmd..]
    pub fn append(
        &mut self,
        id: &[u8; 16],
        timestamp: u64,
        embedding: &[f32],
        raw_command: &str,
    ) -> io::Result<()> {
        let embed_bytes_len = embedding.len() * 4;
        let cmd_bytes_len = raw_command.len();
        let payload_len: u32 = (16 + 8 + 4 + embed_bytes_len + 4 + cmd_bytes_len) as u32;

        self.writer.write_all(&payload_len.to_le_bytes())?;
        self.writer.write_all(id)?;
        self.writer.write_all(&timestamp.to_le_bytes())?;
        self.writer
            .write_all(&(embedding.len() as u32).to_le_bytes())?;
        // Safe: f32 has no invalid bit patterns
        let embed_raw =
            unsafe { std::slice::from_raw_parts(embedding.as_ptr() as *const u8, embed_bytes_len) };
        self.writer.write_all(embed_raw)?;
        self.writer
            .write_all(&(cmd_bytes_len as u32).to_le_bytes())?;
        self.writer.write_all(raw_command.as_bytes())?;

        self.entry_count += 1;
        Ok(())
    }

    pub fn sync(&mut self) -> io::Result<()> {
        self.writer.flush()
    }

    pub fn entry_count(&self) -> u64 {
        self.entry_count
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

// ===========================================================================
// WAL Reader
// ===========================================================================

pub struct WalEntry {
    pub id: [u8; 16],
    pub timestamp: u64,
    pub embedding: Vec<f32>,
    pub raw_command: String,
}

pub fn replay_wal(path: impl AsRef<Path>) -> io::Result<Vec<WalEntry>> {
    let mut file = File::open(path)?;
    let mut entries = Vec::new();

    loop {
        let mut len_buf = [0u8; 4];
        match file.read_exact(&mut len_buf) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e),
        }
        let payload_len = u32::from_le_bytes(len_buf) as usize;
        let mut payload = vec![0u8; payload_len];
        file.read_exact(&mut payload)?;

        let mut pos = 0;
        let mut id = [0u8; 16];
        id.copy_from_slice(&payload[pos..pos + 16]);
        pos += 16;
        let timestamp = u64::from_le_bytes(payload[pos..pos + 8].try_into().unwrap());
        pos += 8;
        let embed_len = u32::from_le_bytes(payload[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;
        let embedding: Vec<f32> = (0..embed_len)
            .map(|i| f32::from_le_bytes(payload[pos + i * 4..pos + i * 4 + 4].try_into().unwrap()))
            .collect();
        pos += embed_len * 4;
        let cmd_len = u32::from_le_bytes(payload[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;
        let raw_command = String::from_utf8_lossy(&payload[pos..pos + cmd_len]).to_string();

        entries.push(WalEntry {
            id,
            timestamp,
            embedding,
            raw_command,
        });
    }

    Ok(entries)
}

// ===========================================================================
// Hot Buffer
// ===========================================================================

struct BufferedIntent {
    id: [u8; 16],
    timestamp: u64,
    context_hash: u64,
    confidence: f32,
    status: u8,
    embedding: Vec<f32>,
    parent_indices: Vec<usize>,
}

pub struct HotBuffer {
    dim: usize,
    intents: Vec<BufferedIntent>,
    capacity: usize,
}

impl HotBuffer {
    pub fn new(dim: usize, capacity: usize) -> Self {
        Self {
            dim,
            intents: Vec::with_capacity(capacity),
            capacity,
        }
    }

    /// Push an intent. Returns `true` if the buffer is now full and should be flushed.
    pub fn push(
        &mut self,
        id: [u8; 16],
        timestamp: u64,
        context_hash: u64,
        confidence: f32,
        status: u8,
        embedding: Vec<f32>,
        parent_indices: Vec<usize>,
    ) -> bool {
        self.intents.push(BufferedIntent {
            id,
            timestamp,
            context_hash,
            confidence,
            status,
            embedding,
            parent_indices,
        });
        self.intents.len() >= self.capacity
    }

    pub fn len(&self) -> usize {
        self.intents.len()
    }

    pub fn is_empty(&self) -> bool {
        self.intents.is_empty()
    }

    /// Flush the buffer into a TBF segment (in-memory byte buffer).
    pub fn flush(&mut self) -> Vec<u8> {
        let mut writer = SegmentWriter::new(self.dim);
        for (i, intent) in self.intents.iter().enumerate() {
            writer.push_node(
                &intent.id,
                intent.timestamp,
                intent.context_hash,
                intent.confidence,
                intent.status,
                &intent.embedding,
            );
            for &parent in &intent.parent_indices {
                writer.add_edge(parent, i, 0, 1.0);
            }
        }
        self.intents.clear();
        writer.finish()
    }
}

// ===========================================================================
// Semantic Query Engine (HNSW-backed)
// ===========================================================================

struct StoredIntent {
    id: [u8; 16],
    timestamp: u64,
    raw_command: String,
}

pub struct QueryEngine {
    hnsw: HnswIndex,
    intents: Vec<StoredIntent>,
}

impl QueryEngine {
    pub fn new(dim: usize) -> Self {
        Self {
            hnsw: HnswIndex::new(dim, 16, 200),
            intents: Vec::new(),
        }
    }

    pub fn insert(
        &mut self,
        id: [u8; 16],
        timestamp: u64,
        raw_command: String,
        embedding: Vec<f32>,
    ) {
        self.hnsw.insert(embedding);
        self.intents.push(StoredIntent {
            id,
            timestamp,
            raw_command,
        });
    }

    pub fn search(&mut self, query: &[f32], k: usize) -> Vec<([u8; 16], f32)> {
        let results = self.hnsw.search(query, k, 50);
        results
            .into_iter()
            .map(|(idx, dist)| (self.intents[idx].id, dist))
            .collect()
    }

    /// Find edge candidates via HNSW approximate search + exact cosine re-rank.
    /// Returns top-k (id, cosine_similarity) pairs sorted descending by similarity.
    /// Replaces O(n²) brute-force: HNSW finds k*4 candidates in O(log n),
    /// then exact cosine on those few narrows to top-k.
    pub fn find_edge_candidates(
        &mut self,
        embedding: &[f32],
        k: usize,
    ) -> Vec<([u8; 16], f32)> {
        if self.hnsw.is_empty() {
            return Vec::new();
        }
        let search_k = k * 4;
        let hnsw_results = self.hnsw.search(embedding, search_k, search_k.max(50));

        let mut candidates: Vec<([u8; 16], f32)> = hnsw_results
            .iter()
            .map(|&(idx, _)| {
                let sim = cosine_similarity(embedding, self.hnsw.get_vector(idx));
                (self.intents[idx].id, sim)
            })
            .collect();

        candidates.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(k);
        candidates
    }

    pub fn len(&self) -> usize {
        self.intents.len()
    }

    pub fn is_empty(&self) -> bool {
        self.intents.is_empty()
    }
}

// ===========================================================================
// Storage Engine — unified IntentStore implementation
// ===========================================================================

/// Unified storage engine implementing the `IntentStore` trait.
/// Uses interior mutability (Mutex/RwLock) for Send + Sync.
pub struct StorageEngine {
    dim: usize,
    /// All intents by ID — RwLock for concurrent reads.
    intents: RwLock<HashMap<IntentId, Intent>>,
    /// HNSW index for semantic search — Mutex because search mutates visit state.
    hnsw: Mutex<HnswIndex>,
    /// Maps HNSW index position → IntentId.
    index_map: RwLock<Vec<IntentId>>,
    /// Write-ahead log for durability.
    wal: Option<Mutex<Wal>>,
    /// In-memory buffer, flushed to segments at capacity.
    hot: Mutex<HotBuffer>,
    /// Directory for segment files.
    segment_dir: PathBuf,
    /// Monotonic segment counter.
    segment_seq: Mutex<u64>,
    /// Lock and pipeline metrics.
    metrics: Arc<StorageMetrics>,
}

impl StorageEngine {
    /// Create a new storage engine. If `dir` is provided, WAL and segments
    /// are persisted there. Otherwise runs in-memory only.
    pub fn open(dim: usize, dir: impl AsRef<Path>, hot_capacity: usize) -> Result<Self, TalaError> {
        let dir = dir.as_ref();
        std::fs::create_dir_all(dir)?;

        let wal = Wal::create(dir.join("current.wal"))?;
        let metrics = Arc::new(StorageMetrics::new());
        metrics
            .hot_buffer_capacity
            .store(hot_capacity as u64, Relaxed);

        Ok(Self {
            dim,
            intents: RwLock::new(HashMap::new()),
            hnsw: Mutex::new(HnswIndex::new(dim, 16, 200)),
            index_map: RwLock::new(Vec::new()),
            wal: Some(Mutex::new(wal)),
            hot: Mutex::new(HotBuffer::new(dim, hot_capacity)),
            segment_dir: dir.to_path_buf(),
            segment_seq: Mutex::new(0),
            metrics,
        })
    }

    /// In-memory only engine (no WAL, no segment persistence).
    pub fn in_memory(dim: usize, hot_capacity: usize) -> Self {
        let metrics = Arc::new(StorageMetrics::new());
        metrics
            .hot_buffer_capacity
            .store(hot_capacity as u64, Relaxed);

        Self {
            dim,
            intents: RwLock::new(HashMap::new()),
            hnsw: Mutex::new(HnswIndex::new(dim, 16, 200)),
            index_map: RwLock::new(Vec::new()),
            wal: None,
            hot: Mutex::new(HotBuffer::new(dim, hot_capacity)),
            segment_dir: PathBuf::new(),
            segment_seq: Mutex::new(0),
            metrics,
        }
    }

    /// Access the storage metrics.
    pub fn metrics(&self) -> &Arc<StorageMetrics> {
        &self.metrics
    }

    fn flush_segment(&self, segment_bytes: Vec<u8>) -> Result<(), TalaError> {
        if self.segment_dir.as_os_str().is_empty() {
            return Ok(()); // in-memory mode, discard
        }
        let flush_t0 = Instant::now();
        let bytes_len = segment_bytes.len() as u64;
        let mut seq = self.segment_seq.lock().unwrap();
        *seq += 1;
        let path = self.segment_dir.join(format!("segment_{:06}.tbf", *seq));
        std::fs::write(path, segment_bytes)?;
        let flush_ns = flush_t0.elapsed().as_nanos() as u64;
        self.metrics.segment_flush_ns.fetch_add(flush_ns, Relaxed);
        self.metrics.segment_flush_count.fetch_add(1, Relaxed);
        self.metrics.segments_flushed_count.fetch_add(1, Relaxed);
        self.metrics.total_bytes_flushed.fetch_add(bytes_len, Relaxed);
        Ok(())
    }
}

impl IntentStore for StorageEngine {
    fn insert(&self, intent: Intent) -> Result<IntentId, TalaError> {
        let id = intent.id;

        if intent.embedding.len() != self.dim {
            return Err(TalaError::DimensionMismatch {
                expected: self.dim,
                got: intent.embedding.len(),
            });
        }

        // WAL first for durability
        if let Some(ref wal) = self.wal {
            let wal_t0 = Instant::now();
            let mut w = wal.lock().unwrap();
            let wal_wait = wal_t0.elapsed().as_nanos() as u64;
            self.metrics.wal_lock.record_acquisition(wal_wait);

            let wal_op_t0 = Instant::now();
            w.append(
                id.as_bytes(),
                intent.timestamp,
                &intent.embedding,
                &intent.raw_command,
            )?;
            w.sync()?;
            let wal_op_ns = wal_op_t0.elapsed().as_nanos() as u64;
            self.metrics.wal_append_ns.fetch_add(wal_op_ns, Relaxed);
            self.metrics.wal_append_count.fetch_add(1, Relaxed);
            self.metrics
                .wal_entry_count
                .store(w.entry_count(), Relaxed);

            let wal_hold = wal_t0.elapsed().as_nanos() as u64 - wal_wait;
            self.metrics.wal_lock.record_release(wal_hold);
        }

        // HNSW index
        {
            let hnsw_t0 = Instant::now();
            let mut hnsw = self.hnsw.lock().unwrap();
            let hnsw_wait = hnsw_t0.elapsed().as_nanos() as u64;
            self.metrics.hnsw_lock.record_acquisition(hnsw_wait);

            let hnsw_op_t0 = Instant::now();
            hnsw.insert(intent.embedding.clone());
            let hnsw_op_ns = hnsw_op_t0.elapsed().as_nanos() as u64;
            self.metrics.hnsw_insert_ns.fetch_add(hnsw_op_ns, Relaxed);
            self.metrics.hnsw_insert_count.fetch_add(1, Relaxed);

            let idx_t0 = Instant::now();
            let mut map = self.index_map.write().unwrap();
            let idx_wait = idx_t0.elapsed().as_nanos() as u64;
            self.metrics.index_map_lock.record_acquisition(idx_wait);
            map.push(id);
            let idx_hold = idx_t0.elapsed().as_nanos() as u64 - idx_wait;
            self.metrics.index_map_lock.record_release(idx_hold);

            let hnsw_hold = hnsw_t0.elapsed().as_nanos() as u64 - hnsw_wait;
            self.metrics.hnsw_lock.record_release(hnsw_hold);
        }

        // Hot buffer
        let should_flush = {
            let hot_t0 = Instant::now();
            let mut hot = self.hot.lock().unwrap();
            let hot_wait = hot_t0.elapsed().as_nanos() as u64;
            self.metrics.hot_lock.record_acquisition(hot_wait);

            let hot_op_t0 = Instant::now();
            let status_byte = match intent.outcome.as_ref().map(|o| o.status) {
                Some(Status::Pending) | None => 0,
                Some(Status::Success) => 1,
                Some(Status::Failure) => 2,
                Some(Status::Partial) => 3,
            };
            let full = hot.push(
                *id.as_bytes(),
                intent.timestamp,
                intent.context_hash,
                intent.confidence,
                status_byte,
                intent.embedding.clone(),
                vec![], // parent edges handled by edge formation layer
            );
            let hot_op_ns = hot_op_t0.elapsed().as_nanos() as u64;
            self.metrics.hot_push_ns.fetch_add(hot_op_ns, Relaxed);
            self.metrics.hot_push_count.fetch_add(1, Relaxed);
            self.metrics
                .hot_buffer_len
                .store(hot.len() as u64, Relaxed);

            let hot_hold = hot_t0.elapsed().as_nanos() as u64 - hot_wait;
            self.metrics.hot_lock.record_release(hot_hold);
            full
        };

        if should_flush {
            let segment_bytes = {
                let hot_t0 = Instant::now();
                let mut hot = self.hot.lock().unwrap();
                let hot_wait = hot_t0.elapsed().as_nanos() as u64;
                self.metrics.hot_lock.record_acquisition(hot_wait);

                let bytes = hot.flush();
                self.metrics.hot_buffer_len.store(0, Relaxed);

                let hot_hold = hot_t0.elapsed().as_nanos() as u64 - hot_wait;
                self.metrics.hot_lock.record_release(hot_hold);
                bytes
            };
            self.flush_segment(segment_bytes)?;
        }

        // Intent store (last — after durability is guaranteed)
        {
            let store_t0 = Instant::now();
            let mut store = self.intents.write().unwrap();
            let store_wait = store_t0.elapsed().as_nanos() as u64;
            self.metrics.intents_lock.record_acquisition(store_wait);

            store.insert(id, intent);

            let store_hold = store_t0.elapsed().as_nanos() as u64 - store_wait;
            self.metrics.intents_lock.record_release(store_hold);
        }

        Ok(id)
    }

    fn get(&self, id: IntentId) -> Result<Option<Intent>, TalaError> {
        let t0 = Instant::now();
        let store = self.intents.read().unwrap();
        let wait = t0.elapsed().as_nanos() as u64;
        self.metrics.intents_lock.record_acquisition(wait);

        let result = store.get(&id).cloned();

        let hold = t0.elapsed().as_nanos() as u64 - wait;
        self.metrics.intents_lock.record_release(hold);

        Ok(result)
    }

    fn query_semantic(&self, embedding: &[f32], k: usize) -> Result<Vec<(IntentId, f32)>, TalaError> {
        if embedding.len() != self.dim {
            return Err(TalaError::DimensionMismatch {
                expected: self.dim,
                got: embedding.len(),
            });
        }

        let search_t0 = Instant::now();

        let hnsw_results = {
            let hnsw_t0 = Instant::now();
            let mut hnsw = self.hnsw.lock().unwrap();
            let hnsw_wait = hnsw_t0.elapsed().as_nanos() as u64;
            self.metrics.hnsw_lock.record_acquisition(hnsw_wait);

            let results = hnsw.search(embedding, k, k.max(50));

            let hnsw_hold = hnsw_t0.elapsed().as_nanos() as u64 - hnsw_wait;
            self.metrics.hnsw_lock.record_release(hnsw_hold);
            self.metrics.hnsw_search_count.fetch_add(1, Relaxed);
            self.metrics
                .hnsw_search_visited
                .fetch_add(results.len() as u64, Relaxed);
            results
        };

        let idx_t0 = Instant::now();
        let map = self.index_map.read().unwrap();
        let idx_wait = idx_t0.elapsed().as_nanos() as u64;
        self.metrics.index_map_lock.record_acquisition(idx_wait);

        let results = hnsw_results
            .into_iter()
            .filter_map(|(idx, _l2_dist)| {
                let id = *map.get(idx)?;
                let hnsw = self.hnsw.lock().unwrap();
                let sim = cosine_similarity(embedding, hnsw.get_vector(idx));
                Some((id, sim))
            })
            .collect();

        let idx_hold = idx_t0.elapsed().as_nanos() as u64 - idx_wait;
        self.metrics.index_map_lock.record_release(idx_hold);

        let search_ns = search_t0.elapsed().as_nanos() as u64;
        self.metrics.edge_search_ns.fetch_add(search_ns, Relaxed);
        self.metrics.edge_search_count.fetch_add(1, Relaxed);

        Ok(results)
    }

    fn query_temporal(&self, range: TimeRange) -> Result<Vec<Intent>, TalaError> {
        let t0 = Instant::now();
        let store = self.intents.read().unwrap();
        let wait = t0.elapsed().as_nanos() as u64;
        self.metrics.intents_lock.record_acquisition(wait);

        let mut results: Vec<Intent> = store
            .values()
            .filter(|intent| intent.timestamp >= range.start && intent.timestamp < range.end)
            .cloned()
            .collect();
        results.sort_by_key(|i| i.timestamp);

        let hold = t0.elapsed().as_nanos() as u64 - wait;
        self.metrics.intents_lock.record_release(hold);

        Ok(results)
    }

    fn attach_outcome(&self, id: IntentId, outcome: Outcome) -> Result<(), TalaError> {
        let t0 = Instant::now();
        let mut store = self.intents.write().unwrap();
        let wait = t0.elapsed().as_nanos() as u64;
        self.metrics.intents_lock.record_acquisition(wait);

        let result = match store.get_mut(&id) {
            Some(intent) => {
                intent.outcome = Some(outcome);
                Ok(())
            }
            None => Err(TalaError::NodeNotFound(id)),
        };

        let hold = t0.elapsed().as_nanos() as u64 - wait;
        self.metrics.intents_lock.record_release(hold);

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tala_core::Status;

    fn test_intent(dim: usize, ts: u64) -> Intent {
        Intent {
            id: IntentId::random(),
            timestamp: ts,
            raw_command: format!("test_cmd_{ts}"),
            embedding: (0..dim).map(|i| (ts as f32 + i as f32 * 0.01).sin()).collect(),
            context_hash: ts * 7,
            parent_ids: vec![],
            outcome: None,
            confidence: 0.95,
        }
    }

    #[test]
    fn storage_engine_insert_get() {
        let engine = StorageEngine::in_memory(8, 1000);
        let intent = test_intent(8, 100);
        let id = intent.id;

        engine.insert(intent.clone()).unwrap();
        let got = engine.get(id).unwrap().expect("should find intent");
        assert_eq!(got.id, id);
        assert_eq!(got.timestamp, 100);
        assert_eq!(got.raw_command, "test_cmd_100");
    }

    #[test]
    fn storage_engine_get_missing() {
        let engine = StorageEngine::in_memory(8, 1000);
        let result = engine.get(IntentId::random()).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn storage_engine_semantic_query() {
        let engine = StorageEngine::in_memory(8, 1000);
        for ts in 0..50 {
            engine.insert(test_intent(8, ts)).unwrap();
        }

        let query = test_intent(8, 25);
        let results = engine.query_semantic(&query.embedding, 5).unwrap();
        assert_eq!(results.len(), 5);
        // All results should have valid similarity scores
        for &(_, sim) in &results {
            assert!(sim >= -1.0 && sim <= 1.0, "sim {sim} out of range");
        }
    }

    #[test]
    fn storage_engine_temporal_query() {
        let engine = StorageEngine::in_memory(8, 1000);
        for ts in 0..20 {
            engine.insert(test_intent(8, ts * 100)).unwrap();
        }

        let results = engine
            .query_temporal(TimeRange { start: 500, end: 1500 })
            .unwrap();
        assert_eq!(results.len(), 10); // timestamps 500,600,...,1400
        // Should be sorted
        for w in results.windows(2) {
            assert!(w[0].timestamp <= w[1].timestamp);
        }
    }

    #[test]
    fn storage_engine_attach_outcome() {
        let engine = StorageEngine::in_memory(8, 1000);
        let intent = test_intent(8, 100);
        let id = intent.id;
        engine.insert(intent).unwrap();

        assert!(engine.get(id).unwrap().unwrap().outcome.is_none());

        let outcome = Outcome {
            status: Status::Success,
            latency_ns: 42_000,
            exit_code: 0,
        };
        engine.attach_outcome(id, outcome).unwrap();

        let got = engine.get(id).unwrap().unwrap();
        assert_eq!(got.outcome.unwrap().status, Status::Success);
    }

    #[test]
    fn storage_engine_attach_outcome_missing() {
        let engine = StorageEngine::in_memory(8, 1000);
        let outcome = Outcome {
            status: Status::Failure,
            latency_ns: 0,
            exit_code: 1,
        };
        let result = engine.attach_outcome(IntentId::random(), outcome);
        assert!(result.is_err());
    }

    #[test]
    fn storage_engine_dimension_mismatch() {
        let engine = StorageEngine::in_memory(8, 1000);
        let mut intent = test_intent(8, 100);
        intent.embedding = vec![1.0; 16]; // wrong dimension
        let result = engine.insert(intent);
        assert!(result.is_err());
    }
}
