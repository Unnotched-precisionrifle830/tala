//! TBF: TALA Binary Format — columnar storage, CSR edges, aligned embeddings, bloom filters.

pub const MAGIC: [u8; 4] = *b"TALB";
pub const HEADER_SIZE: usize = 128;
pub const ALIGN: usize = 64;

/// Round `value` up to the next multiple of `alignment`.
#[inline]
pub fn align_up(value: usize, alignment: usize) -> usize {
    (value + alignment - 1) & !(alignment - 1)
}

// ---------------------------------------------------------------------------
// Bloom Filter
// ---------------------------------------------------------------------------

pub struct BloomFilter {
    pub bits: Vec<u64>,
    pub num_hashes: u32,
    pub num_bits: u64,
}

impl BloomFilter {
    pub fn new(expected_items: usize, fp_rate: f64) -> Self {
        let n = expected_items.max(1) as f64;
        let num_bits =
            (-(n) * fp_rate.ln() / (2.0_f64.ln().powi(2))).ceil() as u64;
        let num_bits = num_bits.max(64);
        let num_hashes =
            ((num_bits as f64 / n) * 2.0_f64.ln()).ceil() as u32;
        let num_hashes = num_hashes.max(1);
        let words = ((num_bits + 63) / 64) as usize;
        Self {
            bits: vec![0u64; words],
            num_hashes,
            num_bits,
        }
    }

    pub fn insert(&mut self, key: &[u8]) {
        let (h1, h2) = hash_pair(key);
        for i in 0..self.num_hashes {
            let bit = bit_index(h1, h2, i, self.num_bits);
            self.bits[bit / 64] |= 1u64 << (bit % 64);
        }
    }

    pub fn contains(&self, key: &[u8]) -> bool {
        let (h1, h2) = hash_pair(key);
        for i in 0..self.num_hashes {
            let bit = bit_index(h1, h2, i, self.num_bits);
            if self.bits[bit / 64] & (1u64 << (bit % 64)) == 0 {
                return false;
            }
        }
        true
    }

    pub fn size_bytes(&self) -> usize {
        self.bits.len() * 8
    }
}

fn hash_pair(key: &[u8]) -> (u64, u64) {
    let h1 = fnv1a(key);
    let h2 = fnv1a(&h1.to_le_bytes()).wrapping_add(1);
    (h1, h2)
}

fn bit_index(h1: u64, h2: u64, i: u32, num_bits: u64) -> usize {
    (h1.wrapping_add(h2.wrapping_mul(i as u64)) % num_bits) as usize
}

fn fnv1a(data: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &byte in data {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

// ---------------------------------------------------------------------------
// CSR (Compressed Sparse Row)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct CsrEdge {
    pub target: u32,
    pub relation: u8,
    pub weight: f32,
}

pub struct CsrBuilder {
    node_count: usize,
    edges: Vec<Vec<CsrEdge>>,
}

impl CsrBuilder {
    pub fn new(node_count: usize) -> Self {
        Self {
            node_count,
            edges: vec![Vec::new(); node_count],
        }
    }

    pub fn add_edge(&mut self, from: usize, to: usize, relation: u8, weight: f32) {
        if from < self.node_count {
            self.edges[from].push(CsrEdge {
                target: to as u32,
                relation,
                weight,
            });
        }
    }

    pub fn build(self) -> CsrIndex {
        let mut row_offsets = Vec::with_capacity(self.node_count + 1);
        let mut flat_edges = Vec::new();
        let mut offset = 0u64;
        for node_edges in &self.edges {
            row_offsets.push(offset);
            for edge in node_edges {
                flat_edges.push(edge.clone());
            }
            offset += node_edges.len() as u64;
        }
        row_offsets.push(offset);
        CsrIndex {
            row_offsets,
            edges: flat_edges,
        }
    }
}

pub struct CsrIndex {
    row_offsets: Vec<u64>,
    edges: Vec<CsrEdge>,
}

impl CsrIndex {
    pub fn edges_from(&self, node: usize) -> &[CsrEdge] {
        let start = self.row_offsets[node] as usize;
        let end = self.row_offsets[node + 1] as usize;
        &self.edges[start..end]
    }

    pub fn degree(&self, node: usize) -> usize {
        (self.row_offsets[node + 1] - self.row_offsets[node]) as usize
    }

    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    pub fn node_count(&self) -> usize {
        self.row_offsets.len() - 1
    }
}

// ---------------------------------------------------------------------------
// Columnar Buffer
// ---------------------------------------------------------------------------

/// In-memory columnar storage for intent node fields.
pub struct ColumnarBuffer {
    pub ids: Vec<[u8; 16]>,
    pub timestamps: Vec<u64>,
    pub context_hashes: Vec<u64>,
    pub confidences: Vec<f32>,
    pub outcome_statuses: Vec<u8>,
}

impl ColumnarBuffer {
    pub fn new() -> Self {
        Self {
            ids: Vec::new(),
            timestamps: Vec::new(),
            context_hashes: Vec::new(),
            confidences: Vec::new(),
            outcome_statuses: Vec::new(),
        }
    }

    pub fn with_capacity(cap: usize) -> Self {
        Self {
            ids: Vec::with_capacity(cap),
            timestamps: Vec::with_capacity(cap),
            context_hashes: Vec::with_capacity(cap),
            confidences: Vec::with_capacity(cap),
            outcome_statuses: Vec::with_capacity(cap),
        }
    }

    pub fn push(
        &mut self,
        id: &[u8; 16],
        timestamp: u64,
        context_hash: u64,
        confidence: f32,
        status: u8,
    ) {
        self.ids.push(*id);
        self.timestamps.push(timestamp);
        self.context_hashes.push(context_hash);
        self.confidences.push(confidence);
        self.outcome_statuses.push(status);
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Serialize all columns into a flat byte buffer.
    /// Returns (buffer, column_offsets) where offsets index:
    /// [0]=ids, [1]=timestamps, [2]=context_hashes, [3]=confidences, [4]=statuses
    pub fn serialize(&self) -> (Vec<u8>, Vec<usize>) {
        let mut buf = Vec::new();
        let mut offsets = Vec::new();

        // ids: 16 bytes each
        let start = align_up(buf.len(), ALIGN);
        buf.resize(start, 0);
        offsets.push(buf.len());
        for id in &self.ids {
            buf.extend_from_slice(id);
        }

        // timestamps: 8 bytes each
        let start = align_up(buf.len(), ALIGN);
        buf.resize(start, 0);
        offsets.push(buf.len());
        for &ts in &self.timestamps {
            buf.extend_from_slice(&ts.to_le_bytes());
        }

        // context_hashes: 8 bytes each
        let start = align_up(buf.len(), ALIGN);
        buf.resize(start, 0);
        offsets.push(buf.len());
        for &ch in &self.context_hashes {
            buf.extend_from_slice(&ch.to_le_bytes());
        }

        // confidences: 4 bytes each
        let start = align_up(buf.len(), ALIGN);
        buf.resize(start, 0);
        offsets.push(buf.len());
        for &c in &self.confidences {
            buf.extend_from_slice(&c.to_le_bytes());
        }

        // outcome_statuses: 1 byte each
        let start = align_up(buf.len(), ALIGN);
        buf.resize(start, 0);
        offsets.push(buf.len());
        buf.extend_from_slice(&self.outcome_statuses);

        (buf, offsets)
    }
}

impl Default for ColumnarBuffer {
    fn default() -> Self {
        Self::new()
    }
}

/// Zero-copy typed access to a serialized column buffer.
pub struct ColumnReader<'a> {
    data: &'a [u8],
}

impl<'a> ColumnReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data }
    }

    #[inline]
    pub fn read_u64(&self, col_offset: usize, index: usize) -> u64 {
        let pos = col_offset + index * 8;
        u64::from_le_bytes(self.data[pos..pos + 8].try_into().unwrap())
    }

    #[inline]
    pub fn read_f32(&self, col_offset: usize, index: usize) -> f32 {
        let pos = col_offset + index * 4;
        f32::from_le_bytes(self.data[pos..pos + 4].try_into().unwrap())
    }

    #[inline]
    pub fn read_u8(&self, col_offset: usize, index: usize) -> u8 {
        self.data[col_offset + index]
    }

    #[inline]
    pub fn read_id(&self, col_offset: usize, index: usize) -> [u8; 16] {
        let pos = col_offset + index * 16;
        self.data[pos..pos + 16].try_into().unwrap()
    }
}

// ---------------------------------------------------------------------------
// Embedding Region — 64-byte aligned vector I/O
// ---------------------------------------------------------------------------

pub struct EmbeddingWriter {
    dim: usize,
    stride: usize,
    data: Vec<u8>,
}

impl EmbeddingWriter {
    pub fn new(dim: usize) -> Self {
        let stride = align_up(dim * 4, ALIGN);
        Self {
            dim,
            stride,
            data: Vec::new(),
        }
    }

    pub fn push(&mut self, embedding: &[f32]) {
        debug_assert_eq!(embedding.len(), self.dim);
        let start = self.data.len();
        self.data.resize(start + self.stride, 0);
        // Safe: f32 has no invalid bit patterns
        let byte_slice = unsafe {
            std::slice::from_raw_parts(embedding.as_ptr() as *const u8, embedding.len() * 4)
        };
        self.data[start..start + byte_slice.len()].copy_from_slice(byte_slice);
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    pub fn stride(&self) -> usize {
        self.stride
    }

    pub fn count(&self) -> usize {
        if self.stride == 0 {
            0
        } else {
            self.data.len() / self.stride
        }
    }
}

/// Zero-copy read access to aligned embedding vectors.
pub struct EmbeddingReader<'a> {
    data: &'a [u8],
    dim: usize,
    stride: usize,
}

impl<'a> EmbeddingReader<'a> {
    pub fn new(data: &'a [u8], dim: usize) -> Self {
        let stride = align_up(dim * 4, ALIGN);
        Self { data, dim, stride }
    }

    #[inline]
    pub fn get(&self, index: usize) -> &[f32] {
        let offset = index * self.stride;
        unsafe { std::slice::from_raw_parts(self.data[offset..].as_ptr() as *const f32, self.dim) }
    }

    pub fn count(&self) -> usize {
        if self.stride == 0 {
            0
        } else {
            self.data.len() / self.stride
        }
    }
}

// ---------------------------------------------------------------------------
// Segment Reader — deserializes a TBF segment from bytes
// ---------------------------------------------------------------------------

/// Parsed TBF segment header.
pub struct SegmentHeader {
    pub version_minor: u16,
    pub version_major: u16,
    pub node_count: u64,
    pub edge_count: u64,
    pub dim: u32,
    pub node_region_offset: u64,
    pub embed_region_offset: u64,
    pub edge_region_offset: u64,
    pub bloom_offset: u64,
}

/// Reads and validates a TBF segment from a byte buffer.
pub struct SegmentReader<'a> {
    data: &'a [u8],
    header: SegmentHeader,
    /// Column offsets relative to the start of the buffer
    col_offsets: [usize; 5],
}

impl<'a> SegmentReader<'a> {
    /// Parse a TBF segment from raw bytes. Returns an error message on failure.
    pub fn open(data: &'a [u8]) -> Result<Self, &'static str> {
        if data.len() < HEADER_SIZE {
            return Err("segment too small for header");
        }
        if data[0..4] != MAGIC {
            return Err("invalid magic bytes");
        }

        let header = SegmentHeader {
            version_minor: u16::from_le_bytes(data[4..6].try_into().unwrap()),
            version_major: u16::from_le_bytes(data[6..8].try_into().unwrap()),
            node_count: u64::from_le_bytes(data[24..32].try_into().unwrap()),
            edge_count: u64::from_le_bytes(data[32..40].try_into().unwrap()),
            dim: u32::from_le_bytes(data[40..44].try_into().unwrap()),
            node_region_offset: u64::from_le_bytes(data[48..56].try_into().unwrap()),
            embed_region_offset: u64::from_le_bytes(data[56..64].try_into().unwrap()),
            edge_region_offset: u64::from_le_bytes(data[64..72].try_into().unwrap()),
            bloom_offset: u64::from_le_bytes(data[80..88].try_into().unwrap()),
        };

        // Derive column offsets within the node region.
        // ColumnarBuffer::serialize() lays out: ids, timestamps, context_hashes,
        // confidences, statuses — each 64-byte aligned.
        let n = header.node_count as usize;
        let base = header.node_region_offset as usize;
        let ids_off = base; // first column, already aligned
        let ts_off = align_up(ids_off + n * 16, ALIGN);
        let ch_off = align_up(ts_off + n * 8, ALIGN);
        let conf_off = align_up(ch_off + n * 8, ALIGN);
        let stat_off = align_up(conf_off + n * 4, ALIGN);

        Ok(Self {
            data,
            header,
            col_offsets: [ids_off, ts_off, ch_off, conf_off, stat_off],
        })
    }

    pub fn header(&self) -> &SegmentHeader {
        &self.header
    }

    pub fn node_count(&self) -> usize {
        self.header.node_count as usize
    }

    pub fn edge_count(&self) -> usize {
        self.header.edge_count as usize
    }

    pub fn dim(&self) -> usize {
        self.header.dim as usize
    }

    /// Column reader over the raw segment bytes.
    pub fn column_reader(&self) -> ColumnReader<'a> {
        ColumnReader::new(self.data)
    }

    /// Read a node ID by index.
    #[inline]
    pub fn read_id(&self, index: usize) -> [u8; 16] {
        self.column_reader().read_id(self.col_offsets[0], index)
    }

    /// Read a timestamp by index.
    #[inline]
    pub fn read_timestamp(&self, index: usize) -> u64 {
        self.column_reader().read_u64(self.col_offsets[1], index)
    }

    /// Read a context hash by index.
    #[inline]
    pub fn read_context_hash(&self, index: usize) -> u64 {
        self.column_reader().read_u64(self.col_offsets[2], index)
    }

    /// Read a confidence score by index.
    #[inline]
    pub fn read_confidence(&self, index: usize) -> f32 {
        self.column_reader().read_f32(self.col_offsets[3], index)
    }

    /// Read an outcome status by index.
    #[inline]
    pub fn read_status(&self, index: usize) -> u8 {
        self.column_reader().read_u8(self.col_offsets[4], index)
    }

    /// Zero-copy embedding access.
    pub fn embedding_reader(&self) -> EmbeddingReader<'a> {
        let offset = self.header.embed_region_offset as usize;
        let dim = self.header.dim as usize;
        let stride = align_up(dim * 4, ALIGN);
        let n = self.header.node_count as usize;
        let end = offset + n * stride;
        EmbeddingReader::new(&self.data[offset..end], dim)
    }

    /// Reconstruct the CSR edge index from the edge region.
    pub fn csr_index(&self) -> CsrIndex {
        let offset = self.header.edge_region_offset as usize;
        let n = self.header.node_count as usize;
        let edge_count = self.header.edge_count as usize;

        // Row pointers: (n+1) × 8 bytes
        let mut row_offsets = Vec::with_capacity(n + 1);
        for i in 0..=n {
            let pos = offset + i * 8;
            row_offsets.push(u64::from_le_bytes(
                self.data[pos..pos + 8].try_into().unwrap(),
            ));
        }

        // Edge entries: 12 bytes each (target:4 + relation:1 + weight:4 + pad:3)
        let edges_start = offset + (n + 1) * 8;
        let mut edges = Vec::with_capacity(edge_count);
        for i in 0..edge_count {
            let pos = edges_start + i * 12;
            let target = u32::from_le_bytes(self.data[pos..pos + 4].try_into().unwrap());
            let relation = self.data[pos + 4];
            let weight = f32::from_le_bytes(self.data[pos + 5..pos + 9].try_into().unwrap());
            edges.push(CsrEdge {
                target,
                relation,
                weight,
            });
        }

        CsrIndex { row_offsets, edges }
    }

    /// Reconstruct the bloom filter from the bloom region.
    pub fn bloom_filter(&self) -> BloomFilter {
        let offset = self.header.bloom_offset as usize;
        let num_hashes = u32::from_le_bytes(self.data[offset..offset + 4].try_into().unwrap());
        let num_bits =
            u64::from_le_bytes(self.data[offset + 4..offset + 12].try_into().unwrap());
        let num_words = ((num_bits + 63) / 64) as usize;
        let mut bits = Vec::with_capacity(num_words);
        for i in 0..num_words {
            let pos = offset + 12 + i * 8;
            bits.push(u64::from_le_bytes(
                self.data[pos..pos + 8].try_into().unwrap(),
            ));
        }
        BloomFilter {
            bits,
            num_hashes,
            num_bits,
        }
    }

    /// Raw segment bytes.
    pub fn as_bytes(&self) -> &[u8] {
        self.data
    }
}

// ---------------------------------------------------------------------------
// Segment Writer — orchestrates all regions into a TBF segment
// ---------------------------------------------------------------------------

pub struct SegmentWriter {
    dim: usize,
    columns: ColumnarBuffer,
    embeddings: EmbeddingWriter,
    edges: Vec<(usize, usize, u8, f32)>,
}

impl SegmentWriter {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            columns: ColumnarBuffer::new(),
            embeddings: EmbeddingWriter::new(dim),
            edges: Vec::new(),
        }
    }

    pub fn push_node(
        &mut self,
        id: &[u8; 16],
        timestamp: u64,
        context_hash: u64,
        confidence: f32,
        status: u8,
        embedding: &[f32],
    ) {
        self.columns
            .push(id, timestamp, context_hash, confidence, status);
        self.embeddings.push(embedding);
    }

    pub fn add_edge(&mut self, from: usize, to: usize, relation: u8, weight: f32) {
        self.edges.push((from, to, relation, weight));
    }

    /// Finalize the segment into a contiguous byte buffer.
    pub fn finish(self) -> Vec<u8> {
        let n = self.columns.len();
        let mut buf = vec![0u8; HEADER_SIZE];

        // -- Node columns --
        let node_region_offset = align_up(buf.len(), ALIGN);
        buf.resize(node_region_offset, 0);
        let (col_data, _offsets) = self.columns.serialize();
        buf.extend_from_slice(&col_data);

        // -- Embeddings --
        let embed_region_offset = align_up(buf.len(), ALIGN);
        buf.resize(embed_region_offset, 0);
        buf.extend_from_slice(self.embeddings.as_bytes());

        // -- Edges (CSR) --
        let edge_region_offset = align_up(buf.len(), ALIGN);
        buf.resize(edge_region_offset, 0);
        let mut csr = CsrBuilder::new(n);
        for &(from, to, rel, weight) in &self.edges {
            csr.add_edge(from, to, rel, weight);
        }
        let csr_index = csr.build();
        // Row pointers
        for &off in &csr_index.row_offsets {
            buf.extend_from_slice(&off.to_le_bytes());
        }
        // Edge entries: target(4) + relation(1) + weight(4) + pad(3) = 12 bytes
        for edge in &csr_index.edges {
            buf.extend_from_slice(&edge.target.to_le_bytes());
            buf.push(edge.relation);
            buf.extend_from_slice(&edge.weight.to_le_bytes());
            buf.extend_from_slice(&[0u8; 3]);
        }

        // -- Bloom filter --
        let bloom_offset = align_up(buf.len(), ALIGN);
        buf.resize(bloom_offset, 0);
        let mut bloom = BloomFilter::new(n.max(1), 0.01);
        for id in &self.columns.ids {
            bloom.insert(id);
        }
        buf.extend_from_slice(&bloom.num_hashes.to_le_bytes());
        buf.extend_from_slice(&bloom.num_bits.to_le_bytes());
        for &word in &bloom.bits {
            buf.extend_from_slice(&word.to_le_bytes());
        }

        // -- Header --
        buf[0..4].copy_from_slice(&MAGIC);
        buf[4..6].copy_from_slice(&0u16.to_le_bytes());
        buf[6..8].copy_from_slice(&1u16.to_le_bytes());
        buf[24..32].copy_from_slice(&(n as u64).to_le_bytes());
        buf[32..40].copy_from_slice(&(self.edges.len() as u64).to_le_bytes());
        buf[40..44].copy_from_slice(&(self.dim as u32).to_le_bytes());
        buf[48..56].copy_from_slice(&(node_region_offset as u64).to_le_bytes());
        buf[56..64].copy_from_slice(&(embed_region_offset as u64).to_le_bytes());
        buf[64..72].copy_from_slice(&(edge_region_offset as u64).to_le_bytes());
        buf[80..88].copy_from_slice(&(bloom_offset as u64).to_le_bytes());

        buf
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_id(i: u8) -> [u8; 16] {
        let mut id = [0u8; 16];
        id[0] = i;
        id[15] = i.wrapping_mul(7);
        id
    }

    fn make_embedding(dim: usize, seed: f32) -> Vec<f32> {
        (0..dim).map(|i| (seed + i as f32 * 0.01).sin()).collect()
    }

    /// Full roundtrip: write a segment with SegmentWriter, read it back with SegmentReader,
    /// verify every field matches.
    #[test]
    fn segment_roundtrip_basic() {
        let dim = 8;
        let n = 5;

        let mut writer = SegmentWriter::new(dim);
        let mut ids = Vec::new();
        let mut timestamps = Vec::new();
        let mut context_hashes = Vec::new();
        let mut confidences = Vec::new();
        let mut statuses = Vec::new();
        let mut embeddings = Vec::new();

        for i in 0..n {
            let id = make_id(i as u8);
            let ts = 1000 + i as u64 * 100;
            let ch = 0xDEAD_0000 + i as u64;
            let conf = 0.5 + i as f32 * 0.1;
            let status = (i % 3) as u8;
            let emb = make_embedding(dim, i as f32);

            writer.push_node(&id, ts, ch, conf, status, &emb);
            ids.push(id);
            timestamps.push(ts);
            context_hashes.push(ch);
            confidences.push(conf);
            statuses.push(status);
            embeddings.push(emb);
        }

        // Add edges: 0→1, 0→2, 3→4
        writer.add_edge(0, 1, 0, 0.9);
        writer.add_edge(0, 2, 1, 0.7);
        writer.add_edge(3, 4, 2, 0.5);

        let buf = writer.finish();
        let reader = SegmentReader::open(&buf).expect("should parse");

        // Header
        assert_eq!(reader.node_count(), n);
        assert_eq!(reader.edge_count(), 3);
        assert_eq!(reader.dim(), dim);
        assert_eq!(reader.header().version_major, 1);

        // Node columns
        for i in 0..n {
            assert_eq!(reader.read_id(i), ids[i], "id mismatch at {i}");
            assert_eq!(reader.read_timestamp(i), timestamps[i], "ts mismatch at {i}");
            assert_eq!(reader.read_context_hash(i), context_hashes[i], "ch mismatch at {i}");
            assert!(
                (reader.read_confidence(i) - confidences[i]).abs() < f32::EPSILON,
                "confidence mismatch at {i}"
            );
            assert_eq!(reader.read_status(i), statuses[i], "status mismatch at {i}");
        }

        // Embeddings
        let emb_reader = reader.embedding_reader();
        assert_eq!(emb_reader.count(), n);
        for i in 0..n {
            let read_emb = emb_reader.get(i);
            for (j, (&got, &expected)) in read_emb.iter().zip(embeddings[i].iter()).enumerate() {
                assert!(
                    (got - expected).abs() < f32::EPSILON,
                    "embedding mismatch at node {i}, dim {j}: {got} != {expected}"
                );
            }
        }

        // CSR edges
        let csr = reader.csr_index();
        assert_eq!(csr.node_count(), n);
        assert_eq!(csr.edge_count(), 3);

        let e0 = csr.edges_from(0);
        assert_eq!(e0.len(), 2);
        assert_eq!(e0[0].target, 1);
        assert_eq!(e0[0].relation, 0);
        assert!((e0[0].weight - 0.9).abs() < f32::EPSILON);
        assert_eq!(e0[1].target, 2);

        assert_eq!(csr.edges_from(1).len(), 0);
        assert_eq!(csr.edges_from(2).len(), 0);

        let e3 = csr.edges_from(3);
        assert_eq!(e3.len(), 1);
        assert_eq!(e3[0].target, 4);

        // Bloom filter
        let bloom = reader.bloom_filter();
        for id in &ids {
            assert!(bloom.contains(id), "bloom should contain inserted id");
        }
        // Spot check: random key should (usually) not match
        let fake = [0xFFu8; 16];
        // Not a guarantee, but with 5 items and 1% FP rate, very unlikely
        let _ = bloom.contains(&fake); // just ensure no panic
    }

    /// Empty segment roundtrip.
    #[test]
    fn segment_roundtrip_empty() {
        let dim = 4;
        let writer = SegmentWriter::new(dim);
        let buf = writer.finish();
        let reader = SegmentReader::open(&buf).expect("should parse empty segment");
        assert_eq!(reader.node_count(), 0);
        assert_eq!(reader.edge_count(), 0);
        assert_eq!(reader.dim(), dim);
    }

    /// Large segment with realistic dimensions.
    #[test]
    fn segment_roundtrip_dim384() {
        let dim = 384;
        let n = 100;
        let mut writer = SegmentWriter::new(dim);

        for i in 0..n {
            let id = make_id(i as u8);
            let emb = make_embedding(dim, i as f32);
            writer.push_node(&id, i as u64, i as u64 * 7, 0.95, 1, &emb);
            if i > 0 {
                writer.add_edge(i - 1, i, 0, 1.0);
            }
        }

        let buf = writer.finish();
        let reader = SegmentReader::open(&buf).expect("should parse");
        assert_eq!(reader.node_count(), n);
        assert_eq!(reader.edge_count(), n - 1);

        // Spot-check first and last embedding
        let emb_reader = reader.embedding_reader();
        let first = emb_reader.get(0);
        let expected = make_embedding(dim, 0.0);
        assert!((first[0] - expected[0]).abs() < f32::EPSILON);
        assert!((first[383] - expected[383]).abs() < f32::EPSILON);

        let last = emb_reader.get(n - 1);
        let expected = make_embedding(dim, (n - 1) as f32);
        assert!((last[0] - expected[0]).abs() < f32::EPSILON);
    }

    /// Validate header rejection for bad magic.
    #[test]
    fn segment_reader_rejects_bad_magic() {
        let mut buf = vec![0u8; HEADER_SIZE];
        buf[0..4].copy_from_slice(b"NOPE");
        assert!(SegmentReader::open(&buf).is_err());
    }

    /// Validate header rejection for truncated input.
    #[test]
    fn segment_reader_rejects_too_small() {
        let buf = vec![0u8; 64];
        assert!(SegmentReader::open(&buf).is_err());
    }

    /// Columnar buffer serialize/deserialize roundtrip.
    #[test]
    fn columnar_roundtrip() {
        let mut cols = ColumnarBuffer::new();
        for i in 0..20 {
            let id = make_id(i);
            cols.push(&id, i as u64 * 1000, i as u64 * 7, 0.5 + i as f32 * 0.01, (i % 4) as u8);
        }

        let (data, offsets) = cols.serialize();
        let reader = ColumnReader::new(&data);

        for i in 0..20 {
            assert_eq!(reader.read_id(offsets[0], i), make_id(i as u8));
            assert_eq!(reader.read_u64(offsets[1], i), i as u64 * 1000);
            assert_eq!(reader.read_u64(offsets[2], i), i as u64 * 7);
            assert!((reader.read_f32(offsets[3], i) - (0.5 + i as f32 * 0.01)).abs() < f32::EPSILON);
            assert_eq!(reader.read_u8(offsets[4], i), (i % 4) as u8);
        }
    }

    /// CSR builder roundtrip.
    #[test]
    fn csr_roundtrip() {
        let mut builder = CsrBuilder::new(4);
        builder.add_edge(0, 1, 0, 1.0);
        builder.add_edge(0, 2, 1, 0.5);
        builder.add_edge(2, 3, 0, 0.8);
        let index = builder.build();

        assert_eq!(index.node_count(), 4);
        assert_eq!(index.edge_count(), 3);
        assert_eq!(index.degree(0), 2);
        assert_eq!(index.degree(1), 0);
        assert_eq!(index.degree(2), 1);
        assert_eq!(index.degree(3), 0);

        let e0 = index.edges_from(0);
        assert_eq!(e0[0].target, 1);
        assert_eq!(e0[1].target, 2);
    }

    /// Bloom filter insert/contains roundtrip.
    #[test]
    fn bloom_roundtrip() {
        let mut bloom = BloomFilter::new(100, 0.01);
        let keys: Vec<[u8; 16]> = (0..100).map(|i| make_id(i)).collect();
        for key in &keys {
            bloom.insert(key);
        }
        for key in &keys {
            assert!(bloom.contains(key), "bloom should contain inserted key");
        }
    }
}
