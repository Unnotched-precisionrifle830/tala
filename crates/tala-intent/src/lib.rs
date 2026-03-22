//! TALA Intent — Converts raw command strings into structured Intent objects.
//!
//! Pipeline: tokenize → embed → classify → assemble Intent.

use tala_core::{Context, Intent, IntentCategory, IntentExtractor, IntentId, TalaError};
use tala_embed::cosine_similarity;

// ===========================================================================
// Configuration
// ===========================================================================

/// Embedding dimension used for intent vectors.
const EMBED_DIM: usize = 384;

// ===========================================================================
// Tokenizer
// ===========================================================================

/// A parsed token from a raw shell command.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Token {
    /// The command name (first word, or word after a pipe).
    Command(String),
    /// A positional argument.
    Arg(String),
    /// A flag (short `-x` or long `--foo`).
    Flag(String),
    /// Pipe operator `|`.
    Pipe,
    /// Input redirect `<`.
    RedirectIn,
    /// Output redirect `>` or append `>>`.
    RedirectOut { append: bool },
}

/// Tokenize a raw shell command into structured tokens.
///
/// This is a simple shell splitter, not a full POSIX parser. It handles:
/// - Pipes (`|`)
/// - Redirects (`<`, `>`, `>>`)
/// - Flags (`-x`, `--flag`)
/// - Quoted strings (single and double)
/// - Backslash escapes within double quotes
pub fn tokenize(raw: &str) -> Vec<Token> {
    let words = split_shell_words(raw);
    let mut tokens = Vec::with_capacity(words.len());
    let mut expect_command = true;

    for word in &words {
        match word.as_str() {
            "|" => {
                tokens.push(Token::Pipe);
                expect_command = true;
            }
            ">>" => {
                tokens.push(Token::RedirectOut { append: true });
            }
            ">" => {
                tokens.push(Token::RedirectOut { append: false });
            }
            "<" => {
                tokens.push(Token::RedirectIn);
            }
            _ if expect_command => {
                tokens.push(Token::Command(word.clone()));
                expect_command = false;
            }
            _ if word.starts_with('-') => {
                tokens.push(Token::Flag(word.clone()));
            }
            _ => {
                tokens.push(Token::Arg(word.clone()));
            }
        }
    }

    tokens
}

/// Split a raw command string into words, respecting quotes and escapes.
fn split_shell_words(input: &str) -> Vec<String> {
    let mut words = Vec::new();
    let mut current = String::new();
    let mut chars = input.chars().peekable();
    let mut in_single_quote = false;
    let mut in_double_quote = false;

    while let Some(ch) = chars.next() {
        if in_single_quote {
            if ch == '\'' {
                in_single_quote = false;
            } else {
                current.push(ch);
            }
        } else if in_double_quote {
            if ch == '\\' {
                // In double quotes, backslash escapes the next character
                if let Some(&next) = chars.peek() {
                    match next {
                        '"' | '\\' | '$' | '`' => {
                            current.push(chars.next().unwrap_or(ch));
                        }
                        _ => {
                            current.push(ch);
                        }
                    }
                } else {
                    current.push(ch);
                }
            } else if ch == '"' {
                in_double_quote = false;
            } else {
                current.push(ch);
            }
        } else {
            match ch {
                '\'' => {
                    in_single_quote = true;
                }
                '"' => {
                    in_double_quote = true;
                }
                '\\' => {
                    if let Some(next) = chars.next() {
                        current.push(next);
                    }
                }
                ' ' | '\t' | '\n' => {
                    if !current.is_empty() {
                        words.push(std::mem::take(&mut current));
                    }
                }
                '|' => {
                    if !current.is_empty() {
                        words.push(std::mem::take(&mut current));
                    }
                    words.push("|".to_string());
                }
                '>' => {
                    if !current.is_empty() {
                        words.push(std::mem::take(&mut current));
                    }
                    if chars.peek() == Some(&'>') {
                        chars.next();
                        words.push(">>".to_string());
                    } else {
                        words.push(">".to_string());
                    }
                }
                '<' => {
                    if !current.is_empty() {
                        words.push(std::mem::take(&mut current));
                    }
                    words.push("<".to_string());
                }
                _ => {
                    current.push(ch);
                }
            }
        }
    }

    if !current.is_empty() {
        words.push(current);
    }

    words
}

// ===========================================================================
// Embedding generation (deterministic hash-based)
// ===========================================================================

/// Generate a deterministic embedding vector from a raw command string.
///
/// Uses a bag-of-characters approach: each byte contributes to the vector
/// via a hash-scatter pattern. The result is L2-normalized to unit length.
/// This is a placeholder for real ML embeddings — it preserves the property
/// that similar strings produce similar vectors.
fn generate_embedding(raw: &str) -> Vec<f32> {
    let mut vec = vec![0.0f32; EMBED_DIM];

    // Hash each character's byte value into multiple positions using a
    // simple multiplicative hash. This gives us a sparse-ish signal that
    // captures character frequency and position information.
    for (pos, byte) in raw.bytes().enumerate() {
        let b = byte as u64;
        let p = pos as u64;

        // Three hash functions for each character — spreads the signal
        let h1 = b.wrapping_mul(2654435761).wrapping_add(p.wrapping_mul(40503));
        let h2 = b.wrapping_mul(2246822519).wrapping_add(p.wrapping_mul(65599));
        let h3 = b.wrapping_mul(3266489917).wrapping_add(p.wrapping_mul(31));

        let idx1 = (h1 % EMBED_DIM as u64) as usize;
        let idx2 = (h2 % EMBED_DIM as u64) as usize;
        let idx3 = (h3 % EMBED_DIM as u64) as usize;

        // Use the hash bits to determine sign, producing both positive
        // and negative contributions (important for cosine similarity).
        let sign1 = if (h1 >> 32) & 1 == 0 { 1.0 } else { -1.0 };
        let sign2 = if (h2 >> 32) & 1 == 0 { 1.0 } else { -1.0 };
        let sign3 = if (h3 >> 32) & 1 == 0 { 1.0 } else { -1.0 };

        // Position decay: earlier characters matter more (command name
        // is at the front). Decay factor: 1 / (1 + pos * 0.01).
        let decay = 1.0 / (1.0 + pos as f32 * 0.01);

        vec[idx1] += sign1 * decay;
        vec[idx2] += sign2 * decay * 0.7;
        vec[idx3] += sign3 * decay * 0.5;
    }

    // L2-normalize to unit length
    l2_normalize(&mut vec);
    vec
}

/// L2-normalize a vector in-place. If the vector is zero, it remains zero.
fn l2_normalize(vec: &mut [f32]) {
    let norm_sq: f32 = vec.iter().map(|x| x * x).sum();
    if norm_sq > f32::EPSILON {
        let inv_norm = 1.0 / norm_sq.sqrt();
        for x in vec.iter_mut() {
            *x *= inv_norm;
        }
    }
}

// ===========================================================================
// Classifier
// ===========================================================================

/// Exemplar commands for each intent category.
/// The classifier embeds these at construction time and compares incoming
/// commands against them using cosine similarity.
struct CategoryExemplars {
    /// (category, embedding) pairs for all exemplars.
    entries: Vec<(IntentCategory, Vec<f32>)>,
}

impl CategoryExemplars {
    fn new() -> Self {
        let exemplars: &[(&[&str], IntentCategory)] = &[
            (
                &[
                    "cargo build",
                    "cargo build --release",
                    "make",
                    "make all",
                    "gcc -o main main.c",
                    "npm run build",
                    "cmake --build .",
                    "cargo test",
                    "go build ./...",
                    "mvn package",
                    "rustc main.rs",
                ],
                IntentCategory::Build,
            ),
            (
                &[
                    "kubectl apply -f deployment.yaml",
                    "docker push image:latest",
                    "ansible-playbook deploy.yml",
                    "terraform apply",
                    "helm install my-release chart",
                    "systemctl restart nginx",
                    "scp binary server:/opt/",
                    "docker-compose up -d",
                    "flyctl deploy",
                    "git push origin main",
                ],
                IntentCategory::Deploy,
            ),
            (
                &[
                    "gdb ./binary",
                    "strace -p 1234",
                    "ltrace ./binary",
                    "dmesg | tail",
                    "journalctl -u service",
                    "perf record -g ./binary",
                    "valgrind ./binary",
                    "tail -f /var/log/syslog",
                    "tcpdump -i eth0",
                    "cargo test -- --nocapture",
                ],
                IntentCategory::Debug,
            ),
            (
                &[
                    "vim ~/.bashrc",
                    "export PATH=$PATH:/usr/local/bin",
                    "echo 'alias ll=ls -la' >> ~/.bashrc",
                    "git config --global user.name foo",
                    "chmod 755 script.sh",
                    "chown user:group file",
                    "ln -s /usr/bin/python3 /usr/local/bin/python",
                    "sudo sysctl -w net.core.somaxconn=1024",
                    "systemctl enable nginx",
                    "ufw allow 8080",
                ],
                IntentCategory::Configure,
            ),
            (
                &[
                    "grep -r pattern .",
                    "find . -name '*.rs'",
                    "rg TODO",
                    "cat /etc/hosts",
                    "ps aux",
                    "df -h",
                    "du -sh *",
                    "top",
                    "htop",
                    "free -m",
                    "netstat -tlnp",
                    "curl http://localhost:8080/health",
                    "wc -l src/*.rs",
                ],
                IntentCategory::Query,
            ),
            (
                &[
                    "cd /home/user/project",
                    "cd ..",
                    "pushd /tmp",
                    "popd",
                    "ls",
                    "ls -la",
                    "pwd",
                    "tree",
                    "exa -la",
                    "z project",
                ],
                IntentCategory::Navigate,
            ),
        ];

        let mut entries = Vec::new();
        for (commands, category) in exemplars {
            for cmd in *commands {
                let embedding = generate_embedding(cmd);
                entries.push((category.clone(), embedding));
            }
        }

        Self { entries }
    }

    /// Classify a command embedding by finding the category with the highest
    /// average cosine similarity among its top exemplars.
    fn classify(&self, embedding: &[f32]) -> IntentCategory {
        // Accumulate (total_similarity, count) per category
        let mut scores: Vec<(IntentCategory, f32, u32)> = Vec::new();

        for (category, exemplar_emb) in &self.entries {
            let sim = cosine_similarity(embedding, exemplar_emb);

            if let Some(entry) = scores.iter_mut().find(|(c, _, _)| c == category) {
                entry.1 += sim;
                entry.2 += 1;
            } else {
                scores.push((category.clone(), sim, 1));
            }
        }

        // Find category with highest average similarity
        let mut best_category = IntentCategory::Other(String::new());
        let mut best_avg = f32::NEG_INFINITY;

        for (category, total, count) in &scores {
            let avg = *total / *count as f32;
            if avg > best_avg {
                best_avg = avg;
                best_category = category.clone();
            }
        }

        // If the best score is very low, fall back to Other
        if best_avg < 0.05 {
            return IntentCategory::Other(String::new());
        }

        best_category
    }
}

// ===========================================================================
// Context hashing
// ===========================================================================

/// Hash a Context into a u64 for the Intent's context_hash field.
///
/// Uses FNV-1a for speed and good distribution on short strings.
pub fn hash_context(ctx: &Context) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325; // FNV-1a offset basis

    let mix = |h: &mut u64, bytes: &[u8]| {
        for &b in bytes {
            *h ^= b as u64;
            *h = h.wrapping_mul(0x100000001b3); // FNV-1a prime
        }
    };

    mix(&mut h, ctx.cwd.as_bytes());
    mix(&mut h, &ctx.env_hash.to_le_bytes());
    mix(&mut h, &ctx.session_id.to_le_bytes());
    mix(&mut h, ctx.shell.as_bytes());
    mix(&mut h, ctx.user.as_bytes());

    h
}

// ===========================================================================
// IntentPipeline
// ===========================================================================

/// Converts raw command strings into structured Intent objects.
///
/// Implements the `IntentExtractor` trait from tala-core. The pipeline:
/// 1. Tokenize the raw command
/// 2. Generate a hash-based embedding
/// 3. Classify the intent category
/// 4. Assemble the complete Intent struct
pub struct IntentPipeline {
    exemplars: CategoryExemplars,
}

impl IntentPipeline {
    /// Create a new pipeline. Pre-computes exemplar embeddings for classification.
    pub fn new() -> Self {
        Self {
            exemplars: CategoryExemplars::new(),
        }
    }

    /// Tokenize a raw command string.
    pub fn tokenize(&self, raw: &str) -> Vec<Token> {
        tokenize(raw)
    }

    /// Generate an embedding for a raw command.
    pub fn embed(&self, raw: &str) -> Vec<f32> {
        generate_embedding(raw)
    }

    /// Classify a command given its embedding.
    pub fn classify(&self, embedding: &[f32]) -> IntentCategory {
        self.exemplars.classify(embedding)
    }
}

impl Default for IntentPipeline {
    fn default() -> Self {
        Self::new()
    }
}

impl IntentExtractor for IntentPipeline {
    fn extract(&self, raw: &str, context: &Context) -> Result<Intent, TalaError> {
        if raw.trim().is_empty() {
            return Err(TalaError::ExtractionFailed(
                "empty command string".to_string(),
            ));
        }

        // 1. Tokenize (validates parsability)
        let tokens = tokenize(raw);
        if tokens.is_empty() {
            return Err(TalaError::ExtractionFailed(
                "no tokens extracted from command".to_string(),
            ));
        }

        // 2. Generate embedding
        let embedding = generate_embedding(raw);

        // 3. Classify
        let _category = self.exemplars.classify(&embedding);

        // 4. Hash context
        let context_hash = hash_context(context);

        // 5. Assemble
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|e| TalaError::ExtractionFailed(e.to_string()))?
            .as_nanos() as u64;

        Ok(Intent {
            id: IntentId::random(),
            timestamp: now,
            raw_command: raw.to_string(),
            embedding,
            context_hash,
            parent_ids: Vec::new(),
            outcome: None,
            confidence: 1.0,
        })
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Tokenizer tests
    // -----------------------------------------------------------------------

    #[test]
    fn tokenize_simple_command() {
        let tokens = tokenize("ls -la /tmp");
        assert_eq!(
            tokens,
            vec![
                Token::Command("ls".to_string()),
                Token::Flag("-la".to_string()),
                Token::Arg("/tmp".to_string()),
            ]
        );
    }

    #[test]
    fn tokenize_pipe() {
        let tokens = tokenize("cat file.txt | grep error");
        assert_eq!(
            tokens,
            vec![
                Token::Command("cat".to_string()),
                Token::Arg("file.txt".to_string()),
                Token::Pipe,
                Token::Command("grep".to_string()),
                Token::Arg("error".to_string()),
            ]
        );
    }

    #[test]
    fn tokenize_redirects() {
        let tokens = tokenize("sort < input.txt >> output.txt");
        assert_eq!(
            tokens,
            vec![
                Token::Command("sort".to_string()),
                Token::RedirectIn,
                Token::Arg("input.txt".to_string()),
                Token::RedirectOut { append: true },
                Token::Arg("output.txt".to_string()),
            ]
        );
    }

    #[test]
    fn tokenize_quoted_string() {
        let tokens = tokenize("echo \"hello world\"");
        assert_eq!(
            tokens,
            vec![
                Token::Command("echo".to_string()),
                Token::Arg("hello world".to_string()),
            ]
        );
    }

    #[test]
    fn tokenize_single_quotes() {
        let tokens = tokenize("grep 'foo bar' file.txt");
        assert_eq!(
            tokens,
            vec![
                Token::Command("grep".to_string()),
                Token::Arg("foo bar".to_string()),
                Token::Arg("file.txt".to_string()),
            ]
        );
    }

    #[test]
    fn tokenize_long_flags() {
        let tokens = tokenize("cargo build --release --target x86_64-unknown-linux-gnu");
        assert_eq!(
            tokens,
            vec![
                Token::Command("cargo".to_string()),
                Token::Arg("build".to_string()),
                Token::Flag("--release".to_string()),
                Token::Flag("--target".to_string()),
                Token::Arg("x86_64-unknown-linux-gnu".to_string()),
            ]
        );
    }

    #[test]
    fn tokenize_empty_string() {
        let tokens = tokenize("");
        assert!(tokens.is_empty());
    }

    #[test]
    fn tokenize_multiple_pipes() {
        let tokens = tokenize("ps aux | grep rust | wc -l");
        assert_eq!(
            tokens,
            vec![
                Token::Command("ps".to_string()),
                Token::Arg("aux".to_string()),
                Token::Pipe,
                Token::Command("grep".to_string()),
                Token::Arg("rust".to_string()),
                Token::Pipe,
                Token::Command("wc".to_string()),
                Token::Flag("-l".to_string()),
            ]
        );
    }

    #[test]
    fn tokenize_redirect_no_space() {
        let tokens = tokenize("echo hello>out.txt");
        assert_eq!(
            tokens,
            vec![
                Token::Command("echo".to_string()),
                Token::Arg("hello".to_string()),
                Token::RedirectOut { append: false },
                Token::Arg("out.txt".to_string()),
            ]
        );
    }

    // -----------------------------------------------------------------------
    // Embedding tests
    // -----------------------------------------------------------------------

    #[test]
    fn embedding_correct_dimension() {
        let emb = generate_embedding("cargo build");
        assert_eq!(emb.len(), EMBED_DIM);
    }

    #[test]
    fn embedding_is_normalized() {
        let emb = generate_embedding("cargo build --release");
        let norm_sq: f32 = emb.iter().map(|x| x * x).sum();
        assert!(
            (norm_sq - 1.0).abs() < 1e-4,
            "expected unit norm, got {norm_sq}"
        );
    }

    #[test]
    fn embedding_deterministic() {
        let e1 = generate_embedding("ls -la");
        let e2 = generate_embedding("ls -la");
        assert_eq!(e1, e2, "same input must produce same embedding");
    }

    #[test]
    fn similar_commands_similar_embeddings() {
        let e1 = generate_embedding("cargo build");
        let e2 = generate_embedding("cargo build --release");
        let e3 = generate_embedding("kubectl apply -f deploy.yaml");

        let sim_close = cosine_similarity(&e1, &e2);
        let sim_far = cosine_similarity(&e1, &e3);

        assert!(
            sim_close > sim_far,
            "similar commands should have higher similarity: close={sim_close}, far={sim_far}"
        );
    }

    // -----------------------------------------------------------------------
    // Classifier tests
    // -----------------------------------------------------------------------

    #[test]
    fn classify_build_commands() {
        let exemplars = CategoryExemplars::new();
        let emb = generate_embedding("cargo build --release");
        assert_eq!(exemplars.classify(&emb), IntentCategory::Build);
    }

    #[test]
    fn classify_navigate_commands() {
        let exemplars = CategoryExemplars::new();
        let emb = generate_embedding("cd /home/user/project");
        assert_eq!(exemplars.classify(&emb), IntentCategory::Navigate);
    }

    #[test]
    fn classify_query_commands() {
        let exemplars = CategoryExemplars::new();
        let emb = generate_embedding("grep -r TODO src/");
        assert_eq!(exemplars.classify(&emb), IntentCategory::Query);
    }

    #[test]
    fn classify_deploy_commands() {
        let exemplars = CategoryExemplars::new();
        let emb = generate_embedding("kubectl apply -f service.yaml");
        assert_eq!(exemplars.classify(&emb), IntentCategory::Deploy);
    }

    #[test]
    fn classify_debug_commands() {
        let exemplars = CategoryExemplars::new();
        let emb = generate_embedding("strace -p 5678");
        assert_eq!(exemplars.classify(&emb), IntentCategory::Debug);
    }

    #[test]
    fn classify_configure_commands() {
        let exemplars = CategoryExemplars::new();
        let emb = generate_embedding("chmod 755 deploy.sh");
        assert_eq!(exemplars.classify(&emb), IntentCategory::Configure);
    }

    // -----------------------------------------------------------------------
    // Context hashing tests
    // -----------------------------------------------------------------------

    #[test]
    fn context_hash_deterministic() {
        let ctx = Context {
            cwd: "/home/user/project".to_string(),
            env_hash: 12345,
            session_id: 99,
            shell: "zsh".to_string(),
            user: "ops".to_string(),
        };
        let h1 = hash_context(&ctx);
        let h2 = hash_context(&ctx);
        assert_eq!(h1, h2);
    }

    #[test]
    fn context_hash_varies_with_cwd() {
        let ctx1 = Context {
            cwd: "/home/user/a".to_string(),
            ..Default::default()
        };
        let ctx2 = Context {
            cwd: "/home/user/b".to_string(),
            ..Default::default()
        };
        assert_ne!(hash_context(&ctx1), hash_context(&ctx2));
    }

    #[test]
    fn context_hash_varies_with_session() {
        let ctx1 = Context {
            session_id: 1,
            ..Default::default()
        };
        let ctx2 = Context {
            session_id: 2,
            ..Default::default()
        };
        assert_ne!(hash_context(&ctx1), hash_context(&ctx2));
    }

    // -----------------------------------------------------------------------
    // Pipeline (IntentExtractor) tests
    // -----------------------------------------------------------------------

    #[test]
    fn pipeline_extract_basic() {
        let pipeline = IntentPipeline::new();
        let ctx = Context {
            cwd: "/home/user".to_string(),
            env_hash: 0,
            session_id: 1,
            shell: "bash".to_string(),
            user: "testuser".to_string(),
        };

        let intent = pipeline.extract("cargo build --release", &ctx).unwrap();

        assert_eq!(intent.raw_command, "cargo build --release");
        assert_eq!(intent.embedding.len(), EMBED_DIM);
        assert!(intent.context_hash != 0);
        assert!(intent.parent_ids.is_empty());
        assert!(intent.outcome.is_none());
        assert!((intent.confidence - 1.0).abs() < f32::EPSILON);
        assert!(intent.timestamp > 0);
    }

    #[test]
    fn pipeline_extract_empty_fails() {
        let pipeline = IntentPipeline::new();
        let ctx = Context::default();

        let result = pipeline.extract("", &ctx);
        assert!(result.is_err());

        let result = pipeline.extract("   ", &ctx);
        assert!(result.is_err());
    }

    #[test]
    fn pipeline_unique_ids() {
        let pipeline = IntentPipeline::new();
        let ctx = Context::default();

        let i1 = pipeline.extract("ls", &ctx).unwrap();
        let i2 = pipeline.extract("ls", &ctx).unwrap();
        assert_ne!(i1.id, i2.id, "each extraction must produce a unique ID");
    }

    #[test]
    fn pipeline_context_hash_matches() {
        let pipeline = IntentPipeline::new();
        let ctx = Context {
            cwd: "/specific/path".to_string(),
            env_hash: 42,
            session_id: 7,
            shell: "zsh".to_string(),
            user: "ops".to_string(),
        };

        let intent = pipeline.extract("pwd", &ctx).unwrap();
        let expected_hash = hash_context(&ctx);
        assert_eq!(intent.context_hash, expected_hash);
    }

    #[test]
    fn pipeline_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<IntentPipeline>();
    }
}
