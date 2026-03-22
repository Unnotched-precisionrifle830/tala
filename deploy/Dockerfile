FROM rust:1.82-bookworm AS builder
WORKDIR /build
COPY Cargo.toml Cargo.lock rust-toolchain.toml ./
COPY crates/ crates/
RUN cargo build --release --bin tala-sim

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=builder /build/target/release/tala-sim /usr/local/bin/tala-sim
COPY media/ /opt/tala/media/
RUN mkdir -p /data
ENTRYPOINT ["tala-sim"]
