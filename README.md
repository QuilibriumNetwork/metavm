# MetaVM

A modular zero-knowledge proof system for proving correct execution of RISC-V, EVM, and Solana BPF programs. MetaVM uses KZG polynomial commitments with support for both BLS48-581 (Quilibrium ceremony) and BLS12-381 (Ethereum KZG) curves, and handles arbitrarily long executions via recursive proof composition with tree-structured folding.

## Architecture

```
metavm/
  crates/
    core/       Field arithmetic, Shamir secret sharing, Fiat-Shamir transcript
    zkp/        Curve-agnostic prover/verifier, KZG commitments, recursive IVC, tree fold
    riscv/      RV64IMAC emulator, ELF loader, Linux boot, execution trace, constraints
    evm/        EVM executor (revm), U256 limb decomposition, EVM constraints
    sbf/        Solana BPF executor (solana-rbpf), SBF constraints
```

## Proof System

### Polynomial IOP

Each VM execution trace is represented as a table of columns (registers, memory, selectors) evaluated over a multiplicative subgroup of a prime field. The prover commits to each column polynomial via KZG, then constructs a combined constraint polynomial:

```
C(X) = sum_{k} alpha^k * selector_k(X) * body_k(X)
```

where `alpha` is a Fiat-Shamir challenge derived from column commitments, each `selector_k` is a one-hot indicator polynomial for an instruction type, and `body_k` encodes the correctness relation for that instruction. The prover divides by the vanishing polynomial `Z(X) = X^n - 1` to obtain the quotient `Q(X) = C(X) / Z(X)`, commits to quotient chunks, and opens all polynomials at a random evaluation point `z` via a batch KZG proof.

The verifier reconstructs C(z) from the claimed column evaluations, checks `Q(z) * Z(z) = C(z)`, and verifies the batch KZG opening with a single pairing check.

### Cross-Row Constraints

PC continuity (`pc(omega*X) - next_pc(X) = 0`) is enforced via shifted polynomial evaluations at `omega*z`, where omega is the domain generator. A second batch opening proof covers the shifted evaluations.

### Constraint Systems

| VM | Columns | Constraints | Selectors | Oracle Selectors |
|----|---------|-------------|-----------|------------------|
| RISC-V (RV64IMAC) | 84 (20 data + 64 selectors) | 128 + 1 shifted | 64 | 0 |
| EVM | 61 (20 data + 41 selectors) | 106 + 1 shifted | 41 | 13 |
| SBF | 39 (17 data + 22 selectors) | 47 + 1 shifted | 22 | 1 |

Oracle selectors mark operations verified externally (complex opcodes like KECCAK, storage, calls). Their execution data is bound to the Fiat-Shamir transcript as public input.

### Auxiliary Arguments

- **LogUp range checks**: Byte decomposition of witness columns with running sum accumulation
- **Memory permutation**: Grand-product argument over (address, value, timestamp, rw) tuples sorted by (address, timestamp), enforcing read consistency
- **Register file permutation**: Multi-port grand product (3 ports for RISC-V, 2 for SBF) with sorted lanes per register access

### Recursive Proof Composition

Long executions are split into fixed-size chunks. Each chunk produces an independent KZG proof. Chunk proofs are aggregated via a binary tree fold that accumulates pairing arguments:

```
L_combined = L_left + r * L_right
R_combined = R_left + r * R_right
```

The final proof is a single accumulated claim verified by one pairing: `e(L_acc, G2) == e(R_acc, [tau]_2)`. State hash chains ensure chunk continuity: each chunk's final state hash must equal the next chunk's initial state hash.

### Supported Curves

| Curve | Source | G1 Size | Scalar Size | Max Domain |
|-------|--------|---------|-------------|------------|
| BLS48-581 | Quilibrium ceremony SRS | 74 bytes | 73 bytes | 256 |
| BLS12-381 | Ethereum KZG ceremony | 48 bytes | 32 bytes | 4096 |

BLS12-381 uses NTT-based polynomial multiplication (O(n log n)) for domains >= 64. BLS48-581 falls back to naive convolution. Both use Pippenger MSM for commitments. The `bls48581-fast` variant uses optimized FFT with precomputed roots of unity.

## Building

Requires the `ceremonyclient` crates at `../ceremonyclient/crates/` relative to this workspace.

```bash
# Build all crates
cargo build --release

# Build a specific binary
cargo build --release --bin prove-elf
cargo build --release --bin prove-boot
cargo build --release --bin prove-evm
cargo build --release --bin prove-block
cargo build --release --bin prove-sbf
cargo build --release --bin prove-slot
```

## Usage

All proving binaries support `--scheme bls12381|bls48581|bls48581-fast` (default: `bls12381`).

### Prove a RISC-V ELF Binary

Compile a bare-metal RISC-V program (no OS, no floating point):

```bash
# Install the RISC-V toolchain
# On macOS: brew install riscv-gnu-toolchain
# On Linux: apt install gcc-riscv64-linux-gnu

# Compile a bare-metal program
riscv64-unknown-elf-gcc -march=rv64imac -mabi=lp64 -nostdlib -static \
    -Wl,-Ttext=0x80000000 -o program.elf program.c

# Prove execution
cargo run --release --bin prove-elf -- program.elf \
    --chunk-size 128 \
    --workers 8 \
    --output proof.bin
```

The ELF runs in unprivileged mode with no devices. Execution halts on ECALL (syscall 0 or 93). The exit code is returned in register `a0`.

Options:
- `--chunk-size N` — Steps per proof chunk (default: 128, max: scheme's SRS limit)
- `--max-steps N` — Stop execution after N steps (0 = unlimited)
- `--workers N` — Parallel proving threads (default: available CPUs)
- `--output FILE` — Output proof file (default: `elf_proof.bin`)

### Prove a Linux Boot

Boot a Linux kernel with OpenSBI and an optional initramfs:

```bash
# Build kernel + OpenSBI (see riscv-build/Dockerfile)
# The kernel must be compiled for rv64imac (no FPU)
# Output: fw_payload.elf (OpenSBI + Linux), initrd.gz

cargo run --release --bin prove-boot -- \
    fw_payload.elf initrd.gz \
    --chunk-size 128 \
    --workers 8 \
    --trace-file boot_trace.bin \
    --output boot_proof.bin
```

The VM runs in M-mode with full device emulation: CLINT (timer/IPI at 0x0200_0000), PLIC (interrupts at 0x0C00_0000), UART 16550A (serial at 0x1000_0000), and VirtIO block (at 0x1000_1000). UART output is printed to stderr in real time.

The `--trace-file` option serializes execution chunks to disk during the execution phase, allowing proving to proceed from disk rather than holding all chunks in memory. This is critical for long boots that produce millions of chunks.

Press Ctrl+C once to stop execution and proceed to proving. Press Ctrl+C again to abort.

Additional option:
- `--max-steps N` — Stop execution after N steps

### Prove an Ethereum Transaction

```bash
cargo run --release --bin prove-evm
```

Currently runs a built-in demo (counter contract: SLOAD, ADD 1, SSTORE, STOP). Produces a single-chunk proof and verifies it.

### Prove an Ethereum Block

Replay all transactions in a block from an RPC endpoint and produce a ZK proof:

```bash
cargo run --release --bin prove-block -- \
    https://eth.llamarpc.com 19000000 \
    --chunk-size 256 \
    --workers 8 \
    --output block_proof.bin
```

The binary fetches the block, replays each transaction with a tracing inspector to capture the EVM execution trace, merges all traces, chunks them, and proves in parallel with tree folding. Supports legacy, EIP-2930, EIP-1559, and EIP-4844 transaction types.

Options:
- `--chunk-size N` — Trace rows per proof chunk (default: 256)
- `--workers N` — Parallel proving threads (default: available CPUs)
- `--output FILE` — Output proof file (default: `block_proof.bin`)

### Prove a Solana BPF Program

```bash
# From an ELF file
cargo run --release --bin prove-sbf -- program.so

# From inline assembly
cargo run --release --bin prove-sbf -- --asm "mov64 r1, 10
mov64 r2, 20
add64 r1, r2
mov64 r0, r1
exit"
```

Without arguments, runs a default demo (r0 = 10 + 20). Produces a single-chunk proof.

Options:
- `--output FILE` — Output proof file (default: `sbf_proof.bin`)

### Prove a Solana Slot

Fetch a Solana slot, download BPF programs, execute each transaction, and prove:

```bash
cargo run --release --bin prove-slot -- \
    https://api.mainnet-beta.solana.com 250000000 \
    --chunk-size 256 \
    --workers 8 \
    --output slot_proof.bin
```

The binary fetches the slot, downloads each referenced BPF program ELF and account data (with caching to avoid refetches), serializes the BPF input, executes with tracing, and proves each transaction. Single-chunk transactions skip the worker pool overhead; multi-chunk transactions use parallel proving with tree folding.

Options:
- `--chunk-size N` — Trace rows per proof chunk (default: 256)
- `--workers N` — Parallel proving threads (default: available CPUs)
- `--output FILE` — Output proof file (default: `slot_proof.bin`)

## Tests

```bash
cargo test                    # All 530 tests
cargo test -p metavm-core     # 22 tests  (field, share, transcript)
cargo test -p metavm-zkp      # 98 tests  (commitment, polynomial, prover, verifier, recursive)
cargo test -p metavm-riscv    # 328 tests (decoder, CPU, memory, VM, trace, CSR, MMU, MMIO, constraints)
cargo test -p metavm-evm      # 45 tests  (executor, trace, constraints, prove/verify)
cargo test -p metavm-sbf      # 37 tests  (executor, trace, constraints, prove/verify)
```
