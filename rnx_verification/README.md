# RNX: Unified Sub-Symbolic Reasoning Neural Architecture
## Executable Mathematical Verification Harness

This repository contains the standalone verification harness and mathematical proof code for **RNX: A Mathematics Model of Unified Sub-Symbolic Reasoning for the Sub-Symbolic Memory Programme**.

### Modules:
- `rnx_core.py`: Implementation of the continuous reasoning dynamical system, spectral contraction normalization, and 7 unified reasoning operators (Deductive, Inductive, Abductive, Analogical, Causal, Probabilistic, Commonsense).
- `rnx_memory.py`: Dual-memory coupling interfaces with Working Memory (WMX) and Long-Term Memory (LTX), including continuous thought trajectory beam search.
- `rnx_storage.py`: Disk-backed multimodal storage engine implementing 4096-byte slotted binary pages, two-bank crash-consistent commits, and CRC32 payload protection.
- `rnx_learning.py`: Analytical reverse-mode autodiff and finite-difference gradient verification harness.
- `rnx_complexity.py`: Empirical timing benchmarks verifying the sub-quadratic complexity contract ($O(1), O(\log n), O(n)$).
- `rnx_harness.py`: Central test suite running 16 end-to-end mathematical verification checks.

### Running the Verification Suite:
```bash
python3 rnx_harness.py
```
Expected output: All 16 checks pass with 0 errors.
