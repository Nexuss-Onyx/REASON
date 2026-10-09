# REASON — RNX: A Mathematics Model of Unified Sub-Symbolic Reasoning

> **Sub-symbolic Memory Programme | Mathematical Foundations of Intelligence**  
> Companion Papers: SMX (Sensory Memory), STX (Short-Term Memory), WMX (Working Memory), LTX (Disk-Backed Long-Term Memory), SLX (Supervised Learning), ULX (Unsupervised Learning), FLO-RL (Reinforcement Learning), FBAL (Active Learning), CLX (Continual Learning), DASP-Meta (Meta-Learning).

---

## Overview

**RNX** is a mathematically unified, sub-symbolic neural reasoning architecture designed as the core reasoning engine for Artificial General Intelligence (AGI). Unlike symbolic systems that rely on discrete rules and exponential search ($O(N!)$), or dense connectionist models that incur quadratic complexity ($O(n^2)$), RNX formulates general reasoning as a **contractive dynamical trajectory on continuous Riemannian Banach manifolds**.

Every premise, relation, hypothesis, and inference step is represented as a real-valued tensor $z \in \mathbb{R}^d$ bounded within the unit ball $\mathcal{B}_1^d$.

### Unified Across 7 Canonical Reasoning Disciplines
A single parameterized recurrent backbone conditioned on a low-dimensional task simplex $c \in \Delta^6$ executes:
1. **Deductive Reasoning**: Continuous Horn constraint manifold projection in $\mathcal{O}(J \cdot d)$ (bypassing exponential Boolean SAT).
2. **Inductive Reasoning**: Streaming Fréchet invariant extraction and covariance contraction in $\mathcal{O}(d)$.
3. **Abductive Reasoning**: Adjoint latent cause inversion via matrix-free vector-Jacobian products (VJP) in $\mathcal{O}(d)$.
4. **Analogical Reasoning**: Unitary Holographic Reduced Representations (HRR) and Fourier deconvolution in $\mathcal{O}(d \log d)$.
5. **Causal Reasoning**: Continuous orthogonal $do(X = x_{\text{do}})$ projection operators guaranteeing exact d-separation in $\mathcal{O}(d)$.
6. **Probabilistic Reasoning**: Langevin drift-diffusion on a variational free-energy landscape in $\mathcal{O}(d)$.
7. **Commonsense Reasoning**: Fast-Slow dual-system dynamic arbitration (System 1 vs System 2) in $\mathcal{O}(d)$.

---

## Strict Complexity Contract

$$\text{Time Complexity} \in \{\mathcal{O}(1), \mathcal{O}(\log n), \mathcal{O}(n)\}, \quad \text{Space Complexity} \in \{\mathcal{O}(1), \mathcal{O}(\log n), \mathcal{O}(n)\}$$

- **No $\mathcal{O}(n^2)$ attention matrices or dense pairwise Gram matrices.**
- **Working Memory (WMX)**: Soft-gated associative readout over $M=16$ slots in $\mathcal{O}(d)$.
- **Long-Term Memory (LTX)**: Binary hierarchical routing tree over $N$ items in $\mathcal{O}(\log N \cdot d)$.
- **Continuous Thought Search**: Beam search of width $B$ across $K$ steps in $\mathcal{O}(K \cdot d)$.

---

## Hardware Storage Engine

- **RAM ceiling**: $16\text{ MiB}$ (4,096 pages)
- **Disk quota**: $64\text{ MiB}$ (16,384 pages)
- **Binary Page Format**: 4096-byte slotted pages with 32-byte header, magic `0x524E5831` (`RNX1`), generation counter, modality tags, and CRC32 payload protection.
- **Two-Bank Atomic Commit**: Crash-resilient alternating superblock commits with automatic rollback.

---

## Deliverables in this Repository

- `RNX_Reasoning_Paper.pdf`: The complete 26-page academic paper with formal theorems, analytic proofs, and translation specifications.
- `RNX_Reasoning_Paper.md`: Full Markdown source with all LaTeX mathematical formulations.
- `RNX_Reasoning_Verification_Harness.zip`: Standalone verification archive.
- `rnx_verification/`: Executable Python/NumPy verification codebase.

---

## Running the Verification Suite

```bash
cd rnx_verification
python3 rnx_harness.py
```

### Verification Results (16/16 Passed)
```
[PASS] 01_contractive_convergence: rho=0.4058, init_diff=2.9273e+00, final_diff=6.4522e-06
[PASS] 02_deductive_projection: init_viol=12.2529, final_viol=0.0000
[PASS] 03_inductive_prototype: L2 error to ground truth mean = 4.7546e-02
[PASS] 04_abductive_inversion: init_residual=1.1200, final_residual=0.0824
[PASS] 05_analogical_binding: cosine_similarity=0.9982 (expected > 0.80 at d=512)
[PASS] 06_causal_intervention: intervened_err=0.0e+00, untouched_err=0.0e+00
[PASS] 07_probabilistic_dissipation: init_energy=144.0000, final_energy=0.0017
[PASS] 08_commonsense_routing: gate_fast=0.0007, gate_slow=0.9993
[PASS] 09_trajectory_linearity: log-log slope=0.9577, R2=0.9994
[PASS] 10_ltx_log_routing: actual_depth=11, expected=11, R2=0.9977
[PASS] 11_wmx_bounded_slots: norm_read=0.2811 (<= 1.0), slot0_norm=1.0000
[PASS] 12_gradient_exactness: max_rel_error=2.2375e-08 (threshold 1e-6)
[PASS] 13_storage_serializer: max_abs_error=0.0e+00
[PASS] 14_crc32_fault_detection: Corrupted bit triggered IOError as expected
[PASS] 15_twobank_recovery: recovered_bank=1, verified content identical
[PASS] 16_banach_ball_stability: max_norm=0.9999 (bounded <= 1.0)
```

---

## C++20 / Rust Translation Contract
See Section 14 of `RNX_Reasoning_Paper.pdf` for zero-allocation memory layouts, 64-byte alignment contracts, and AVX-512 vectorization specifications.
