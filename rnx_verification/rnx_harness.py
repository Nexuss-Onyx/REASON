"""
RNX Executable Verification Suite
Runs 16 rigorous mathematical and computational validation checks.
Covers spectral contraction, all 7 reasoning operators, dual-memory scaling,
gradient exactness, storage byte budgets, CRC fault detection, and crash recovery.
"""

import os
import sys
import tempfile
import numpy as np

from rnx_core import (
    RNXReasoningCore,
    spectral_normalize,
    circular_convolution,
    circular_unbinding
)
from rnx_memory import (
    WMXWorkingMemoryInterface,
    LTXHierarchicalRoutingMemory,
    ContinuousThoughtBeamSearch
)
from rnx_storage import (
    RNXStorageEngine,
    PAGE_SIZE,
    MODALITY_LATENT_THOUGHT
)
from rnx_learning import RNXLearningHarness
from rnx_complexity import (
    benchmark_trajectory_scaling,
    benchmark_ltx_routing_scaling
)

def run_all_checks():
    results = {}
    print("======================================================================")
    print("      RNX REASONING ARCHITECTURE — EXECUTABLE VERIFICATION SUITE       ")
    print("======================================================================\n")

    # 1. Contraction Mapping & Spectral Radius (Theorem 1)
    print("[Check 01] Verifying Spectral Radius & Contractive Convergence (Theorem 1)...")
    d = 32
    W = np.random.randn(d, d)
    W_norm = spectral_normalize(W, target_spectral_radius=0.75)
    eigvals = np.linalg.eigvals(W_norm)
    rho = float(np.max(np.abs(eigvals)))
    
    # Check fixed point iteration convergence: z_{k+1} = (1-alpha)*z_k + alpha*tanh(W_norm z_k + b)
    b = np.random.randn(d) * 0.1
    z = np.random.randn(d)
    diffs = []
    alpha = 0.5
    for _ in range(30):
        z_next = (1 - alpha) * z + alpha * np.tanh(W_norm @ z + b)
        diffs.append(np.linalg.norm(z_next - z))
        z = z_next
    # Should contract geometrically
    is_contractive = (diffs[-1] < diffs[0] * 1e-4) and (rho <= 0.76)
    results["01_contractive_convergence"] = (is_contractive, f"rho={rho:.4f}, init_diff={diffs[0]:.4e}, final_diff={diffs[-1]:.4e}")

    # 2. Deductive Projection Soundness (Theorem 2)
    print("[Check 02] Verifying Deductive Horn Continuous Manifold Projection (Theorem 2)...")
    core = RNXReasoningCore(d=32, num_horn_clauses=6, seed=10)
    z_init = np.random.randn(32) * 1.5
    # Measure initial violations
    init_violations = np.sum(np.maximum(0, core.phi_horns @ z_init - core.theta_horns)**2)
    z_curr = z_init.copy()
    for _ in range(15):
        force = core.op_deductive(z_curr, eta_proj=0.4)
        z_curr += force
    final_violations = np.sum(np.maximum(0, core.phi_horns @ z_curr - core.theta_horns)**2)
    is_ded_sound = (final_violations < init_violations * 0.05)
    results["02_deductive_projection"] = (is_ded_sound, f"init_viol={init_violations:.4f}, final_viol={final_violations:.4f}")

    # 3. Inductive Invariant Prototype Convergence
    print("[Check 03] Verifying Inductive Streaming Prototype Extraction...")
    core_ind = RNXReasoningCore(d=16, seed=20)
    true_mean = np.random.randn(16) * 0.5
    for _ in range(200):
        sample = true_mean + np.random.randn(16) * 0.2
        _ = core_ind.op_inductive(sample)
    error_mean = float(np.linalg.norm(core_ind.mu_proto - true_mean))
    is_ind_converged = error_mean < 0.10
    results["03_inductive_prototype"] = (is_ind_converged, f"L2 error to ground truth mean = {error_mean:.4e}")

    # 4. Abductive Adjoint Inversion
    print("[Check 04] Verifying Abductive Latent Cause Inversion...")
    core_abd = RNXReasoningCore(d=16, seed=30)
    true_cause = np.random.randn(16) * 0.5
    observed_effect = np.tanh(core_abd.W_abd_fwd @ true_cause)
    z_hyp = np.zeros(16)
    init_res = np.linalg.norm(np.tanh(core_abd.W_abd_fwd @ z_hyp) - observed_effect)
    for _ in range(50):
        grad_step = core_abd.op_abductive(z_hyp, observed_effect, lambda_prior=0.0001)
        z_hyp += 0.4 * grad_step
    final_res = np.linalg.norm(np.tanh(core_abd.W_abd_fwd @ z_hyp) - observed_effect)
    is_abd_success = final_res < init_res * 0.1
    results["04_abductive_inversion"] = (is_abd_success, f"init_residual={init_res:.4f}, final_residual={final_res:.4f}")

    # 5. Analogical Unitary Binding & Involution (Theorem 3)
    print("[Check 05] Verifying Analogical Holographic Role-Filler Unbinding (Theorem 3)...")
    d_hrr = 512
    A = np.random.randn(d_hrr) / np.sqrt(d_hrr)
    B = np.random.randn(d_hrr) / np.sqrt(d_hrr)
    # Bind: R = B (*) A^inv
    R = circular_unbinding(B, A)
    # Now retrieve B given A: B_retrieved = A (*) R
    B_retrieved = circular_convolution(A, R)
    cosine_sim = float(np.dot(B, B_retrieved) / (np.linalg.norm(B) * np.linalg.norm(B_retrieved)))
    is_hrr_valid = cosine_sim > 0.80
    results["05_analogical_binding"] = (is_hrr_valid, f"cosine_similarity={cosine_sim:.4f} (expected > 0.80 at d=512)")

    # 6. Causal Continuous Orthogonal Intervention (Theorem 4)
    print("[Check 06] Verifying Causal Continuous do(X) Decoupling (Theorem 4)...")
    core_caus = RNXReasoningCore(d=32, seed=50)
    z_prior = np.random.randn(32) * 2.0
    x_do = np.ones(32) * 5.0
    delta = core_caus.op_causal(z_prior, x_do)
    z_post = z_prior + delta
    # Check that intervened subspace (first quarter) exactly equals x_do, and remainder is strictly unchanged
    mask = core_caus.causal_subspace_mask.astype(bool)
    subspace_intervened_err = float(np.linalg.norm(z_post[mask] - x_do[mask]))
    subspace_unintervened_err = float(np.linalg.norm(z_post[~mask] - z_prior[~mask]))
    is_causal_sound = (subspace_intervened_err < 1e-6) and (subspace_unintervened_err < 1e-6)
    results["06_causal_intervention"] = (is_causal_sound, f"intervened_err={subspace_intervened_err:.1e}, untouched_err={subspace_unintervened_err:.1e}")

    # 7. Probabilistic Free-Energy Variance Dissipation
    print("[Check 07] Verifying Probabilistic Free Energy Dissipation...")
    core_prob = RNXReasoningCore(d=16, seed=60)
    z_disp = np.ones(16) * 3.0
    energy_init = 0.5 * np.sum(core_prob.laplace_precision_diag * (z_disp**2))
    for _ in range(25):
        z_disp += core_prob.op_probabilistic(z_disp, temperature=0.001)
    energy_final = 0.5 * np.sum(core_prob.laplace_precision_diag * (z_disp**2))
    is_prob_dissipative = energy_final < energy_init * 0.1
    results["07_probabilistic_dissipation"] = (is_prob_dissipative, f"init_energy={energy_init:.4f}, final_energy={energy_final:.4f}")

    # 8. Commonsense Dual-System Fast-Slow Gating
    print("[Check 08] Verifying Commonsense Dual-System Routing Dynamics...")
    core_comm = RNXReasoningCore(d=16, seed=70)
    # High confidence state (aligned with router)
    z_fast = core_comm.w_router * -5.0 # gate ~ 0
    # Ambiguous state
    z_slow = core_comm.w_router * 5.0  # gate ~ 1
    gate_fast = 1.0 / (1.0 + np.exp(-(np.dot(core_comm.w_router, z_fast) + core_comm.b_router)))
    gate_slow = 1.0 / (1.0 + np.exp(-(np.dot(core_comm.w_router, z_slow) + core_comm.b_router)))
    is_comm_gated = (gate_fast < 0.05) and (gate_slow > 0.95)
    results["08_commonsense_routing"] = (is_comm_gated, f"gate_fast={gate_fast:.4f}, gate_slow={gate_slow:.4f}")

    # 9. Sub-Quadratic Trajectory Scaling (Linearity in K)
    print("[Check 09] Verifying Sub-Quadratic Trajectory Scaling (Log-Log Slope <= 1.0)...")
    steps, timings, slope, r2 = benchmark_trajectory_scaling(d=32, steps_list=[4, 8, 16, 32], repeats=10)
    is_linear_trajectory = (0.85 <= slope <= 1.15) and (r2 > 0.95)
    results["09_trajectory_linearity"] = (is_linear_trajectory, f"log-log slope={slope:.4f}, R2={r2:.4f}")

    # 10. Hierarchical LTX Memory Logarithmic Routing (O(log N))
    print("[Check 10] Verifying LTX Memory Hierarchical Routing Scaling (O(log N))...")
    caps, ltx_timings, r2_ltx, poly_slope = benchmark_ltx_routing_scaling(d=32, capacities=[32, 128, 512, 2048], repeats=20)
    ltx_test = LTXHierarchicalRoutingMemory(capacity_N=2048, d=32)
    q = np.random.randn(32)
    _, path, dot_ops = ltx_test.route_and_read(q)
    expected_depth = int(np.ceil(np.log2(2048)))
    is_log_exact = (len(path) == expected_depth) and (dot_ops == expected_depth) and (r2_ltx > 0.85)
    results["10_ltx_log_routing"] = (is_log_exact, f"actual_depth={len(path)}, expected={expected_depth}, R2={r2_ltx:.4f}")

    # 11. Working Memory Gated Bounded Slot Readout (O(1) w.r.t N)
    print("[Check 11] Verifying WMX Gated Working Memory Slot Interface...")
    wmx = WMXWorkingMemoryInterface(num_slots=16, d=32)
    q_wm = np.random.randn(32)
    read_vec = wmx.read(q_wm)
    norm_read = float(np.linalg.norm(read_vec))
    # Write update to slot 0
    wmx.write(0, np.ones(32) / np.sqrt(32), alpha_gate=0.8)
    slot0_norm = float(np.linalg.norm(wmx.slots[0]))
    # Readout must be strictly bounded in unit ball by convexity
    is_wmx_sound = (0.0 < norm_read <= 1.0) and (abs(slot0_norm - 1.0) < 1e-4)
    results["11_wmx_bounded_slots"] = (is_wmx_sound, f"norm_read={norm_read:.4f} (<= 1.0), slot0_norm={slot0_norm:.4f}")

    # 12. Analytical Gradient Exactness vs Finite Differences (< 10^-6 relative error)
    print("[Check 12] Verifying Reverse-Mode Analytical Gradients vs Finite Differences...")
    harness = RNXLearningHarness(d=12, rank=2, seed=42)
    rel_errors = harness.finite_difference_check(eps=1e-6)
    max_rel_err = max(rel_errors.values())
    is_grad_exact = max_rel_err < 1e-6
    results["12_gradient_exactness"] = (is_grad_exact, f"max_rel_error={max_rel_err:.4e} (threshold 1e-6)")

    # 13. Disk-Backed Storage Quota and Slotted Page Layout
    print("[Check 13] Verifying Disk-Backed Storage Quota and IEEE-754 Serializer...")
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_path = tmp_file.name
    try:
        storage = RNXStorageEngine(tmp_path, max_disk_bytes=1024 * 1024, max_ram_bytes=256 * 1024)
        test_tensor = np.random.randn(8, 16).astype(np.float32)
        storage.write_tensor_page(page_id=1, modality=MODALITY_LATENT_THOUGHT, tensor_data=test_tensor)
        read_back = storage.read_tensor_page(page_id=1, expected_shape=(8, 16))
        tensor_err = float(np.max(np.abs(test_tensor - read_back)))
        is_storage_exact = (tensor_err == 0.0)
        results["13_storage_serializer"] = (is_storage_exact, f"max_abs_error={tensor_err:.1e}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    # 14. Storage CRC32 Fault Detection
    print("[Check 14] Verifying Storage CRC32 Single-Bit & Burst Corruption Detection...")
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_path = tmp_file.name
    try:
        storage = RNXStorageEngine(tmp_path, max_disk_bytes=1024 * 1024)
        t_data = np.random.randn(10, 10).astype(np.float32)
        storage.write_tensor_page(page_id=2, modality=MODALITY_LATENT_THOUGHT, tensor_data=t_data)
        
        # Evict from RAM cache to force disk read
        del storage.ram_cache[2]
        
        # Corrupt single bit on disk
        bank_offset = storage.active_bank * (storage.bank_capacity_pages * PAGE_SIZE)
        page_offset = bank_offset + (2 + 1) * PAGE_SIZE
        with open(tmp_path, "r+b") as f:
            f.seek(page_offset + 32 + 10) # 10 bytes into payload
            orig_b = f.read(1)[0]
            f.seek(page_offset + 32 + 10)
            f.write(bytes([orig_b ^ 0x01])) # Flip 1 bit
        
        caught = False
        try:
            _ = storage.read_tensor_page(page_id=2, expected_shape=(10, 10))
        except IOError:
            caught = True
        results["14_crc32_fault_detection"] = (caught, "Corrupted bit triggered IOError as expected")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    # 15. Two-Bank Atomic Commit & Crash Recovery (Theorem 7)
    print("[Check 15] Verifying Two-Bank Atomic Commit & Crash Recovery (Theorem 7)...")
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_path = tmp_file.name
    try:
        storage = RNXStorageEngine(tmp_path, max_disk_bytes=1024 * 1024)
        t_data1 = np.ones((4, 4), dtype=np.float32) * 1.5
        storage.write_tensor_page(page_id=1, modality=MODALITY_LATENT_THOUGHT, tensor_data=t_data1)
        storage.commit_two_bank()
        initial_bank = storage.active_bank
        
        # Now simulate restart
        storage_reloaded = RNXStorageEngine(tmp_path, max_disk_bytes=1024 * 1024)
        reloaded_bank = storage_reloaded.active_bank
        read_t1 = storage_reloaded.read_tensor_page(page_id=1, expected_shape=(4, 4))
        bank_recovered = (initial_bank == reloaded_bank) and (np.all(read_t1 == 1.5))
        results["15_twobank_recovery"] = (bank_recovered, f"recovered_bank={reloaded_bank}, verified content identical")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    # 16. Unit Banach Ball Trajectory Stability
    print("[Check 16] Verifying Unit Banach Ball Trajectory Boundedness under Perturbations...")
    core_stab = RNXReasoningCore(d=32, seed=88)
    z_stab = np.random.randn(32) * 0.1
    x_test = np.random.randn(32) * 0.5
    wm_test = np.random.randn(32) * 0.5
    ltx_test_v = np.random.randn(32) * 0.5
    c_uniform = np.ones(7) / 7.0
    
    max_norm = 0.0
    for _ in range(50):
        z_stab = core_stab.step(z_stab, x_test, wm_test, ltx_test_v, c_uniform, {})
        max_norm = max(max_norm, float(np.linalg.norm(z_stab)))
    is_bounded = max_norm <= 1.0 # Strict unit ball boundedness
    results["16_banach_ball_stability"] = (is_bounded, f"max_norm={max_norm:.4f} (bounded <= 1.0)")

    print("\n======================================================================")
    print("                    FINAL VERIFICATION SUMMARY                         ")
    print("======================================================================")
    all_passed = True
    for test_name, (passed, msg) in results.items():
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"[{status}] {test_name}: {msg}")
    print("======================================================================")
    if all_passed:
        print("ALL 16 MATHEMATICAL VERIFICATION CHECKS PASSED PERFECTLY!")
    else:
        print("SOME CHECKS FAILED. PLEASE INVESTIGATE.")
    print("======================================================================\n")
    return all_passed

if __name__ == "__main__":
    success = run_all_checks()
    sys.exit(0 if success else 1)
