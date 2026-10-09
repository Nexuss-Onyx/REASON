"""
RNX Algorithmic Complexity Benchmark and Log-Log Verification
Empirically audits that all runtime profiles satisfy the strict complexity contract:
{O(1), O(log n), O(n)} with zero quadratic terms.
"""

import time
import numpy as np
from rnx_core import RNXReasoningCore
from rnx_memory import LTXHierarchicalRoutingMemory, WMXWorkingMemoryInterface

def benchmark_trajectory_scaling(d: int = 64, steps_list = [4, 8, 16, 32, 64], repeats: int = 20):
    core = RNXReasoningCore(d=d)
    x = np.random.randn(d).astype(np.float32)
    m_wm = np.random.randn(d).astype(np.float32)
    m_ltx = np.random.randn(d).astype(np.float32)
    c = np.ones(7, dtype=np.float32) / 7.0
    aux = {"abductive_target": np.random.randn(d).astype(np.float32)}

    timings = []
    for K in steps_list:
        # Warmup
        z = np.random.randn(d).astype(np.float32)
        for _ in range(K):
            z = core.step(z, x, m_wm, m_ltx, c, aux)
        
        # Timing
        t0 = time.perf_counter()
        for _ in range(repeats):
            z = np.random.randn(d).astype(np.float32)
            for _ in range(K):
                z = core.step(z, x, m_wm, m_ltx, c, aux)
        elapsed = (time.perf_counter() - t0) / repeats
        timings.append(elapsed)

    # Log-Log fit: log(T) = slope * log(K) + intercept
    log_k = np.log(steps_list)
    log_t = np.log(timings)
    poly = np.polyfit(log_k, log_t, 1)
    slope = poly[0]
    
    # R^2 correlation
    y_pred = np.polyval(poly, log_k)
    r2 = 1.0 - np.sum((log_t - y_pred)**2) / np.sum((log_t - np.mean(log_t))**2)

    return steps_list, timings, slope, r2

def benchmark_ltx_routing_scaling(d: int = 64, capacities = [64, 256, 1024, 4096], repeats: int = 50):
    query = np.random.randn(d).astype(np.float32)
    timings = []
    depths = []
    
    for cap in capacities:
        ltx = LTXHierarchicalRoutingMemory(capacity_N=cap, d=d)
        depths.append(ltx.depth)
        
        # Warmup
        _ = ltx.route_and_read(query)

        t0 = time.perf_counter()
        for _ in range(repeats):
            _ = ltx.route_and_read(query)
        elapsed = (time.perf_counter() - t0) / repeats
        timings.append(elapsed)

    # Linear fit against log2(capacity)
    log_cap = np.log2(capacities)
    poly = np.polyfit(log_cap, timings, 1)
    y_pred = np.polyval(poly, log_cap)
    r2 = 1.0 - np.sum((timings - y_pred)**2) / np.sum((timings - np.mean(timings))**2)

    return capacities, timings, r2, poly[0]
