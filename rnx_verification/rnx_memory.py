"""
RNX Dual-Memory Coupling and Hierarchical Routing
Interfaces with WMX (Working Memory) and LTX (Long-Term Memory).
Guarantees O(1) / O(d) for working memory and O(log N * d) for hierarchical LTX routing.
Strictly prohibits any O(N^2) pairwise attention or full Gram matrix computations.
"""

import numpy as np

class WMXWorkingMemoryInterface:
    """
    Sub-symbolic Working Memory Coupling (WMX style).
    Maintains a bounded bank of M slots (default M=16).
    Readout via soft-gated attention over M slots: O(M * d) = O(d) since M is fixed.
    """
    def __init__(self, num_slots: int = 16, d: int = 64):
        self.num_slots = num_slots
        self.d = d
        self.slots = np.random.randn(num_slots, d).astype(np.float32) / np.sqrt(d)
        # Unit norm normalization
        self.slots /= (np.linalg.norm(self.slots, axis=1, keepdims=True) + 1e-12)

    def read(self, query: np.ndarray) -> np.ndarray:
        """
        Soft-gated associative read in O(M * d).
        """
        scores = self.slots @ query / np.sqrt(self.d) # (M,)
        # Numerically stable softmax
        exp_s = np.exp(scores - np.max(scores))
        attn = exp_s / np.sum(exp_s)
        readout = np.sum(attn[:, None] * self.slots, axis=0) # (d,)
        return readout

    def write(self, slot_idx: int, update_vector: np.ndarray, alpha_gate: float = 0.5):
        """
        Contractive gated write in O(d).
        """
        self.slots[slot_idx] = (1.0 - alpha_gate) * self.slots[slot_idx] + alpha_gate * update_vector
        self.slots[slot_idx] /= (np.linalg.norm(self.slots[slot_idx]) + 1e-12)

class LTXHierarchicalRoutingMemory:
    """
    Hierarchical Long-Term Memory (LTX style).
    Binary decision tree of depth L = log2(N).
    Retrieval routes query along the tree nodes in O(log N * d).
    """
    def __init__(self, capacity_N: int = 1024, d: int = 64, seed: int = 42):
        np.random.seed(seed)
        self.capacity_N = capacity_N
        self.d = d
        self.depth = int(np.ceil(np.log2(capacity_N)))
        self.num_internal_nodes = 2 ** self.depth - 1
        
        # Routing hyperplane vectors at each internal node
        self.split_vectors = np.random.randn(self.num_internal_nodes, d).astype(np.float32)
        self.split_vectors /= (np.linalg.norm(self.split_vectors, axis=1, keepdims=True) + 1e-12)
        
        # Leaf storage: 2^depth leaves
        self.num_leaves = 2 ** self.depth
        self.leaf_values = np.random.randn(self.num_leaves, d).astype(np.float32)
        self.leaf_values /= (np.linalg.norm(self.leaf_values, axis=1, keepdims=True) + 1e-12)

    def route_and_read(self, query: np.ndarray) -> tuple:
        """
        Routes query from root to leaf in exactly L = depth steps.
        Time complexity: O(L * d) = O(log N * d).
        Returns: (retrieved_vector, route_path_indices, total_dot_products)
        """
        node_idx = 0
        path = []
        dot_count = 0
        for _ in range(self.depth):
            path.append(node_idx)
            split_v = self.split_vectors[node_idx]
            dot = float(np.dot(split_v, query))
            dot_count += 1
            if dot >= 0:
                node_idx = 2 * node_idx + 2 # Right child
            else:
                node_idx = 2 * node_idx + 1 # Left child
        
        leaf_idx = node_idx - self.num_internal_nodes
        retrieved = self.leaf_values[leaf_idx]
        return retrieved, path, dot_count

class ContinuousThoughtBeamSearch:
    """
    Continuous Multi-Step Reasoning Search Engine.
    Explores beam of width B across K thought steps.
    Each expansion checks b branches.
    Complexity: O(B * b * K * d), which is strictly O(K * d) for constant beam size.
    """
    def __init__(self, core, beam_width: int = 4, branch_factor: int = 2):
        self.core = core
        self.B = beam_width
        self.b = branch_factor

    def search_trajectory(self, z_init: np.ndarray, x_in: np.ndarray, m_wm: np.ndarray,
                          m_ltx: np.ndarray, c: np.ndarray, K_steps: int = 5, aux_dict: dict = None) -> list:
        if aux_dict is None:
            aux_dict = {}
        # Candidate tuples: (cumulative_score, trajectory_list)
        beam = [(0.0, [z_init])]

        for step_idx in range(K_steps):
            new_candidates = []
            for score, traj in beam:
                curr_z = traj[-1]
                # Branch out with varying step sizes (multi-scale thought refinement)
                step_sizes = np.linspace(0.2, 0.4, self.b)
                for alpha_s in step_sizes:
                    next_z = self.core.step(curr_z, x_in, m_wm, m_ltx, c, aux_dict, alpha_step=alpha_s)
                    # Continuous score: combination of energy dissipation and alignment
                    norm_change = float(np.linalg.norm(next_z - curr_z))
                    step_score = -0.5 * norm_change # prefers contractive stability
                    new_candidates.append((score + step_score, traj + [next_z]))
            
            # Select top B candidates
            new_candidates.sort(key=lambda item: item[0], reverse=True)
            beam = new_candidates[: self.B]

        best_score, best_trajectory = beam[0]
        return best_trajectory
