import numpy as np
import math
import time
import csv
import numba
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32

# =============================================================================
# 1. DEVICE FUNCTIONS
# =============================================================================

@cuda.jit(device=True)
def popcount64_gpu(x):
    x = x - ((x >> numba.uint64(1)) & numba.uint64(0x5555555555555555))
    x = (x & numba.uint64(0x3333333333333333)) + ((x >> numba.uint64(2)) & numba.uint64(0x3333333333333333))
    x = (x + (x >> numba.uint64(4))) & numba.uint64(0x0F0F0F0F0F0F0F0F)
    return numba.int32((x * numba.uint64(0x0101010101010101)) >> numba.uint64(56))

# =============================================================================
# 2. THE PARAMETRIC CUDA KERNEL (Supports N up to 1024)
# =============================================================================

@cuda.jit
def gpu_sa_kernel(targets, out_losses, rng_states, steps, N):
    tid = cuda.grid(1)
    if tid >= targets.shape[0]:
        return

    # Calculate required 64-bit words for adjacency (N=1024 -> 16 words)
    words = (N + 63) // 64
    mid = N // 2

    # Local arrays with fixed buffer size for up to 1024 nodes
    # adj uses (nodes, words) structure
    adj = cuda.local.array((1024, 16), dtype=numba.uint64)
    part = cuda.local.array(1024, dtype=numba.int32)
    degrees = cuda.local.array(1024, dtype=numba.int32)
    D_array = cuda.local.array(1024, dtype=numba.int32)

    target_rho = targets[tid, 0]
    target_eta = targets[tid, 1]
    target_c   = targets[tid, 2]
    target_mu  = targets[tid, 3]

    # Initialization
    for i in range(N):
        for w in range(words): adj[i, w] = 0
        degrees[i] = 0
        part[i] = 1 if i >= mid else 0

    # Initial Random Graph
    for i in range(N):
        for j in range(i + 1, N):
            if xoroshiro128p_uniform_float32(rng_states, tid) < target_rho:
                adj[i, j // 64] |= (numba.uint64(1) << numba.uint64(j % 64))
                adj[j, i // 64] |= (numba.uint64(1) << numba.uint64(i % 64))

    # Initial Metrics
    E, sum_k, sum_k2 = 0, 0, 0
    for i in range(N):
        k = 0
        for w in range(words): k += popcount64_gpu(adj[i, w])
        degrees[i] = k
        E += k
        sum_k += k
        sum_k2 += k * k
    E //= 2

    T, cross_edges = 0, 0
    for i in range(N):
        for j in range(i + 1, N):
            if (adj[i, j // 64] & (numba.uint64(1) << numba.uint64(j % 64))) > 0:
                c_val = 0
                for w in range(words): c_val += popcount64_gpu(adj[i, w] & adj[j, w])
                T += c_val
                if part[i] != part[j]: cross_edges += 1
    T //= 3
    P = sum_k2 - 2 * E

    # Baseline Loss
    rho = 2.0 * E / (N * (N - 1.0))
    mean_k = sum_k / float(N)
    eta = max(0.0, min(1.0, ((sum_k2 / float(N)) / (mean_k**2)) - 1.0)) if mean_k > 0 else 0.0
    c = 6.0 * T / P if P > 0 else 0.0
    mu = cross_edges / float(E) if E > 0 else 0.0

    current_loss = math.sqrt((rho-target_rho)**2 + (eta-target_eta)**2 + (c-target_c)**2 + (mu-target_mu)**2)
    best_loss = current_loss

    temp, cooling = 0.2, 0.998
    for step in range(steps):
        u = numba.int32(xoroshiro128p_uniform_float32(rng_states, tid) * N)
        v = numba.int32(xoroshiro128p_uniform_float32(rng_states, tid) * N)
        while u == v: v = numba.int32(xoroshiro128p_uniform_float32(rng_states, tid) * N)

        mask_v, mask_u = numba.uint64(1) << numba.uint64(v % 64), numba.uint64(1) << numba.uint64(u % 64)
        is_edge = (adj[u, v // 64] & mask_v) > 0
        delta = -1 if is_edge else 1

        common = 0
        for w in range(words): common += popcount64_gpu(adj[u, w] & adj[v, w])

        T_new = T + delta * common
        new_deg_u, new_deg_v = degrees[u] + delta, degrees[v] + delta
        P_new = P + (new_deg_u*(new_deg_u-1) + new_deg_v*(new_deg_v-1)) - (degrees[u]*(degrees[u]-1) + degrees[v]*(degrees[v]-1))
        sum_k_new, sum_k2_new = sum_k + 2*delta, sum_k2 + (new_deg_u**2 - degrees[u]**2) + (new_deg_v**2 - degrees[v]**2)
        E_new = E + delta
        cross_edges_new = cross_edges + (delta if part[u] != part[v] else 0)

        rho_n = 2.0 * E_new / (N * (N - 1.0))
        mean_k_n = sum_k_new / float(N)
        eta_n = max(0.0, min(1.0, ((sum_k2_new / float(N)) / (mean_k_n**2)) - 1.0)) if mean_k_n > 0 else 0.0
        c_n = 6.0 * T_new / P_new if P_new > 0 else 0.0
        mu_n = cross_edges_new / float(E_new) if E_new > 0 else 0.0

        new_loss = math.sqrt((rho_n-target_rho)**2 + (eta_n-target_eta)**2 + (c_n-target_c)**2 + (mu_n-target_mu)**2)

        if new_loss < current_loss or xoroshiro128p_uniform_float32(rng_states, tid) < math.exp((current_loss - new_loss) / temp):
            current_loss, E, T, P, sum_k, sum_k2 = new_loss, E_new, T_new, P_new, sum_k_new, sum_k2_new
            degrees[u], degrees[v], cross_edges = new_deg_u, new_deg_v, cross_edges_new
            adj[u, v // 64] ^= mask_v
            adj[v, u // 64] ^= mask_u
            if current_loss < best_loss: best_loss = current_loss

        # Stochastic partition update logic remains based on dynamic N
        if step % 50 == 0:
            for i in range(N):
                ext, int_ = 0, 0
                for j in range(N):
                    if i == j: continue
                    if (adj[i, j // 64] & (numba.uint64(1) << numba.uint64(j % 64))) > 0:
                        if part[i] != part[j]: ext += 1
                        else: int_ += 1
                D_array[i] = ext - int_

            bg, bu, bv = -9999, -1, -1
            for su in range(N):
                if part[su] == 0:
                    for sv in range(N):
                        if part[sv] == 1:
                            is_e = (adj[su, sv // 64] & (numba.uint64(1) << numba.uint64(sv % 64))) > 0
                            gn = D_array[su] + D_array[sv] - 2 * (1 if is_e else 0)
                            if gn > bg: bg, bu, bv = gn, su, sv
            if bg > 0:
                part[bu], part[bv] = 1, 0
                cross_edges += bg
        temp *= cooling
    out_losses[tid] = best_loss

from google.colab import drive
import logging
drive.mount('/content/drive')

print(f"🚀 Initializing GPU Grid Sweep (104,060,401 Combinations)...")
start_step = 0
print(f"⏩ RESUMING FROM STEP {start_step}/101. Skipping earlier batches.")

# 101 steps to cover 0.00 to 1.00 inclusive
grid_1d = np.linspace(0, 1, 101, dtype=np.float32)

# Pre-build the inner 3D chunk (Eta, C, Mu).
E, C, M = np.meshgrid(grid_1d, grid_1d, grid_1d, indexing='ij')
ecm_chunk = np.column_stack((E.ravel(), C.ravel(), M.ravel()))

batch_size = 1030301  # 1,030,301
threads_per_block = 256
blocks = (batch_size + threads_per_block - 1) // threads_per_block

print(f"🎲 Seeding {batch_size} GPU random states...")
rng_states = create_xoroshiro128p_states(threads_per_block * blocks, seed=42)

csv_file = f"/content/drive/MyDrive/global_manifold_filtered-{start_step}.csv"

# FILE HANDLING: Only write the header if we are starting fresh from Step 0.
# If resuming, we just leave the file alone and prepare to append.
with open(csv_file, 'w', newline='') as f:
    csv.writer(f).writerow(['rho', 'eta', 'c', 'mu', 'loss'])

start_total = time.time()
total_feasible_found = 0

# SLICE THE GRID: Only process from 'start_step' to the end.
remaining_rho_values = grid_1d[start_step:]

from tqdm.notebook import tqdm

# SET YOUR CUSTOM N HERE
N_NODES = 500  # Supports up to 1024
SA_STEPS = 2000

for offset, rho in tqdm(enumerate(remaining_rho_values), total=len(remaining_rho_values), desc=f"Processing Rho for N={N_NODES}"):
    actual_step = start_step + offset
    batch_targets = np.empty((batch_size, 4), dtype=np.float32)
    batch_targets[:, 0] = rho
    batch_targets[:, 1:] = ecm_chunk

    d_targets = cuda.to_device(batch_targets)
    d_losses = cuda.device_array(batch_size, dtype=np.float32)

    # The kernel now uses N_NODES to govern internal complexity
    gpu_sa_kernel[blocks, threads_per_block](d_targets, d_losses, rng_states, SA_STEPS, N_NODES)
    cuda.synchronize()

    losses = d_losses.copy_to_host()
    feasible_mask = losses <= 0.05
    feasible_data = batch_targets[feasible_mask]
    feasible_loss = losses[feasible_mask].reshape(-1, 1)
    total_feasible_found += len(feasible_data)

    if len(feasible_data) > 0:
        save_array = np.hstack((feasible_data, feasible_loss))
        with open(csv_file, 'a', newline='') as f:
            csv.writer(f).writerows(save_array)

print(f"\n✅ MAPPING COMPLETE for N={N_NODES}!")
