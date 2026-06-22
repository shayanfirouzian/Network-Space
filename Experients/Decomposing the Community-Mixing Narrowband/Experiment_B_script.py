import os
import time
import numpy as np
import pandas as pd
from numba import njit
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from tqdm import tqdm


# ==============================================================================
# 0. CONFIGURATION
#
#    Every scale parameter is overridable via an EXPB_* environment
#    variable without editing this file, e.g.:
#        EXPB_N_INSTANCES=3 EXPB_MU_GRID_STEP=0.05 EXPB_SA_STEPS=2000 \
#            python3 experiment_B_production.py
#    Unset (the normal case) => the PRODUCTION default is used.
# ==============================================================================

def _env_int(name, default):
    v = os.environ.get(name)
    return int(v) if v is not None else default


def _env_float(name, default):
    v = os.environ.get(name)
    return float(v) if v is not None else default


N = 100

# Number of cores to use. -1 = all available (joblib convention).
N_JOBS = -1
_env_n_jobs = os.environ.get('EXPB_N_JOBS')
if _env_n_jobs is not None:
    N_JOBS = int(_env_n_jobs)

# Archetype (rho, eta, c) centers -- mu is the swept variable, not part of
# the center coordinate here (Results Table 1, N=100, rounded to 1% grid).
ARCHETYPE_RHO_ETA_C = {
    'I-ER':   (0.13, 0.06, 0.14),
    'II-WST': (0.36, 0.03, 0.37),
    'III-WSD':(0.70, 0.02, 0.69),
    'IV-SF':  (0.06, 0.15, 0.07),
    'V-CP':   (0.03, 0.32, 0.04),
}

EPS_HALF = 0.025  # perturbation radius for multi-instance (rho,eta,c) sampling

N_INSTANCES_PER_ARCHETYPE = _env_int('EXPB_N_INSTANCES', 10)  # center + (N-1) perturbed

_mu_step = _env_float('EXPB_MU_GRID_STEP', 0.0125)
MU_TARGET_GRID = np.round(np.arange(0.0, 1.0 + _mu_step / 2, _mu_step), 4)
MU_TARGET_GRID = MU_TARGET_GRID[MU_TARGET_GRID <= 1.0 + 1e-9]
MU_TARGET_GRID = np.clip(MU_TARGET_GRID, 0.0, 1.0)

SA_T0 = 0.2
SA_KAPPA = 0.998
SA_STEPS = _env_int('EXPB_SA_STEPS', 8000)
SA_RESTARTS = _env_int('EXPB_SA_RESTARTS', 5)

# Partition sizes for K=2..5 (all divide N=100 evenly)
K_SIZES = {
    2: (50, 50),
    3: (34, 33, 33),
    4: (25, 25, 25, 25),
    5: (20, 20, 20, 20, 20),
}


def _expected_cross_fraction(sizes, N=N):
    total_pairs = N * (N - 1) // 2
    within_pairs = sum(s * (s - 1) // 2 for s in sizes)
    return 1.0 - within_pairs / total_pairs


MIDPOINT_K = {k: _expected_cross_fraction(sizes) for k, sizes in K_SIZES.items()}

# Five conditions: C1 = K2-original (Phase II baseline), C2..C5 = K2..K5 symmetric.
CONDITIONS = ['C1', 'C2', 'C3', 'C4', 'C5']
CONDITION_K = {'C1': 2, 'C2': 2, 'C3': 3, 'C4': 4, 'C5': 5}
CONDITION_LABELS = {
    'C1': 'K=2, original refresh',
    'C2': 'K=2, symmetric refresh',
    'C3': 'K=3, symmetric refresh',
    'C4': 'K=4, symmetric refresh',
    'C5': 'K=5, symmetric refresh',
}

BATCH_SIZE = _env_int('EXPB_BATCH_SIZE', 200)

OUTPUT_DIR = '.'


def _path(name):
    return os.path.join(OUTPUT_DIR, name)


np.random.seed(0)


def make_seed(*indices):
    """Deterministic uint32 seed from a tuple of non-negative integer
    indices, via numpy SeedSequence -- see experiment_A_production.py for
    the overflow-avoidance rationale (identical pattern used here)."""
    return int(np.random.SeedSequence(indices).generate_state(1, dtype=np.uint32)[0])


_TAG_INSTANCE_PERTURB = 50
_TAG_SA_RESTART = 51


# ==============================================================================
# 1. METRICS, PROXY FORMULA, AND VERIFICATION
#    (Carried over verbatim from the validated pilot -- generic in K, see
#    experiment_B_partition_topology.py for the +2*adj derivation history.)
# ==============================================================================

@njit(cache=True)
def _compute_metrics_k(adj, part, N):
    """
    Same as Experiment A's _compute_metrics, but mu is computed as the
    fraction of edges with part[i] != part[j], for ANY number of distinct
    labels in `part` (K=2 or K=3 here).
    """
    degrees = np.zeros(N, dtype=np.int64)
    for i in range(N):
        s = 0
        for j in range(N):
            s += adj[i, j]
        degrees[i] = s

    E = 0
    for i in range(N):
        E += degrees[i]
    E //= 2

    sum_k = 0
    sum_k2 = 0
    for i in range(N):
        sum_k += degrees[i]
        sum_k2 += degrees[i] * degrees[i]

    T = 0
    cross_edges = 0
    for i in range(N):
        for j in range(i + 1, N):
            if adj[i, j]:
                common = 0
                for w in range(N):
                    if adj[i, w] and adj[j, w]:
                        common += 1
                T += common
                if part[i] != part[j]:
                    cross_edges += 1
    T //= 3
    P = sum_k2 - 2 * E

    rho = 2.0 * E / (N * (N - 1))
    mean_k = sum_k / N
    eta = 0.0
    if mean_k > 0:
        eta = (sum_k2 / N) / (mean_k * mean_k) - 1.0
        if eta > 1.0:
            eta = 1.0
        if eta < 0.0:
            eta = 0.0
    c = 0.0
    if P > 0:
        c = 6.0 * T / P
    mu = 0.0
    if E > 0:
        mu = cross_edges / E

    return rho, eta, c, mu, E, T, P, sum_k, sum_k2, cross_edges, degrees


@njit(cache=True)
def _compute_Eto(adj, part, N, K):
    """
    Eto[i, ell] = number of edges from node i to nodes currently labeled
    `ell`. O(N^2 * K). Used to evaluate the proxy formula for ALL
    cross-community (u,v) candidate pairs in O(1) per pair after this
    O(N^2*K) precompute.
    """
    Eto = np.zeros((N, K), dtype=np.int64)
    for i in range(N):
        for j in range(N):
            if i != j and adj[i, j]:
                Eto[i, part[j]] += 1
    return Eto


# ==============================================================================
# 2. PROXY FORMULA + VERIFICATION
# ==============================================================================

@njit(cache=True)
def _proxy(Eto, adj, part, u, v):
    """
    proxy(u,v) = [E(u->a)-E(u->b)] + [E(v->b)-E(v->a)] + 2*adj[u,v]

    The +2*adj[u,v] term corrects for the (u,v) edge itself: if u,v are
    connected, E(u->b) includes this edge (since part[v]=b) and E(v->a)
    also includes it (since part[u]=a) -- but the (u,v) edge's cross/
    internal status is UNCHANGED by an a<->b label swap (it is cross
    both before, as a-vs-b, and after, as b-vs-a, since a!=b). Without
    this correction the formula double-subtracts that edge.
    """
    a = part[u]
    b = part[v]
    return (Eto[u, a] - Eto[u, b]) + (Eto[v, b] - Eto[v, a]) + 2 * adj[u, v]


def verify_proxy_formula(n_trials=200, N=N, K=3, seed=0):
    """
    Brute-force check: for random graphs/partitions and random
    cross-community (u,v) pairs, confirm that
        cross_edges(after swap) == cross_edges(before) + proxy(u,v)
    via direct O(N^2) recount before and after. Prints PASS/FAIL.
    """
    rng = np.random.RandomState(seed)
    n_checked = 0
    n_failed = 0
    for trial in range(n_trials):
        adj = (rng.random((N, N)) < 0.15).astype(np.int8)
        adj = np.triu(adj, 1)
        adj = adj + adj.T

        part = rng.randint(0, K, size=N).astype(np.int32)
        # ensure at least 2 distinct labels present
        if len(np.unique(part)) < 2:
            continue

        # pick a random cross-community pair
        for _ in range(5):
            u, v = rng.randint(0, N, size=2)
            if u != v and part[u] != part[v]:
                break
        else:
            continue

        def count_cross(adj, part, N):
            c = 0
            for i in range(N):
                for j in range(i + 1, N):
                    if adj[i, j] and part[i] != part[j]:
                        c += 1
            return c

        cross_before = count_cross(adj, part, N)
        Eto = _compute_Eto(adj, part, N, K)
        p = _proxy(Eto, adj, part, u, v)

        part2 = part.copy()
        part2[u], part2[v] = part2[v], part2[u]
        cross_after = count_cross(adj, part2, N)

        n_checked += 1
        if cross_after != cross_before + p:
            n_failed += 1

    status = "PASS" if n_failed == 0 else "FAIL"
    print(f"Proxy formula verification: {status} "
          f"({n_checked - n_failed}/{n_checked} checks correct, "
          f"K={K}, N={N})")
    return n_failed == 0




# ==============================================================================
# 2. PARTITION INITIALIZATION AND SA SYNTHESIZER
#    (Carried over verbatim from the validated pilot -- generic in K and
#    condition_code. C1 always uses K=2 with the original Phase II refresh;
#    C2-C5 use the symmetric, target-aware refresh at K=2..5 respectively.)
# ==============================================================================

@njit(cache=True)
def _init_partition(N, sizes):
    """Build a part array with the given block sizes, e.g. (50,50) or (34,33,33)."""
    part = np.zeros(N, dtype=np.int32)
    idx = 0
    for label, size in enumerate(sizes):
        for _ in range(size):
            part[idx] = label
            idx += 1
    return part


@njit(cache=True)
def _synthesize_expB(target_rho, target_eta, target_c, target_mu,
                      N, S, T0, kappa, seed,
                      part_init, K, condition_code, midpoint):
    """
    condition_code: 0 = C1 (K=2, original always-minimize refresh)
                     1 = C2/C3 (symmetric, target-aware refresh; K from
                         len(unique(part_init)))

    Returns (best_adj, best_part, best_loss). Checkpointing (best-so-far
    snapshot) follows the same rationale as Experiment A's synthesizer.
    """
    np.random.seed(seed)

    adj = np.zeros((N, N), dtype=np.int8)
    part = part_init.copy()

    for i in range(N):
        for j in range(i + 1, N):
            if np.random.random() < target_rho:
                adj[i, j] = 1
                adj[j, i] = 1

    rho, eta, c, mu, E, T_count, P, sum_k, sum_k2, cross_edges, degrees = \
        _compute_metrics_k(adj, part, N)

    def _loss(r, e, cc, m):
        dr = r - target_rho
        de = e - target_eta
        dc = cc - target_c
        dm = m - target_mu
        return np.sqrt(dr * dr + de * de + dc * dc + dm * dm)

    current_loss = _loss(rho, eta, c, mu)
    best_loss = current_loss
    best_adj = adj.copy()
    best_part = part.copy()

    temp = T0
    for step in range(S):
        u = np.random.randint(0, N)
        v = np.random.randint(0, N)
        while u == v:
            v = np.random.randint(0, N)

        is_edge = adj[u, v] == 1
        delta = -1 if is_edge else 1

        common = 0
        for w in range(N):
            if adj[u, w] and adj[v, w]:
                common += 1
        T_new = T_count + delta * common

        deg_u = degrees[u]
        deg_v = degrees[v]
        old_contrib = deg_u * (deg_u - 1) + deg_v * (deg_v - 1)
        new_deg_u = deg_u + delta
        new_deg_v = deg_v + delta
        P_new = P + (new_deg_u * (new_deg_u - 1) + new_deg_v * (new_deg_v - 1)) - old_contrib

        sum_k_new = sum_k + 2 * delta
        sum_k2_new = sum_k2 + (new_deg_u * new_deg_u - deg_u * deg_u) \
                             + (new_deg_v * new_deg_v - deg_v * deg_v)
        E_new = E + delta
        cross_new = cross_edges + (delta if part[u] != part[v] else 0)

        rho_n = 2.0 * E_new / (N * (N - 1))
        mean_k_n = sum_k_new / N
        eta_n = 0.0
        if mean_k_n > 0:
            eta_n = (sum_k2_new / N) / (mean_k_n * mean_k_n) - 1.0
            if eta_n > 1.0:
                eta_n = 1.0
            if eta_n < 0.0:
                eta_n = 0.0
        c_n = 0.0
        if P_new > 0:
            c_n = 6.0 * T_new / P_new
        mu_n = 0.0
        if E_new > 0:
            mu_n = cross_new / float(E_new)

        new_loss = _loss(rho_n, eta_n, c_n, mu_n)

        if new_loss < current_loss or \
           np.random.random() < np.exp((current_loss - new_loss) / temp):
            current_loss = new_loss
            E, T_count, P, sum_k, sum_k2, cross_edges = \
                E_new, T_new, P_new, sum_k_new, sum_k2_new, cross_new
            degrees[u] = new_deg_u
            degrees[v] = new_deg_v
            adj[u, v] = 1 - adj[u, v]
            adj[v, u] = adj[u, v]
            if current_loss < best_loss:
                best_loss = current_loss
                best_adj = adj.copy()
                best_part = part.copy()

        # --- Partition refresh every 50 steps ---
        if step % 50 == 0 and step > 0:
            if condition_code == 0:
                # C1: ORIGINAL replication. K=2 assumed. Always seeks to
                # reduce cross-community edges via the Phase II D/gain
                # formula, irrespective of target_mu.
                D = np.zeros(N, dtype=np.int64)
                for i in range(N):
                    ext_ = 0
                    int_ = 0
                    for j in range(N):
                        if i != j and adj[i, j]:
                            if part[i] != part[j]:
                                ext_ += 1
                            else:
                                int_ += 1
                    D[i] = ext_ - int_

                best_gain = -10**9
                best_u = -1
                best_v = -1
                for su in range(N):
                    if part[su] == 0:
                        for sv in range(N):
                            if part[sv] == 1:
                                gain = D[su] + D[sv] - 2 * adj[su, sv]
                                if gain > best_gain:
                                    best_gain = gain
                                    best_u = su
                                    best_v = sv
                if best_gain > 0:
                    part[best_u] = 1
                    part[best_v] = 0
                    cross_edges += best_gain
                    if E > 0 and sum_k > 0:
                        mu_recalc = cross_edges / float(E)
                        mean_k_now = sum_k / N
                        eta_recalc = (sum_k2 / N) / (mean_k_now * mean_k_now) - 1.0
                        if eta_recalc > 1.0:
                            eta_recalc = 1.0
                        if eta_recalc < 0.0:
                            eta_recalc = 0.0
                        c_recalc = 6.0 * T_count / P if P > 0 else 0.0
                        rho_recalc = 2.0 * E / (N * (N - 1))
                        recalc_loss = _loss(rho_recalc, eta_recalc, c_recalc, mu_recalc)
                        if recalc_loss < best_loss:
                            best_loss = recalc_loss
                            best_adj = adj.copy()
                            best_part = part.copy()
                        current_loss = recalc_loss

            else:
                # C2/C3: SYMMETRIC, target-aware refresh using the proxy
                # formula. Direction (minimize vs maximize cross_edges)
                # depends on target_mu vs the partition's natural midpoint.
                Eto = _compute_Eto(adj, part, N, K)
                want_minimize = target_mu < midpoint

                best_proxy = 0  # only accept strictly-improving swaps
                best_u = -1
                best_v = -1
                for su in range(N):
                    for sv in range(N):
                        if su == sv or part[su] == part[sv]:
                            continue
                        p_val = _proxy(Eto, adj, part, su, sv)
                        if want_minimize:
                            if p_val < best_proxy:
                                best_proxy = p_val
                                best_u, best_v = su, sv
                        else:
                            if p_val > best_proxy:
                                best_proxy = p_val
                                best_u, best_v = su, sv

                if best_u >= 0:
                    a, b = part[best_u], part[best_v]
                    part[best_u], part[best_v] = b, a
                    cross_edges += best_proxy
                    if E > 0 and sum_k > 0:
                        mu_recalc = cross_edges / float(E)
                        mean_k_now = sum_k / N
                        eta_recalc = (sum_k2 / N) / (mean_k_now * mean_k_now) - 1.0
                        if eta_recalc > 1.0:
                            eta_recalc = 1.0
                        if eta_recalc < 0.0:
                            eta_recalc = 0.0
                        c_recalc = 6.0 * T_count / P if P > 0 else 0.0
                        rho_recalc = 2.0 * E / (N * (N - 1))
                        recalc_loss = _loss(rho_recalc, eta_recalc, c_recalc, mu_recalc)
                        if recalc_loss < best_loss:
                            best_loss = recalc_loss
                            best_adj = adj.copy()
                            best_part = part.copy()
                        current_loss = recalc_loss

        temp *= kappa

    return best_adj, best_part, best_loss




def synthesize_expB(mu_target, condition, archetype_rho_eta_c, base_seed,
                     N=N, S=SA_STEPS, restarts=SA_RESTARTS):
    """
    condition in {'C1','C2','C3','C4','C5'}.
    archetype_rho_eta_c = (rho,eta,c) for this INSTANCE (center or
    eps-perturbed variant); mu_target is the swept variable.
    Returns (best_adj, best_part, best_loss, xhat) where xhat=(rho,eta,c,mu).
    """
    target_rho, target_eta, target_c = archetype_rho_eta_c
    K = CONDITION_K[condition]
    sizes = K_SIZES[K]
    midpoint = MIDPOINT_K[K]
    condition_code = 0 if condition == 'C1' else 1

    part_init = _init_partition(N, np.array(sizes))

    best_loss = np.inf
    best_adj = None
    best_part = None
    for r in range(restarts):
        seed_r = make_seed(_TAG_SA_RESTART, base_seed, r)
        adj, part, loss = _synthesize_expB(
            target_rho, target_eta, target_c, mu_target,
            N, S, SA_T0, SA_KAPPA, seed=seed_r,
            part_init=part_init, K=K, condition_code=condition_code,
            midpoint=midpoint
        )
        if loss < best_loss:
            best_loss = loss
            best_adj = adj.copy()
            best_part = part.copy()

    xhat = _compute_metrics_k(best_adj, best_part, N)[:4]
    return best_adj, best_part, best_loss, xhat


# ==============================================================================
# 3. CHECKPOINTING HELPERS (same pattern as Experiment A production)
# ==============================================================================

def load_completed_keys(csv_path, key_cols):
    if not os.path.exists(csv_path):
        return set()
    try:
        df = pd.read_csv(csv_path, usecols=key_cols)
    except (pd.errors.EmptyDataError, ValueError):
        return set()
    if len(df) == 0:
        return set()
    return set(map(tuple, df[key_cols].values.tolist()))


def append_rows(csv_path, rows):
    if not rows:
        return
    df = pd.DataFrame(rows)
    write_header = not os.path.exists(csv_path)
    df.to_csv(csv_path, mode='a', header=write_header, index=False)


# ==============================================================================
# 4. MULTI-INSTANCE (rho,eta,c) SAMPLING PER ARCHETYPE
#
#    Instance 0 is always the archetype's class center. Instances 1..N-1
#    are sampled by perturbing (rho,eta,c) within a 3D ball of radius
#    EPS_HALF=0.025 (mirroring Experiment A's eps/2 perturbation, but in
#    the 3D (rho,eta,c) subspace since mu is the swept variable here).
#    Perturbations are clipped to [0,1] and to rho>0.01 (avoiding the
#    degenerate near-empty-graph boundary excluded throughout this study).
# ==============================================================================

def build_archetype_instances():
    """Returns {archetype_name: [(rho,eta,c), ...]} with
    N_INSTANCES_PER_ARCHETYPE entries each (instance 0 = center)."""
    instances = {}
    for arche_idx, (arche_name, center) in enumerate(ARCHETYPE_RHO_ETA_C.items()):
        pts = [center]
        rng = np.random.RandomState(make_seed(_TAG_INSTANCE_PERTURB, arche_idx))
        for inst_id in range(1, N_INSTANCES_PER_ARCHETYPE):
            while True:
                delta = rng.normal(0, 1, size=3)
                delta = delta / np.linalg.norm(delta) * rng.uniform(0, EPS_HALF)
                pt = np.clip(np.array(center) + delta, 0.0, 1.0)
                if pt[0] > 0.01 and np.linalg.norm(delta) <= EPS_HALF:
                    break
            pts.append(tuple(pt))
        instances[arche_name] = pts
    return instances


# ==============================================================================
# 5. TASK GENERATION AND WORKER
# ==============================================================================

SWEEP_KEY_COLS = ['archetype', 'instance_id', 'mu_target', 'condition']


def sweep_task_key(task):
    return (task['archetype'], task['instance_id'], task['mu_target'], task['condition'])


def sweep_worker(task):
    archetype = task['archetype']
    instance_id = task['instance_id']
    mu_target = task['mu_target']
    condition = task['condition']
    rec = task['rho_eta_c']
    base_seed = task['base_seed']

    adj, part, loss, xhat = synthesize_expB(mu_target, condition, rec, base_seed=base_seed)

    return {
        'archetype': archetype, 'instance_id': instance_id,
        'mu_target': mu_target, 'condition': condition,
        'K': CONDITION_K[condition],
        'rho_target': rec[0], 'eta_target': rec[1], 'c_target': rec[2],
        'rho_hat': xhat[0], 'eta_hat': xhat[1],
        'c_hat': xhat[2], 'mu_hat': xhat[3],
        'loss': loss,
        'feasible': loss <= 0.05,
    }


def build_sweep_tasks(instances):
    tasks = []
    for arche_idx, (arche_name, pts) in enumerate(instances.items()):
        for instance_id, rec in enumerate(pts):
            for mu_idx, mu_target in enumerate(MU_TARGET_GRID):
                for cond_idx, condition in enumerate(CONDITIONS):
                    base_seed = make_seed(arche_idx, instance_id, mu_idx, cond_idx)
                    tasks.append({
                        'archetype': arche_name, 'instance_id': instance_id,
                        'mu_target': float(mu_target), 'condition': condition,
                        'rho_eta_c': rec, 'base_seed': base_seed,
                    })
    return tasks


# Optional task-count truncation for quick verification runs (see
# Experiment A production for the analogous EXPA_MAX_TASKS).
_env_max_tasks = os.environ.get('EXPB_MAX_TASKS')
MAX_TASKS_DEBUG = int(_env_max_tasks) if _env_max_tasks is not None else None


# ==============================================================================
# 6. MAIN SWEEP
# ==============================================================================

def run_sweep():
    print("=" * 70)
    print("EXPERIMENT B (PRODUCTION): mu-RANGE UNDER K=2..5 PARTITION SCHEMES")
    print("=" * 70)
    print(f"N_INSTANCES_PER_ARCHETYPE={N_INSTANCES_PER_ARCHETYPE}, "
          f"|MU_TARGET_GRID|={len(MU_TARGET_GRID)} (step={_mu_step}), "
          f"conditions={CONDITIONS}")
    print(f"SA_STEPS={SA_STEPS}, SA_RESTARTS={SA_RESTARTS}")
    for k, mp in MIDPOINT_K.items():
        print(f"  MIDPOINT_K{k} = {mp:.4f}  (sizes={K_SIZES[k]})")

    instances = build_archetype_instances()
    tasks = build_sweep_tasks(instances)
    if MAX_TASKS_DEBUG is not None:
        tasks = tasks[:MAX_TASKS_DEBUG]
        print(f"[DEBUG] EXPB_MAX_TASKS set: truncated to {len(tasks)} tasks")

    print(f"\n=> {len(ARCHETYPE_RHO_ETA_C)} archetypes x {N_INSTANCES_PER_ARCHETYPE} "
          f"instances x {len(MU_TARGET_GRID)} mu-targets x {len(CONDITIONS)} "
          f"conditions = {len(tasks)} tasks\n")

    sweep_csv = _path('expB_sweep_tasks.csv')
    completed = load_completed_keys(sweep_csv, SWEEP_KEY_COLS)
    todo = [t for t in tasks if sweep_task_key(t) not in completed]
    print(f"{len(completed)} already completed, {len(todo)} remaining "
          f"(of {len(tasks)} total)")

    t0 = time.time()
    for batch_start in tqdm(range(0, len(todo), BATCH_SIZE),
                             desc="B sweep", unit="batch"):
        batch = todo[batch_start:batch_start + BATCH_SIZE]
        results = Parallel(n_jobs=N_JOBS)(delayed(sweep_worker)(t) for t in batch)

        append_rows(sweep_csv, results)

    print(f"\n[Sweep: {len(todo)} new tasks in {time.time()-t0:.1f}s]")

    return pd.read_csv(sweep_csv).drop_duplicates(subset=SWEEP_KEY_COLS), instances


# ==============================================================================
# 7. AGGREGATION
# ==============================================================================

def aggregate_results(sweep_df):
    """
    Per (archetype, instance_id, condition): achievable mu-range from the
    RAW achieved mu_hat across the full mu_target grid (width_raw), and
    from the FEASIBLE (loss<=0.05) subset (width_feasible, possibly 0 if
    no targets are feasible). Then aggregate width_raw across instances
    -> mean +/- 95% CI per (archetype, condition).
    """
    width_rows = []
    for (archetype, instance_id, condition), grp in sweep_df.groupby(
            ['archetype', 'instance_id', 'condition']):
        mu_min_raw, mu_max_raw = grp.mu_hat.min(), grp.mu_hat.max()
        width_raw = mu_max_raw - mu_min_raw

        feas = grp[grp.feasible]
        if len(feas) > 0:
            mu_min_f, mu_max_f = feas.mu_hat.min(), feas.mu_hat.max()
            width_feas = mu_max_f - mu_min_f
        else:
            mu_min_f, mu_max_f, width_feas = np.nan, np.nan, 0.0

        width_rows.append({
            'archetype': archetype, 'instance_id': instance_id,
            'condition': condition, 'K': CONDITION_K[condition],
            'mu_raw_min': mu_min_raw, 'mu_raw_max': mu_max_raw,
            'width_raw': width_raw,
            'mu_feasible_min': mu_min_f, 'mu_feasible_max': mu_max_f,
            'width_feasible': width_feas,
            'n_feasible': len(feas), 'n_total': len(grp),
        })
    width_df = pd.DataFrame(width_rows)
    width_df.to_csv(_path('expB_width_summary.csv'), index=False)

    # Aggregate across instances
    by_k_rows = []
    for (archetype, condition), grp in width_df.groupby(['archetype', 'condition']):
        for col in ['width_raw', 'width_feasible']:
            vals = grp[col].values
            mean_v = vals.mean()
            std_v = vals.std(ddof=1) if len(vals) > 1 else 0.0
            se_v = std_v / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
            by_k_rows.append({
                'archetype': archetype, 'condition': condition,
                'K': CONDITION_K[condition], 'metric': col,
                'mean': mean_v, 'std': std_v, 'ci95': 1.96 * se_v,
                'n_instances': len(vals),
            })
    by_k_df = pd.DataFrame(by_k_rows)
    by_k_df.to_csv(_path('expB_width_by_K.csv'), index=False)

    print("\n" + "=" * 70)
    print("ACHIEVABLE mu-RANGE WIDTH (raw, mean +/- 95% CI across instances)")
    print("=" * 70)
    pivot = by_k_df[by_k_df.metric == 'width_raw'].pivot(
        index='archetype', columns='condition', values='mean')
    pivot = pivot.reindex(columns=CONDITIONS)
    print(pivot.round(3).to_string())

    return width_df, by_k_df


# ==============================================================================
# 8. FIGURES
# ==============================================================================

CONDITION_COLORS = {
    'C1': '#4C72B0', 'C2': '#55A868', 'C3': '#C44E52',
    'C4': '#8172B2', 'C5': '#CCB974',
}
ARCHETYPE_LIST = list(ARCHETYPE_RHO_ETA_C.keys())


def make_mu_range_figure(sweep_df):
    """Mean achieved-mu vs target-mu, shaded band = +/-1 SEM across
    instances, faceted by archetype, one line per condition."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True, sharey=True)
    axes = axes.flatten()

    for idx, arche_name in enumerate(ARCHETYPE_LIST):
        ax = axes[idx]
        for condition in CONDITIONS:
            sub = sweep_df[(sweep_df.archetype == arche_name) &
                            (sweep_df.condition == condition)]
            agg = sub.groupby('mu_target')['mu_hat'].agg(['mean', 'sem']).reset_index()
            agg = agg.sort_values('mu_target')
            ax.plot(agg.mu_target, agg['mean'], '-',
                    color=CONDITION_COLORS[condition],
                    label=f"{condition} (K={CONDITION_K[condition]})",
                    linewidth=1.5, alpha=0.9)
            sem = agg['sem'].fillna(0.0)
            ax.fill_between(agg.mu_target, agg['mean'] - sem, agg['mean'] + sem,
                             color=CONDITION_COLORS[condition], alpha=0.15)

        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1,
                label='target=achieved' if idx == 0 else None)
        ax.axvspan(0.45, 0.60, color='gray', alpha=0.15,
                   label='Phase II narrowband [0.45,0.60]' if idx == 0 else None)
        ax.set_title(arche_name)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        if idx >= 3:
            ax.set_xlabel(r'Target $\mu$')
        if idx % 3 == 0:
            ax.set_ylabel(r'Achieved $\mu$')
        ax.grid(alpha=0.3)

    axes[5].axis('off')
    handles, labels = axes[0].get_legend_handles_labels()
    axes[5].legend(handles, labels, loc='center', fontsize=9)
    axes[5].text(0.5, 0.92,
                  f'Mean $\\pm$ 1 SEM across {N_INSTANCES_PER_ARCHETYPE} instances\n'
                  f'per archetype ({len(MU_TARGET_GRID)}-point mu grid)',
                  ha='center', transform=axes[5].transAxes, fontsize=9, style='italic')

    fig.suptitle(r'Achievable $\mu$ vs Target $\mu$, by Partition '
                  r'Scheme (K=2:5)', fontsize=13)
    fig.tight_layout()
    fig.savefig(_path('expB_fig_mu_range.png'), dpi=300)
    plt.close(fig)
    print("\nFigure saved: expB_fig_mu_range.png")


def make_width_vs_K_figure(by_k_df):
    """For each archetype: width_raw (mean +/- 95% CI across instances) vs
    K, for the symmetric conditions C2-C5 (K=2..5), with C1 (K=2, original
    refresh) shown as a separate reference marker."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 9), sharey=True)

    sym_conditions = ['C2', 'C3', 'C4', 'C5']

    for idx, arche_name in enumerate(ARCHETYPE_LIST):
        ax = axes.flat[idx]
        sub = by_k_df[(by_k_df.archetype == arche_name) & (by_k_df.metric == 'width_raw')]

        sym = sub[sub.condition.isin(sym_conditions)].sort_values('K')
        ax.errorbar(
            sym.K, sym['mean'], yerr=sym['ci95'],
            marker='o', color='#55A868', linewidth=2, capsize=4,
            label='Symmetric refresh (C2-C5)'
        )

        c1 = sub[sub.condition == 'C1']
        if len(c1) > 0:
            ax.errorbar(
                [2], c1['mean'], yerr=c1['ci95'],
                marker='s', color='#4C72B0', markersize=8, capsize=4,
                label='C1 (K=2, original refresh)'
            )

        ax.set_title(arche_name)
        ax.set_xlabel('K (number of communities)' if idx == 0 else 'K')
        ax.set_xticks([2, 3, 4, 5])
        ax.grid(alpha=0.3)

    axes.flat[5].axis('off')

    title_line1 = r'Achievable $\mu$-Range Width vs Partition Granularity K'
    title_line2 = (
        f'(mean $\\pm$ 95% CI across {N_INSTANCES_PER_ARCHETYPE} '
        'instances per archetype)'
    )
    fig.suptitle(title_line1 + '\n' + title_line2, fontsize=12)

    fig.tight_layout(rect=[0.06, 0, 1, 0.96])

    row1_center = (axes[0, 0].get_position().y0 + axes[0, 0].get_position().y1) / 2
    row2_center = (axes[1, 0].get_position().y0 + axes[1, 0].get_position().y1) / 2

    fig.text(
        0.015, row1_center, r'Achievable $\mu$-range width',
        rotation=90, va='center', ha='center'
    )
    fig.text(
        0.015, row2_center, r'Achievable $\mu$-range width',
        rotation=90, va='center', ha='center'
    )

    # Legend in the blank bottom-right panel
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[5].legend(handles, labels, loc='center', fontsize=8, frameon=True)

    fig.savefig(_path('expB_fig_width_vs_K.png'), dpi=300)
    plt.close(fig)
    print("Figure saved: expB_fig_width_vs_K.png")


# ==============================================================================
# 9. MAIN
# ==============================================================================

if __name__ == '__main__':
    t_total0 = time.time()

    n_cores = os.cpu_count() or 1
    effective_jobs = n_cores if N_JOBS == -1 else N_JOBS
    print(f"Detected {n_cores} CPU cores. N_JOBS={N_JOBS} "
          f"(effective: {effective_jobs}).\n")

    print("Verifying proxy formula for K=2,3,4,5...")
    all_ok = True
    for k in [2, 3, 4, 5]:
        ok = verify_proxy_formula(n_trials=200, K=k, seed=100 + k)
        all_ok = all_ok and ok
    if not all_ok:
        raise RuntimeError("Proxy formula verification FAILED for some K -- "
                            "aborting. Results would be unreliable.")
    print()

    sweep_df, instances = run_sweep()
    width_df, by_k_df = aggregate_results(sweep_df)

    make_mu_range_figure(sweep_df)
    make_width_vs_K_figure(by_k_df)

    print(f"\n=== TOTAL RUNTIME: {(time.time()-t_total0)/60:.2f} minutes ===")
    print("\nAll outputs written to current directory:")
    for f in ['expB_sweep_tasks.csv', 'expB_width_summary.csv',
              'expB_width_by_K.csv', 'expB_fig_mu_range.png',
              'expB_fig_width_vs_K.png']:
        print(f"  {f}")
    print("\nRe-running this script will SKIP all completed tasks (checkpointed "
          "via expB_sweep_tasks.csv) and only compute what's missing.")
