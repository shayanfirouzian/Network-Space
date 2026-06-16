import os
import time
import itertools
import numpy as np
import pandas as pd
from numba import njit
from scipy import stats
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from tqdm import tqdm


# ==============================================================================
# 0. CONFIGURATION
# ==============================================================================

N = 100
RNG_SEED_BASE = 1000  # replication r uses seed RNG_SEED_BASE + r (r=0..N_REPLICATIONS-1)

# Number of cores to use. -1 = all available (joblib convention).
N_JOBS = -1

# Archetype class centers, rounded to 1% grid (Results Table 1, N=100)
ARCHETYPE_CENTERS = {
    'I-ER':   (0.13, 0.06, 0.14, 0.53),
    'II-WST': (0.36, 0.03, 0.37, 0.55),
    'III-WSD':(0.70, 0.02, 0.69, 0.53),
    'IV-SF':  (0.06, 0.15, 0.07, 0.54),
    'V-CP':   (0.03, 0.32, 0.04, 0.52),
}

# tau_c and RI per archetype (Methods: tau_c = 1/((N-1)*rho_bar*(1+eta_bar));
# RI = c_bar), computed from per-class parameter means.
TAU_C = {'I-ER': 0.0734, 'II-WST': 0.0275, 'III-WSD': 0.0142,
         'IV-SF': 0.1592, 'V-CP': 0.2833}
RI    = {'I-ER': 0.138,  'II-WST': 0.367,  'III-WSD': 0.692,
         'IV-SF': 0.070, 'V-CP': 0.041}

EPS = 0.05
# Both perturbation radii are tested in production (pilot tested only eps/2).
EPS_CONDITIONS = {
    'eps_half': EPS / 2.0,  # 0.025 -- stricter, with margin from class boundary
    'eps_full': EPS,        # 0.05  -- AT the nominal equivalence-class boundary
}

# --- Env-var override helper -------------------------------------------
# Every "scale" parameter below can be overridden via an EXPA_* environment
# variable without editing this file, e.g.:
#     EXPA_N_REPLICATIONS=3 EXPA_N_WITHIN_CLASS_GRAPHS=8 EXPA_SA_STEPS=2000 \
#         python3 experiment_A_production.py
# Unset (the normal case) => the PRODUCTION default shown is used. This is
# the primary mechanism for quick reduced-scale verification runs.
def _env_int(name, default):
    v = os.environ.get(name)
    return int(v) if v is not None else default


# --- Part A1 (within-class) production scale ---
N_REPLICATIONS = _env_int('EXPA_N_REPLICATIONS', 10)       # independent seeds -> ICC distribution
N_WITHIN_CLASS_GRAPHS = _env_int('EXPA_N_WITHIN_CLASS_GRAPHS', 30)  # graphs per (archetype,eps,rep)
N_SIR_RUNS = _env_int('EXPA_N_SIR_RUNS', 200)
N_CASCADE_RUNS = _env_int('EXPA_N_CASCADE_RUNS', 200)

# --- Part A2 (between-class) production scale ---
N_A2_INSTANCES = _env_int('EXPA_N_A2_INSTANCES', 5)         # center-graphs per archetype
_n_sweep = _env_int('EXPA_SWEEP_POINTS', 20)
BETA_SWEEP = np.geomspace(0.005, 0.6, _n_sweep)
THETA_SWEEP = np.linspace(0.02, 0.60, _n_sweep)

# --- SA synthesis parameters ---
SA_T0 = 0.2
SA_KAPPA = 0.998
SA_STEPS = _env_int('EXPA_SA_STEPS', 8000)
SA_RESTARTS = _env_int('EXPA_SA_RESTARTS', 5)

# --- Cascade transition-theta scan (computed once per archetype) ---
THETA_SCAN_GRID = np.arange(0.02, 0.62, 0.02)
CASCADE_SEED_FRACTION = 0.05
N_THETA_SCAN_SEEDS = _env_int('EXPA_N_THETA_SCAN_SEEDS', 50)

# --- Checkpointing batch size: results are appended to CSV every
#     BATCH_SIZE completed tasks. Smaller = more frequent checkpoints
#     (safer against interruption) but slightly more I/O overhead. ---
BATCH_SIZE = 60

OUTPUT_DIR = '.'


def _path(name):
    return os.path.join(OUTPUT_DIR, name)


# Optional overrides for quick verification runs WITHOUT editing the file.
# Leave both unset for a full production run.
#   EXPA_MAX_TASKS=12 python3 experiment_A_production.py
#     -> truncates the A1 and A2 task lists to (at most) this many tasks,
#        using PRODUCTION SA_STEPS/SA_RESTARTS etc. -- useful to confirm
#        the pipeline runs end-to-end (incl. figures) before committing to
#        a full run.
#   EXPA_N_JOBS=1 python3 experiment_A_production.py
#     -> overrides N_JOBS (e.g. force sequential execution for debugging).
# Both are read by every process (including joblib workers, which
# re-import this module), since environment variables are inherited by
# subprocesses -- unlike in-process monkeypatching of module globals.
_env_max_tasks = os.environ.get('EXPA_MAX_TASKS')
MAX_TASKS_DEBUG = int(_env_max_tasks) if _env_max_tasks is not None else None

_env_n_jobs = os.environ.get('EXPA_N_JOBS')
if _env_n_jobs is not None:
    N_JOBS = int(_env_n_jobs)

np.random.seed(0)  # top-level seed for any non-task-specific randomness


def make_seed(*indices):
    """
    Deterministic uint32 seed derived from a tuple of non-negative integer
    indices, via numpy's SeedSequence. Used everywhere multiple hierarchical
    indices (archetype, eps_condition, replication, graph_id, run_id, ...)
    must be combined into a single seed for np.random.seed()/RandomState().

    SeedSequence hashes its entropy internally, so indices of ANY magnitude
    are accepted with no overflow risk -- unlike naive multiplicative
    combination (e.g. `idx1*1000+idx2`), which silently overflows uint32
    (raising ValueError inside numba/numpy) once idx1 exceeds ~4.3 million,
    or once a CHAIN of such combinations (e.g. base_seed*1000+r where
    base_seed is itself the output of an earlier multiplicative combination)
    pushes the product past 2**32-1.
    """
    return int(np.random.SeedSequence(indices).generate_state(1, dtype=np.uint32)[0])


# Namespace tags to keep seed streams for different purposes independent
# (purely for clarity/avoiding accidental collisions -- SeedSequence
# already makes collisions astronomically unlikely even without these).
_TAG_A1_BASE = 10
_TAG_A1_SIR = 11
_TAG_A1_CASCADE = 12
_TAG_A2_BASE = 20
_TAG_A2_SIR = 21
_TAG_A2_CASCADE = 22
_TAG_THETA_SCAN_REF = 30
_TAG_SA_RESTART = 40

# ==============================================================================
# 1. CORE SA SYNTHESIZER AND CONTAGION DYNAMICS
#    (Carried over verbatim from the validated pilot -- see
#    experiment_A_equivalence_validation.py for derivation/verification
#    notes, including the checkpoint-on-improvement fix for the SA
#    synthesizer and the +2*adj correction history.)
# ==============================================================================

@njit(cache=True)
def _compute_metrics(adj, part, N):
    """Compute (rho, eta, c, mu, E, T, P, sum_k, sum_k2, cross_edges, degrees)."""
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
def _synthesize_single(target_rho, target_eta, target_c, target_mu,
                        N, S, T0, kappa, seed):
    """
    One Simulated Annealing run targeting (target_rho, target_eta,
    target_c, target_mu). Returns (best_adj, best_part, best_loss), where
    best_adj/best_part are checkpointed snapshots of the adjacency matrix
    and community partition AT the moment best_loss was achieved (the
    final annealing state at finite temperature need not coincide with the
    best state found, so it is not used directly).
    Mirrors the Phase II GPU kernel's incremental metric updates,
    Metropolis acceptance, geometric cooling, and single-pass KL refresh
    every 50 steps (always minimizing cross-edges, as in the main sweep).
    """
    np.random.seed(seed)

    adj = np.zeros((N, N), dtype=np.int8)
    part = np.zeros(N, dtype=np.int32)
    for i in range(N // 2, N):
        part[i] = 1

    # Random initialization at target density
    for i in range(N):
        for j in range(i + 1, N):
            if np.random.random() < target_rho:
                adj[i, j] = 1
                adj[j, i] = 1

    rho, eta, c, mu, E, T_count, P, sum_k, sum_k2, cross_edges, degrees = \
        _compute_metrics(adj, part, N)

    def _loss(r, e, cc, m):
        dr = r - target_rho
        de = e - target_eta
        dc = cc - target_c
        dm = m - target_mu
        return np.sqrt(dr * dr + de * de + dc * dc + dm * dm)

    current_loss = _loss(rho, eta, c, mu)
    best_loss = current_loss
    # CRITICAL: checkpoint the adjacency/partition AT the best loss seen so
    # far. At finite temperature the trajectory continues to wander after
    # finding a good state (Metropolis still accepts mildly worse moves), so
    # the FINAL state at step S can differ substantially from the BEST state
    # found along the way. Returning the final state would silently return a
    # graph whose metrics do not match `best_loss`. We therefore snapshot
    # (adj, part) every time best_loss improves and return that snapshot.
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

        # Single-pass KL bisection refresh every 50 steps
        if step % 50 == 0 and step > 0:
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

        temp *= kappa

    return best_adj, best_part, best_loss


def synthesize_graph(target, N=N, S=SA_STEPS, restarts=SA_RESTARTS, base_seed=0):
    """Multi-restart wrapper; returns (best_adj, best_part, best_loss, achieved_xhat)."""
    target_rho, target_eta, target_c, target_mu = target
    best_loss = np.inf
    best_adj = None
    best_part = None
    for r in range(restarts):
        adj, part, loss = _synthesize_single(
            target_rho, target_eta, target_c, target_mu,
            N, S, SA_T0, SA_KAPPA, seed=make_seed(_TAG_SA_RESTART, base_seed, r)
        )
        if loss < best_loss:
            best_loss = loss
            best_adj = adj.copy()
            best_part = part.copy()
    xhat = _compute_metrics(best_adj, best_part, N)[:4]
    return best_adj, best_part, best_loss, xhat


# ==============================================================================
# 2. CONTAGION DYNAMICS
# ==============================================================================

@njit(cache=True)
def run_sir(adj, N, beta, seed, max_steps=300):
    """
    Discrete-time SIR, gamma=1 (each infected node is infectious for
    exactly one step, then recovers). Maps to the bond-percolation /
    Reed-Frost final-size model, with HMF epidemic threshold beta_c = tau_c.
    Returns final outbreak size as a fraction of N.
    """
    np.random.seed(seed)
    state = np.zeros(N, dtype=np.int8)  # 0=S, 1=I, 2=R
    patient_zero = np.random.randint(0, N)
    state[patient_zero] = 1

    for _ in range(max_steps):
        new_state = state.copy()
        any_infected = False
        for i in range(N):
            if state[i] == 1:
                any_infected = True
                for j in range(N):
                    if adj[i, j] and state[j] == 0:
                        if np.random.random() < beta:
                            new_state[j] = 1
                new_state[i] = 2
        state = new_state
        if not any_infected:
            break

    return np.sum(state == 2) / N


@njit(cache=True)
def run_threshold_cascade(adj, N, degrees, theta, seed_nodes, max_steps=300):
    """
    Centola/Granovetter complex-contagion threshold model. A node adopts
    once the FRACTION of its neighbors that have adopted reaches theta.
    seed_nodes: int array of initially-adopted node indices.
    Returns final adoption fraction.
    """
    state = np.zeros(N, dtype=np.int8)  # 0=not adopted, 1=adopted
    for s in seed_nodes:
        state[s] = 1

    for _ in range(max_steps):
        new_state = state.copy()
        changed = False
        for i in range(N):
            if state[i] == 0 and degrees[i] > 0:
                n_adopted = 0
                for j in range(N):
                    if adj[i, j] and state[j] == 1:
                        n_adopted += 1
                if (n_adopted / degrees[i]) >= theta:
                    new_state[i] = 1
                    changed = True
        state = new_state
        if not changed:
            break

    return np.sum(state == 1) / N


def degrees_of(adj, N=N):
    return adj.sum(axis=1)


def find_transition_theta(adj, degrees, theta_grid, n_seeds=20, seed_frac=0.05,
                           rng_seed=0):
    """
    Scan theta_grid on a given graph and return the theta value that
    maximizes the VARIANCE of the cascade outcome across n_seeds random
    seed-sets. This identifies each archetype's OWN bootstrap-percolation
    transition point, analogous to choosing beta = 1.5 * tau_c for SIR.

    RATIONALE: a single fixed theta produces wildly different absolute
    adoption thresholds (theta * mean_degree) across archetypes whose mean
    degrees range from ~3 (Core-Periphery) to ~69 (WS-Dense). At a fixed
    theta=0.20, high-degree archetypes are pinned at the seed fraction
    (cascade impossible) while low-degree archetypes saturate to full
    adoption (cascade trivial) -- both are ZERO-VARIANCE degeneracies that
    make a variance-equality test (Levene) uninformative (0/0 -> NaN).
    Selecting theta per-archetype at its own transition restores a
    meaningful, non-degenerate comparison.
    """
    n_seed_nodes = max(1, int(seed_frac * len(degrees)))
    N_local = len(degrees)
    best_theta, best_var = theta_grid[0], -1.0
    for theta in theta_grid:
        outcomes = np.empty(n_seeds)
        for s in range(n_seeds):
            rng = np.random.RandomState(make_seed(rng_seed, int(round(theta * 1000)), s))
            seed_nodes = rng.choice(N_local, size=n_seed_nodes, replace=False).astype(np.int64)
            outcomes[s] = run_threshold_cascade(adj, N_local, degrees, float(theta), seed_nodes)
        var = outcomes.var(ddof=1)
        if var > best_var:
            best_var = var
            best_theta = float(theta)
    return best_theta, best_var



# ==============================================================================
# 2. STATISTICS: ANOVA + ICC FROM PER-GRAPH SUMMARY STATISTICS
#
#    Production stores, per graph, only (mean, var, n) of its SIR/cascade
#    outcomes -- not the raw n-length arrays. This keeps checkpoint rows
#    tiny (one row per graph regardless of n) and generalizes to
#    unbalanced designs (different n per graph, e.g. if a run is retried).
#    The one-way ANOVA sums-of-squares decompose exactly from group-level
#    (mean, var, n) via the standard identities below -- no information is
#    lost relative to the raw-array formulation used in the pilot.
# ==============================================================================

def anova_from_group_stats(group_stats):
    """
    group_stats: list of (mean_i, var_i, n_i) tuples, one per group
                  (here: one per perturbed graph).

    Returns (F, p, ms_between, ms_within, k, df_within), where:
        SS_within  = sum_i (n_i - 1) * var_i
        SS_between = sum_i n_i * (mean_i - grand_mean)^2
        grand_mean = sum_i n_i*mean_i / sum_i n_i

    Degenerate cases (ms_within == 0) are handled exactly as in the
    validated pilot's safe_f_oneway: both-zero -> (nan, 1.0); between>0,
    within==0 -> (inf, 0.0).
    """
    means = np.array([g[0] for g in group_stats], dtype=np.float64)
    vars_ = np.array([g[1] for g in group_stats], dtype=np.float64)
    ns = np.array([g[2] for g in group_stats], dtype=np.float64)
    k = len(group_stats)

    N_total = ns.sum()
    grand_mean = np.sum(ns * means) / N_total

    ss_between = np.sum(ns * (means - grand_mean) ** 2)
    ss_within = np.sum((ns - 1) * vars_)

    df_between = k - 1
    df_within = N_total - k

    ms_between = ss_between / df_between if df_between > 0 else 0.0
    ms_within = ss_within / df_within if df_within > 0 else 0.0

    if ms_within < 1e-12:
        if ms_between < 1e-12:
            return np.nan, 1.0, ms_between, ms_within, k, df_within
        else:
            return np.inf, 0.0, ms_between, ms_within, k, df_within

    F = ms_between / ms_within
    p = 1.0 - stats.f.cdf(F, df_between, df_within)
    return F, p, ms_between, ms_within, k, df_within


def icc1_from_ms(ms_between, ms_within, n_avg):
    """
    ICC(1) for a (possibly mildly unbalanced) one-way random-effects
    design, using the average group size n_avg in place of the balanced-
    design 'n'. For exactly-balanced designs (all n_i equal, as in this
    script's production tasks) this reduces to the standard formula.

        ICC(1) = (MS_between - MS_within) / (MS_between + (n_avg-1)*MS_within)

    Interpretation as in the pilot: ~0 => postulate supported (structural
    perturbation within the eps-ball contributes negligible outcome
    variance beyond stochastic noise); ~1 => violated.
    """
    denom = ms_between + (n_avg - 1) * ms_within
    if denom < 1e-12:
        return np.nan
    return (ms_between - ms_within) / denom


# ==============================================================================
# 3. CHECKPOINTING HELPERS
#
#    Pattern: each "task" is uniquely identified by a tuple of KEY columns.
#    Before running, load any existing output CSV and build a set of
#    completed keys; skip tasks whose key is already present. Results are
#    appended in batches (BATCH_SIZE) so an interruption loses at most one
#    batch of work.
# ==============================================================================

def load_completed_keys(csv_path, key_cols):
    """Return a set of tuples (one per key_cols combination) already
    present in csv_path, or an empty set if the file doesn't exist yet."""
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
    """Append a list of dicts to csv_path, writing a header iff the file
    doesn't yet exist. No-op if rows is empty."""
    if not rows:
        return
    df = pd.DataFrame(rows)
    write_header = not os.path.exists(csv_path)
    df.to_csv(csv_path, mode='a', header=write_header, index=False)


def run_tasks_with_checkpointing(tasks, key_fn, worker_fn, csv_path,
                                  key_cols, batch_size=BATCH_SIZE,
                                  n_jobs=N_JOBS, desc=""):
    """
    Generic checkpointed-parallel task runner.

      tasks:    list of task-description dicts/tuples
      key_fn:   tasks[i] -> tuple matching key_cols (for skip-detection)
      worker_fn: tasks[i] -> dict of result columns (including the key
                 columns, so the output row is self-describing)
      csv_path: output CSV (also serves as the checkpoint file)
      key_cols: list of column names forming the unique key

    Returns the number of NEWLY completed tasks (tasks already present in
    csv_path are skipped and not counted).
    """
    completed = load_completed_keys(csv_path, key_cols)
    todo = [t for t in tasks if key_fn(t) not in completed]

    print(f"[{desc}] {len(completed)} already completed, "
          f"{len(todo)} remaining (of {len(tasks)} total)")

    if not todo:
        return 0

    n_new = 0
    for batch_start in tqdm(range(0, len(todo), batch_size),
                             desc=desc, unit="batch"):
        batch = todo[batch_start:batch_start + batch_size]

        results = Parallel(n_jobs=n_jobs)(
            delayed(worker_fn)(t) for t in batch
        )

        append_rows(csv_path, results)
        n_new += len(results)

    return n_new


# ==============================================================================
# 4. PART A1: WITHIN-CLASS EQUIVALENCE (PRODUCTION)
#
#    For each archetype, for each eps-condition (eps/2 and full eps), for
#    each of N_REPLICATIONS independent seeds, N_WITHIN_CLASS_GRAPHS graphs
#    are synthesized at targets perturbed within the eps-ball of the
#    archetype center. Each graph's SIR (beta=1.5*tau_c) and cascade
#    (theta = archetype's own transition point) outcomes are summarized as
#    (mean, var, n) over N_SIR_RUNS / N_CASCADE_RUNS repeats.
#
#    ANOVA + ICC are computed per (archetype, eps_cond, replication) from
#    the N_WITHIN_CLASS_GRAPHS group-summaries, then aggregated across the
#    N_REPLICATIONS replications to give a DISTRIBUTION of ICC estimates.
# ==============================================================================

def precompute_theta_tests(verbose=True):
    """
    Compute theta_test once per archetype via find_transition_theta on a
    dedicated reference synthesis at the archetype center (fixed seed,
    independent of replication/graph_id/eps_condition). theta_test
    characterizes the ARCHETYPE's bootstrap-percolation transition point
    and is shared across all A1 tasks for that archetype.
    """
    theta_tests = {}
    for arche_idx, (arche_name, center) in enumerate(ARCHETYPE_CENTERS.items()):
        ref_seed = make_seed(_TAG_THETA_SCAN_REF, arche_idx)
        adj, part, loss, xhat = synthesize_graph(center, base_seed=ref_seed)
        degrees = degrees_of(adj)
        theta_test, theta_var = find_transition_theta(
            adj, degrees, THETA_SCAN_GRID,
            n_seeds=N_THETA_SCAN_SEEDS, seed_frac=CASCADE_SEED_FRACTION,
            rng_seed=arche_idx
        )
        theta_tests[arche_name] = theta_test
        if verbose:
            print(f"  {arche_name}: theta_test={theta_test:.2f} "
                  f"(scan variance={theta_var:.4f}, ref loss={loss:.4f})")
    return theta_tests


def a1_task_key(task):
    return (task['archetype'], task['eps_cond'], task['replication'], task['graph_id'])


A1_KEY_COLS = ['archetype', 'eps_cond', 'replication', 'graph_id']


def a1_worker(task):
    archetype = task['archetype']
    eps_cond = task['eps_cond']
    eps_radius = task['eps_radius']
    replication = task['replication']
    graph_id = task['graph_id']
    center = task['center']
    beta_test = task['beta_test']
    theta_test = task['theta_test']
    base_seed = task['base_seed']

    rng = np.random.RandomState(base_seed)
    while True:
        delta = rng.normal(0, 1, size=4)
        delta = delta / np.linalg.norm(delta) * rng.uniform(0, eps_radius)
        target = np.clip(np.array(center) + delta, 0.0, 1.0)
        if np.linalg.norm(delta) <= eps_radius:
            break

    adj, part, loss, xhat = synthesize_graph(tuple(target), base_seed=base_seed,
                                              S=SA_STEPS, restarts=SA_RESTARTS)
    degrees = degrees_of(adj)

    sir_outcomes = np.empty(N_SIR_RUNS)
    for r in range(N_SIR_RUNS):
        sir_outcomes[r] = run_sir(adj, N, beta_test,
                                   seed=make_seed(_TAG_A1_SIR, base_seed, r))

    n_seeds = max(1, int(CASCADE_SEED_FRACTION * N))
    cascade_outcomes = np.empty(N_CASCADE_RUNS)
    for r in range(N_CASCADE_RUNS):
        seed_rng = np.random.RandomState(make_seed(_TAG_A1_CASCADE, base_seed, r))
        seed_nodes = seed_rng.choice(N, size=n_seeds, replace=False).astype(np.int64)
        cascade_outcomes[r] = run_threshold_cascade(adj, N, degrees, theta_test, seed_nodes)

    return {
        'archetype': archetype, 'eps_cond': eps_cond, 'replication': replication,
        'graph_id': graph_id,
        'eps_radius': eps_radius,
        'target_rho': target[0], 'target_eta': target[1],
        'target_c': target[2], 'target_mu': target[3],
        'xhat_rho': xhat[0], 'xhat_eta': xhat[1],
        'xhat_c': xhat[2], 'xhat_mu': xhat[3],
        'loss': loss,
        'beta_test': beta_test, 'theta_test': theta_test,
        'sir_mean': sir_outcomes.mean(), 'sir_var': sir_outcomes.var(ddof=1),
        'sir_n': N_SIR_RUNS,
        'cascade_mean': cascade_outcomes.mean(),
        'cascade_var': cascade_outcomes.var(ddof=1),
        'cascade_n': N_CASCADE_RUNS,
    }


def build_a1_tasks(theta_tests):
    tasks = []
    for arche_idx, (arche_name, center) in enumerate(ARCHETYPE_CENTERS.items()):
        beta_test = 1.5 * TAU_C[arche_name]
        theta_test = theta_tests[arche_name]
        for eps_idx, (eps_name, eps_radius) in enumerate(EPS_CONDITIONS.items()):
            for replication in range(N_REPLICATIONS):
                for graph_id in range(N_WITHIN_CLASS_GRAPHS):
                    base_seed = make_seed(_TAG_A1_BASE, RNG_SEED_BASE,
                                           arche_idx, eps_idx, replication, graph_id)
                    tasks.append({
                        'archetype': arche_name, 'eps_cond': eps_name,
                        'eps_radius': eps_radius,
                        'replication': replication, 'graph_id': graph_id,
                        'center': center, 'beta_test': beta_test,
                        'theta_test': theta_test, 'base_seed': base_seed,
                    })
    return tasks


def run_part_A1():
    print("=" * 70)
    print("PART A1: WITHIN-CLASS EQUIVALENCE (production)")
    print("=" * 70)
    print(f"N_REPLICATIONS={N_REPLICATIONS}, "
          f"N_WITHIN_CLASS_GRAPHS={N_WITHIN_CLASS_GRAPHS}, "
          f"eps_conditions={list(EPS_CONDITIONS.keys())}")
    print(f"=> {len(ARCHETYPE_CENTERS)} archetypes x {len(EPS_CONDITIONS)} eps x "
          f"{N_REPLICATIONS} reps x {N_WITHIN_CLASS_GRAPHS} graphs = "
          f"{len(ARCHETYPE_CENTERS)*len(EPS_CONDITIONS)*N_REPLICATIONS*N_WITHIN_CLASS_GRAPHS} tasks\n")

    print("Precomputing per-archetype theta_test (transition scan)...")
    theta_tests = precompute_theta_tests()
    print()

    tasks = build_a1_tasks(theta_tests)
    if MAX_TASKS_DEBUG is not None:
        tasks = tasks[:MAX_TASKS_DEBUG]
        print(f"[DEBUG] EXPA_MAX_TASKS set: truncated A1 tasks to {len(tasks)}")
    tasks_csv = _path('expA_within_class_tasks.csv')

    t0 = time.time()
    n_new = run_tasks_with_checkpointing(
        tasks, a1_task_key, a1_worker, tasks_csv, A1_KEY_COLS,
        desc="A1 graphs"
    )
    print(f"[A1 synthesis+dynamics: {n_new} new tasks in {time.time()-t0:.1f}s]\n")

    # --- ANOVA per (archetype, eps_cond, replication) ---
    tasks_df = pd.read_csv(tasks_csv)
    anova_rows = []
    for (archetype, eps_cond, replication), grp in tasks_df.groupby(
            ['archetype', 'eps_cond', 'replication']):
        sir_stats = list(zip(grp.sir_mean, grp.sir_var, grp.sir_n))
        cas_stats = list(zip(grp.cascade_mean, grp.cascade_var, grp.cascade_n))

        sir_F, sir_p, sir_msb, sir_msw, sir_k, sir_dfw = anova_from_group_stats(sir_stats)
        cas_F, cas_p, cas_msb, cas_msw, cas_k, cas_dfw = anova_from_group_stats(cas_stats)

        n_avg_sir = grp.sir_n.mean()
        n_avg_cas = grp.cascade_n.mean()
        sir_icc = icc1_from_ms(sir_msb, sir_msw, n_avg_sir)
        cas_icc = icc1_from_ms(cas_msb, cas_msw, n_avg_cas)

        anova_rows.append({
            'archetype': archetype, 'eps_cond': eps_cond, 'replication': replication,
            'n_graphs': len(grp),
            'sir_MS_between': sir_msb, 'sir_MS_within': sir_msw,
            'sir_F': sir_F, 'sir_p': sir_p, 'sir_ICC': sir_icc,
            'cascade_MS_between': cas_msb, 'cascade_MS_within': cas_msw,
            'cascade_F': cas_F, 'cascade_p': cas_p, 'cascade_ICC': cas_icc,
        })

    anova_df = pd.DataFrame(anova_rows)
    anova_df.to_csv(_path('expA_within_class_anova.csv'), index=False)

    # --- Aggregate ICC across replications ---
    summary_rows = []
    for (archetype, eps_cond), grp in anova_df.groupby(['archetype', 'eps_cond']):
        for dyn, icc_col, p_col in [('SIR', 'sir_ICC', 'sir_p'),
                                     ('CASCADE', 'cascade_ICC', 'cascade_p')]:
            icc_vals = grp[icc_col].dropna().values
            p_vals = grp[p_col].values
            if len(icc_vals) > 0:
                mean_icc = icc_vals.mean()
                std_icc = icc_vals.std(ddof=1) if len(icc_vals) > 1 else 0.0
                se_icc = std_icc / np.sqrt(len(icc_vals)) if len(icc_vals) > 1 else 0.0
                ci95 = 1.96 * se_icc
            else:
                mean_icc = std_icc = ci95 = np.nan
            summary_rows.append({
                'archetype': archetype, 'eps_cond': eps_cond, 'dynamic': dyn,
                'n_replications': len(grp),
                'ICC_mean': mean_icc, 'ICC_std': std_icc, 'ICC_ci95': ci95,
                'frac_p_gt_0.05': (p_vals > 0.05).mean(),
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(_path('expA_within_class_icc_summary.csv'), index=False)

    print("\n" + "=" * 70)
    print("PART A1 SUMMARY: ICC across replications (mean +/- 95% CI)")
    print("=" * 70)
    pd.set_option('display.width', 200)
    print(summary_df.to_string(index=False))

    return tasks_df, anova_df, summary_df, theta_tests


# ==============================================================================
# 5. PART A2: BETWEEN-CLASS DISCRIMINATION (PRODUCTION, MULTI-INSTANCE)
#
#    For each archetype, N_A2_INSTANCES independently-synthesized graphs
#    (same target coordinate = archetype center, different SA seeds) are
#    each run through the full BETA_SWEEP (SIR) and THETA_SWEEP (cascade).
#    Figures aggregate mean +/- SEM across (instance x repeat), separating
#    "which graph did we draw" from pure stochastic dynamics.
#
#    Checkpointing: task-level (archetype, instance_id) via the summary
#    CSV. Long-format sweep CSVs may contain duplicate rows if interrupted
#    between writing sweep-rows and the summary row for a task; figure
#    generation deduplicates on (archetype, instance_id, beta/theta, run_id).
# ==============================================================================

A2_KEY_COLS = ['archetype', 'instance_id']


def a2_task_key(task):
    return (task['archetype'], task['instance_id'])


def a2_worker(task):
    archetype = task['archetype']
    instance_id = task['instance_id']
    center = task['center']
    base_seed = task['base_seed']

    adj, part, loss, xhat = synthesize_graph(center, base_seed=base_seed,
                                              S=SA_STEPS, restarts=SA_RESTARTS)
    degrees = degrees_of(adj)

    sir_rows = []
    for b_idx, beta in enumerate(BETA_SWEEP):
        for r in range(N_SIR_RUNS):
            seed = make_seed(_TAG_A2_SIR, base_seed, b_idx, r)
            outcome = run_sir(adj, N, float(beta), seed)
            sir_rows.append({
                'archetype': archetype, 'instance_id': instance_id,
                'beta': float(beta), 'run_id': r, 'outcome': outcome,
                'tau_c': TAU_C[archetype],
            })

    n_seeds = max(1, int(CASCADE_SEED_FRACTION * N))
    cascade_rows = []
    for t_idx, theta in enumerate(THETA_SWEEP):
        for r in range(N_CASCADE_RUNS):
            seed_rng = np.random.RandomState(make_seed(_TAG_A2_CASCADE, base_seed, t_idx, r))
            seed_nodes = seed_rng.choice(N, size=n_seeds, replace=False).astype(np.int64)
            outcome = run_threshold_cascade(adj, N, degrees, float(theta), seed_nodes)
            cascade_rows.append({
                'archetype': archetype, 'instance_id': instance_id,
                'theta': float(theta), 'run_id': r, 'outcome': outcome,
                'RI': RI[archetype],
            })

    summary = {
        'archetype': archetype, 'instance_id': instance_id,
        'xhat_rho': xhat[0], 'xhat_eta': xhat[1],
        'xhat_c': xhat[2], 'xhat_mu': xhat[3], 'loss': loss,
    }
    return summary, sir_rows, cascade_rows


def build_a2_tasks():
    tasks = []
    for arche_idx, (arche_name, center) in enumerate(ARCHETYPE_CENTERS.items()):
        for instance_id in range(N_A2_INSTANCES):
            base_seed = make_seed(_TAG_A2_BASE, arche_idx, instance_id)
            tasks.append({
                'archetype': arche_name, 'instance_id': instance_id,
                'center': center, 'base_seed': base_seed,
            })
    return tasks


def run_part_A2():
    print("\n" + "=" * 70)
    print("PART A2: BETWEEN-CLASS DISCRIMINATION (production, multi-instance)")
    print("=" * 70)
    print(f"N_A2_INSTANCES={N_A2_INSTANCES}, "
          f"|BETA_SWEEP|={len(BETA_SWEEP)}, |THETA_SWEEP|={len(THETA_SWEEP)}, "
          f"N_SIR_RUNS={N_SIR_RUNS}, N_CASCADE_RUNS={N_CASCADE_RUNS}")
    print(f"=> {len(ARCHETYPE_CENTERS)} archetypes x {N_A2_INSTANCES} instances = "
          f"{len(ARCHETYPE_CENTERS)*N_A2_INSTANCES} synthesis tasks, each with "
          f"{len(BETA_SWEEP)*N_SIR_RUNS + len(THETA_SWEEP)*N_CASCADE_RUNS} sweep runs\n")

    tasks = build_a2_tasks()
    if MAX_TASKS_DEBUG is not None:
        tasks = tasks[:max(1, MAX_TASKS_DEBUG // 10)]
        print(f"[DEBUG] EXPA_MAX_TASKS set: truncated A2 tasks to {len(tasks)}")
    summary_csv = _path('expA_between_class_tasks.csv')
    sir_csv = _path('expA_between_class_sir.csv')
    cascade_csv = _path('expA_between_class_cascade.csv')

    completed = load_completed_keys(summary_csv, A2_KEY_COLS)
    todo = [t for t in tasks if a2_task_key(t) not in completed]
    print(f"[A2 instances] {len(completed)} already completed, "
          f"{len(todo)} remaining (of {len(tasks)} total)")

    t0 = time.time()
    a2_batch_size = min(BATCH_SIZE, max(1, N_JOBS if N_JOBS > 0 else os.cpu_count() or 4))
    for batch_start in tqdm(range(0, len(todo), a2_batch_size),
                             desc="A2 instances", unit="batch"):
        batch = todo[batch_start:batch_start + a2_batch_size]
        results = Parallel(n_jobs=N_JOBS)(delayed(a2_worker)(t) for t in batch)


        all_sir, all_cascade, all_summary = [], [], []
        for summary, sir_rows, cascade_rows in results:
            all_summary.append(summary)
            all_sir.extend(sir_rows)
            all_cascade.extend(cascade_rows)

        append_rows(sir_csv, all_sir)
        append_rows(cascade_csv, all_cascade)
        append_rows(summary_csv, all_summary)

    print(f"[A2: {len(todo)} new instances in {time.time()-t0:.1f}s]")

    sir_df = pd.read_csv(sir_csv).drop_duplicates(
        subset=['archetype', 'instance_id', 'beta', 'run_id'])
    cascade_df = pd.read_csv(cascade_csv).drop_duplicates(
        subset=['archetype', 'instance_id', 'theta', 'run_id'])
    summary_df = pd.read_csv(summary_csv).drop_duplicates(subset=A2_KEY_COLS)

    # --- Pairwise discrimination tests (Mann-Whitney U), pooled across instances ---
    print("\n--- Pairwise discrimination (Mann-Whitney U), pooled across "
          f"{N_A2_INSTANCES} instances ---")
    beta_fixed = float(BETA_SWEEP[np.argmin(np.abs(BETA_SWEEP - 0.08))])
    print(f"\nSIR at beta={beta_fixed:.4f}:")
    for a1, a2 in itertools.combinations(ARCHETYPE_CENTERS.keys(), 2):
        o1 = sir_df[(sir_df.archetype == a1) & (np.isclose(sir_df.beta, beta_fixed))]['outcome']
        o2 = sir_df[(sir_df.archetype == a2) & (np.isclose(sir_df.beta, beta_fixed))]['outcome']
        if len(o1) > 0 and len(o2) > 0:
            stat, p = stats.mannwhitneyu(o1, o2, alternative='two-sided')
            sig = "***" if p < 0.001 else ("*" if p < 0.05 else "ns")
            print(f"  {a1:8s} (mean={o1.mean():.3f}, n={len(o1)}) vs "
                  f"{a2:8s} (mean={o2.mean():.3f}, n={len(o2)}): p={p:.2e} {sig}")

    theta_fixed = float(THETA_SWEEP[np.argmin(np.abs(THETA_SWEEP - 0.20))])
    print(f"\nCASCADE at theta={theta_fixed:.4f}:")
    for a1, a2 in itertools.combinations(ARCHETYPE_CENTERS.keys(), 2):
        o1 = cascade_df[(cascade_df.archetype == a1) & (np.isclose(cascade_df.theta, theta_fixed))]['outcome']
        o2 = cascade_df[(cascade_df.archetype == a2) & (np.isclose(cascade_df.theta, theta_fixed))]['outcome']
        if len(o1) > 0 and len(o2) > 0:
            stat, p = stats.mannwhitneyu(o1, o2, alternative='two-sided')
            sig = "***" if p < 0.001 else ("*" if p < 0.05 else "ns")
            print(f"  {a1:8s} (mean={o1.mean():.3f}, n={len(o1)}) vs "
                  f"{a2:8s} (mean={o2.mean():.3f}, n={len(o2)}): p={p:.2e} {sig}")

    return sir_df, cascade_df, summary_df


# ==============================================================================
# 6. FIGURES
# ==============================================================================

ARCHETYPE_COLORS = {
    'I-ER':    '#4C72B0',
    'II-WST':  '#55A868',
    'III-WSD': '#C44E52',
    'IV-SF':   '#8172B2',
    'V-CP':    '#CCB974',
}
EPS_COLORS = {'eps_half': '#4C72B0', 'eps_full': '#C44E52'}


def make_sir_cascade_figures(sir_df, cascade_df):
    # --- Figure 1: SIR epidemic curves, mean +/- SEM across (instance x repeat) ---
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for arche_name in ARCHETYPE_CENTERS:
        sub = sir_df[sir_df.archetype == arche_name]
        means = sub.groupby('beta')['outcome'].mean()
        sems = sub.groupby('beta')['outcome'].sem()
        ax.errorbar(means.index, means.values, yerr=sems.values,
                     label=f"{arche_name} (tau_c={TAU_C[arche_name]:.3f})",
                     color=ARCHETYPE_COLORS[arche_name], marker='o',
                     markersize=4, capsize=2, linewidth=1.5)
        ax.axvline(TAU_C[arche_name], color=ARCHETYPE_COLORS[arche_name],
                   linestyle='--', alpha=0.4)
    ax.set_xlabel(r'Transmission probability $\beta$')
    ax.set_ylabel('Final outbreak size (fraction of N)')
    n_total = N_A2_INSTANCES * N_SIR_RUNS
    ax.set_title('Simple Contagion (SIR): Epidemic Curves by Archetype\n'
                  f'(mean $\\pm$ SEM, n={n_total} = {N_A2_INSTANCES} instances '
                  f'x {N_SIR_RUNS} repeats per point)')
    ax.set_xscale('log')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(_path('expA_fig_sir_curves.png'), dpi=150)
    plt.close(fig)

    # --- Figure 2: Cascade adoption curves ---
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for arche_name in ARCHETYPE_CENTERS:
        sub = cascade_df[cascade_df.archetype == arche_name]
        means = sub.groupby('theta')['outcome'].mean()
        sems = sub.groupby('theta')['outcome'].sem()
        ax.errorbar(means.index, means.values, yerr=sems.values,
                     label=f"{arche_name} (RI={RI[arche_name]:.3f})",
                     color=ARCHETYPE_COLORS[arche_name], marker='o',
                     markersize=4, capsize=2, linewidth=1.5)
    ax.set_xlabel(r'Adoption threshold $\theta$')
    ax.set_ylabel('Final adoption fraction')
    n_total = N_A2_INSTANCES * N_CASCADE_RUNS
    ax.set_title('Complex Contagion (Threshold Model): Cascade Curves by Archetype\n'
                  f'(mean $\\pm$ SEM, n={n_total} = {N_A2_INSTANCES} instances '
                  f'x {N_CASCADE_RUNS} repeats per point)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(_path('expA_fig_cascade_curves.png'), dpi=150)
    plt.close(fig)

    print("\nFigures saved: expA_fig_sir_curves.png, expA_fig_cascade_curves.png")


def make_icc_figure(anova_df):
    """
    Boxplots of ICC across the N_REPLICATIONS replications, grouped by
    archetype and eps_condition, one subplot per dynamic (SIR/CASCADE).
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)

    for ax, (dyn, icc_col) in zip(axes, [('SIR', 'sir_ICC'), ('CASCADE', 'cascade_ICC')]):
        archetypes = list(ARCHETYPE_CENTERS.keys())
        eps_conds = list(EPS_CONDITIONS.keys())
        positions = []
        data = []
        colors = []
        labels = []
        width = 0.35
        for i, arche in enumerate(archetypes):
            for j, eps_cond in enumerate(eps_conds):
                vals = anova_df[(anova_df.archetype == arche) &
                                 (anova_df.eps_cond == eps_cond)][icc_col].dropna().values
                pos = i + (j - 0.5) * width * 1.2
                positions.append(pos)
                data.append(vals if len(vals) > 0 else [np.nan])
                colors.append(EPS_COLORS[eps_cond])
                if i == 0:
                    labels.append(eps_cond)

        bp = ax.boxplot(data, positions=positions, widths=width, patch_artist=True,
                         showfliers=True, manage_ticks=False)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.axhline(0, color='black', linewidth=0.8, linestyle=':')
        ax.set_xticks(range(len(archetypes)))
        ax.set_xticklabels(archetypes, rotation=30, ha='right')
        ax.set_title(f'{dyn}: ICC(1) across {N_REPLICATIONS} replications')
        ax.set_ylabel('ICC(1)')
        ax.grid(alpha=0.3, axis='y')

        # legend via proxy patches
        from matplotlib.patches import Patch
        handles = [Patch(facecolor=EPS_COLORS[e], alpha=0.6, label=e) for e in eps_conds]
        ax.legend(handles=handles, fontsize=8, loc='upper right')

    fig.suptitle('Macroscopic Equivalence Postulate: ICC(1) Distributions\n'
                  '(~0 = postulate supported; ~1 = violated; box = IQR across '
                  'independent replications)', fontsize=12)
    fig.tight_layout()
    fig.savefig(_path('expA_fig_icc_distributions.png'), dpi=150)
    plt.close(fig)
    print("Figure saved: expA_fig_icc_distributions.png")


# ==============================================================================
# 7. JIT WARM-UP
#
#    numba's file-based cache (cache=True) is shared across processes, but
#    concurrent first-time compilation from multiple joblib worker
#    processes can race. Calling each @njit function once here (in the
#    main process, single-threaded, with trivial inputs) populates the
#    cache before any Parallel(...) call, so workers only ever READ the
#    cache.
# ==============================================================================

def warm_up_jit():
    print("Warming up numba JIT cache (one-time compilation)...")
    t0 = time.time()
    center = ARCHETYPE_CENTERS['I-ER']
    adj, part, loss, xhat = synthesize_graph(center, base_seed=0, S=200, restarts=1)
    degrees = degrees_of(adj)
    run_sir(adj, N, 0.1, seed=0, max_steps=20)
    seed_nodes = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    run_threshold_cascade(adj, N, degrees, 0.2, seed_nodes, max_steps=20)
    find_transition_theta(adj, degrees, THETA_SCAN_GRID[:3], n_seeds=2, rng_seed=0)
    print(f"  done in {time.time()-t0:.1f}s\n")


# ==============================================================================
# 8. MAIN
# ==============================================================================

if __name__ == '__main__':
    t_total0 = time.time()

    n_cores = os.cpu_count() or 1
    effective_jobs = n_cores if N_JOBS == -1 else N_JOBS
    print(f"Detected {n_cores} CPU cores. N_JOBS={N_JOBS} "
          f"(effective: {effective_jobs}).\n")

    warm_up_jit()

    tasks_df, anova_df, icc_summary_df = run_part_A1()[:3]
    sir_df, cascade_df, a2_summary_df = run_part_A2()

    make_sir_cascade_figures(sir_df, cascade_df)
    make_icc_figure(anova_df)

    print(f"\n=== TOTAL RUNTIME: {(time.time()-t_total0)/60:.2f} minutes ===")
    print("\nAll outputs written to current directory:")
    for f in ['expA_within_class_tasks.csv', 'expA_within_class_anova.csv',
              'expA_within_class_icc_summary.csv', 'expA_between_class_tasks.csv',
              'expA_between_class_sir.csv', 'expA_between_class_cascade.csv',
              'expA_fig_sir_curves.png', 'expA_fig_cascade_curves.png',
              'expA_fig_icc_distributions.png']:
        print(f"  {f}")
    print("\nRe-running this script will SKIP all completed tasks (checkpointed "
          "via the *_tasks.csv files) and only compute what's missing.")
