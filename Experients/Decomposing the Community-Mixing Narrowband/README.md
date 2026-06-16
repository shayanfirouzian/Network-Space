# Decomposing the Community-Mixing Narrowband

## Purpose

Refresh Symmetry, Partition Granularity ($K = 2:5$), and Generalization Across Each Archetype’s Region ($N = 100$)

## Background

The Phase II exhaustive sweep found 86.3% of feasible coordinates have $\mu \in [0.45, 0.60]$. The validated pilot showed this narrowband is partly a refresh-heuristic artifact (C1: $K = 2 +$ always-minimize-cross-edges refresh stays flat near $\mu \sim 0.5$ regardless of target) and partly a genuine consequence of $(\rho,\eta,c)$ (C2/C3: $K=2/K=3$ with target-aware “symmetric” refresh recover a monotonic response but still do not reach $\mu \in {0,1}$).

## Core scaling axes

1. **$K = 2, 3, 4, 5$**. Five conditions:

   * C1: $K=2$, always-minimize refresh — the Phase II baseline
   * C2: $K=2$, symmetric (target-aware) refresh
   * C3: $K=3$, symmetric refresh
   * C4: $K=4$, symmetric refresh
   * C5: $K=5$, symmetric refresh

   Tests whether achievable-$\mu$ width keeps growing with partition granularity or saturates — directly informing “how many communities is enough”.

2. **Finer $\mu_{\text{target}}$ grid**: 81 points (step 0.0125, was 11 points / step 0.1).

3. **Multi-instance $(\rho,\eta,c)$ sampling**: `N_INSTANCES_PER_ARCHETYPE` points per archetype (the class center plus `N_INSTANCES_PER_ARCHETYPE - 1` points perturbed within $\epsilon/2 = 0.025$ in $(\rho,\eta,c)$-space, mirroring previous experiment’s within-class perturbation). Tests whether the C1 vs C2:C5 pattern is a property of the archetype class, not just its centroid.

## Proxy Formula and Verification

  - $proxy(u,v) = \left[E(u\rightarrow a)-E(u\rightarrow b)\right]+\left[E(v\rightarrow b)-E(v\rightarrow a)\right]+2\times A_{u,v}$
  - cross_edges $\mathrel{:=}$ cross_edges + proxy(u,v)

Verified by brute-force recount at script startup for every $K \in {2,3,4,5}$.

* $\operatorname*{arg\,min}_{\mu_{\text{target}} < \text{midpoint}_K} proxy(u,v)$ | minimize $\mu_{\text{target}} < \text{midpoint}_K$: pick the swap minimizing proxy.
* $\operatorname*{arg\,max}_{\mu_{\text{target}} \ge \text{midpoint}_K} proxy(u,v)$ | maximize $\mu_{\text{target}} < \text{midpoint}_K$: pick the swap minimizing proxy.

Midpoints (expected cross-fraction of a random graph under each balanced partition):
$K=2 \to 0.5051$, $K=3 \to 0.6733$, $K=4 \to 0.7576$, $K=5 \to 0.8081$.

## Engineering

* **Checkpointing**: results appended incrementally to CSV, keyed by `(archetype, instance_id, mu_target, condition)`. Re-running skips completed tasks.
* **Parallelization**: `joblib.Parallel`, `N_JOBS` workers (default: all cores).
* **Progress**: `tqdm`.
* All scale parameters are overridable via environment variables for quick reduced-scale verification runs (see `CONFIGURATION` in the script).

## Outputs

* `expB_sweep_tasks.csv`
  One row per `(archetype, instance_id, mu_target, condition)`: `xhat`, `loss`. **THE CHECKPOINT FILE.**

* `expB_width_summary.csv`
  One row per `(archetype, instance_id, condition)`: achievable $\mu$-range `[min, max]`, width, feasible count.

* `expB_width_by_K.csv`
  Aggregated across instances: mean ± CI width per `(archetype, condition / K)`.

* `expB_fig_mu_range.png`
  Achieved $\mu$ vs target $\mu$, mean ± band across instances, faceted by archetype, one line per condition.

* `expB_fig_width_vs_K.png`
  Achievable width vs $K$, per archetype, for the symmetric conditions `C2-C5` (with `C1` as a `K=2` reference point).