# Dual-Contagion Validation of the Postulate of Macroscopic Equivalence

## Purpose

Empirically test the **Postulate of Macroscopic Equivalence**:

> Two graphs $G_1$, $G_2$ with $\|x(G_1) - x(G_2)\|_2 \le \epsilon$ belong to the same Macroscopic Equivalence Class and are dynamically substitutable for diffusion experiments.

## Core scaling axes

1. **Multi-replication** (`N_REPLICATIONS` independent seeds): the entire within-class ANOVA/ICC analysis (Part A1) is repeated under `N_REPLICATIONS` independent random seeds, each producing its own set of `N_WITHIN_CLASS_GRAPHS` perturbed graphs. This yields a **distribution** of ICC estimates per `(archetype, dynamic, eps-condition)` — reporting mean $\pm$ CI rather than a single-seed point estimate.

2. **Both $\frac{{\epsilon}}{{2}}$ and full $\epsilon$ perturbation radii**: in addition to $\frac{{\epsilon}}{{2}} = 0.025$, the script also tests the **full $\epsilon = 0.05$ radius** — directly answering whether the postulate holds at the nominal equivalence-class boundary, or only safely inside it.

3. **Multi-instance between-class comparison (Part A2)**: `N_A2_INSTANCES` independently synthesized graphs (same target coordinate, different SA seeds) are each run through the full $\frac{{\beta}}{{\theta}}$ sweep (instead of one center-graph per archetype). Curves report mean $\pm$ SEM across (instance $\times$ stochastic-repeat), separating “which specific graph did we happen to draw” from pure stochastic dynamics.

## Statistical Method

* **One-way ANOVA** (`H_0`: all `k` perturbed graphs are dynamically exchangeable) computed from per-graph `(mean, var, n)` summary statistics rather than raw run-level arrays. This makes checkpoints tiny (one row per graph) and generalizes to unbalanced designs.
* **ICC(1)** as effect size: fraction of total outcome variance attributable to between-graph (structural) vs within-graph (stochastic) sources.
* **SIR**: discrete-time, `gamma = 1` (Reed–Frost / bond-percolation mapping), HMF threshold
  `tau_c = 1 / ((N-1) * rho_bar * (1 + eta_bar))`.
* **Cascade**: Centola/Granovetter fractional-threshold complex contagion, with per-archetype `theta_test` found via variance-maximizing scan (`find_transition_theta`) — computed **once per archetype** (not per replication/instance), since it characterizes the archetype’s bootstrap-percolation transition, not any single graph’s idiosyncrasy.

## Engineering

* **Checkpointing**: results are written incrementally to CSV. Re-running this script skips any `(archetype, eps_condition, replication, graph_id)` combination already present in the output CSV — safe to interrupt and resume at any point.
* **Parallelization**: the script uses `joblib.Parallel` with `N_JOBS` workers (default: all cores). Each worker JIT-compiles the numba functions independently (~1–3 s one-time cost per worker process).
* **Progress**: `tqdm` progress bars.

## Configuration

All tunable parameters are declared in the `CONFIGURATION` block within the script with inline justification. To scale further (more replications, finer sweeps, etc.), edit these constants directly — the rest of the script adapts automatically.

## Outputs

* `expA_within_class_tasks.csv`
  One row per `(archetype, eps_cond, replication, graph_id)`: `xhat`, `loss`, and `(mean, var, n)` for SIR and cascade outcomes. THE CHECKPOINT FILE.

* `expA_within_class_anova.csv`
  One row per `(archetype, eps_cond, replication, dynamic)`: `F`, `p`, `ICC`, `MS_between`, `MS_within`.

* `expA_within_class_icc_summary.csv`
  Aggregated across replications: mean/std/CI of ICC per `(archetype, eps_cond, dynamic)`.

* `expA_between_class_tasks.csv`
  One row per `(archetype, instance_id)`: `xhat`, `loss`. THE CHECKPOINT FILE for A2 synthesis.

* `expA_between_class_sir.csv`
  Long format: `(archetype, instance_id, beta, run_id, outcome)`.

* `expA_between_class_cascade.csv`
  Long format: `(archetype, instance_id, theta, run_id, outcome)`.

* `expA_fig_sir_curves.png`
  SIR epidemic curves, mean $\pm$ SEM across instances $\times$ repeats.

* `expA_fig_cascade_curves.png`
  Cascade curves, same aggregation.

* `expA_fig_icc_distributions.png`
  ICC distributions across replications, $\frac{{\epsilon}}{{2}}$ vs full-$\epsilon$, per archetype/dynamic.