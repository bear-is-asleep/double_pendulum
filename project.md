# Double Pendulum Surrogate: Training Strategies

## Goal

Compare four training strategies for a vanilla MLP that learns double-pendulum trajectories under chaos. Citations live in `writeup/refs.bib`.

1. Baseline
2. Curriculum
3. Active
4. Progressive

HNN / physics-structured models are **out of scope for now** (see Future directions).

## Setup

- System: planar double pendulum, numerical RK4 ground truth (no closed form).
- **Surrogate (ANN):** MLP
  $$
  (t, \theta_{10}, \theta_{20}, \omega_{10}, \omega_{20}, m_{1}, m_{2}, g)
  \to (\sin\theta_{1}, \cos\theta_{1}, \sin\theta_{2}, \cos\theta_{2}, \omega_{1}, \omega_{2})
  $$
  Output dim **6**. Lengths fixed, so not inputs. Decode $\theta_i = \mathrm{atan2}(\sin\theta_i, \cos\theta_i)$ only for plots / physics checks, not for the loss target.
- **Angle representation:** store labels as $(\sin\theta, \cos\theta)$ (and optionally wrapped $\theta$ for debugging). Raw $\theta$ MSE is wrong near the cut: $0$ vs $1.999\pi$ look far but are close on the circle [@yang2020csl].
- **Loss (what literature does + what we use):**
**Double-pendulum / physics-ML defaults (Euclidean on dynamics):**
  - LNNs on double pendulum train with an L2 / MSE-style loss on **accelerations** (Euler–Lagrange residual), not circular $\theta$ error [@cranmer2020lnn].
  - HNNs use L2 on the **symplectic gradient** of $H$ vs true $(\dot q, \dot p)$ [@greydanus2019hnn]; energy-regularized variants add an extra energy-level term [@eichelsdoerfer2021physics].
  - Modular / related Lagrangian nets similarly regress dynamics with MSE on derived accelerations under small-data double-pendulum settings [@lu2022modlanets].
  - Baseline MLPs in those papers are usually the same Euclidean targets (next-state or $\ddot q$), so **plain $\theta$ MSE is common when people ignore the branch cut**.
  **When the target is an angle on a circle:**
  - Periodicity / discontinuous boundaries of angle regression are a known failure mode; circular encodings fix it [@yang2020csl].
  - Recipe we lock: **predict $(\sin\theta, \cos\theta)$, supervise with MSE**; decode with $\mathrm{atan2}$ when a scalar angle is needed.
  **This project (ANN surrogate):**
  - Angles: **sin/cos MSE** on the four trig heads (same for train and eval).
  - Velocities: ordinary MSE on $\omega_1, \omega_2$.
  - Weights `loss_weight_theta`, `loss_weight_omega` from config (theta weight applies to the four sin/cos channels together).
  - We are **not** using HNN/LNN residual losses in the main experiments (deferred).
- Shared constraints across the four strategies: same frozen test set, same locked FC width / final depth / LR / batch / $k$, same stage definitions. Progressive starts shallower (`depth_start`) and grows to locked `hidden_depth`; width fixed.
- **ICs need not start at rest.** Sampler draws $\omega_{10}, \omega_{20}$ (bounded). Needed so zero-$g$ stages still move.



## Experimental controls

- **Locked from baseline establishment:** FC width/depth, train subsample $k$, learning rate, batch size. Reuse for all four strategies.
- **Allowed to float:** sample / trajectory budget used per run **and** epochs / early stopping. Report both. No forced equal-N across methods.
- Active: **QbC-style** deep ensemble (`n_ensemble`, default **2**). Uncertainty = mean ensemble std over $t$ of the 6 heads. **Do not train on high-var only:** each draw mixes uniform vs uncertainty with weight $\lambda$ that **increases with epoch** (`lambda_start` → `lambda_end` in model YAML), so late training oversamples uncertain ICs more. Progressive = grow **depth** only (same locked width); schedule in that model YAML. Active query-loop details deferred.



## Config (YAML)

- **Sampler / stage bounds:** `configs/data/full.yaml` (shared). Keys include `stage_min`, `num_stages` (max id), `g_min`, `g_max`, `g_zero`, `g_low`, `g_high`, `omega0_max`, `omega_traj_max`, energy-drift tol, Stage 5/6 `pe_min`, `stage6` mass gap, IC boxes `stage0`–`stage6`, $T$, $dt$, discard retries. There is no `g_stage1` key.
- **Per-model configs:** `configs/models/<model_name>.yaml` (one file per model / strategy variant). Holds architecture, LR, batch, $k$, loss weights, seeds, and Active-only `n_ensemble`, etc. Sampler code loads sampler YAML; train/eval load the model YAML for that run.



## Curriculum stages

Seven stages, ids **0–6**. Bounds live in `configs/data/full.yaml` (`stage_min`, `num_stages` = max id). Do not hardcode numbers in train scripts.

**Stage 0 (added later):** same physics and masses as stage 1 ($m_2 = 0$, $g = g_{\mathrm{zero}}$), same `stage0` IC box as stage 1, but **exactly one** of $\omega_{10},\omega_{20}$ is fixed to $0$ per draw (the other is drawn from the box). Easiest uniform-spin subset before stage 1.

Stage 1 is the full $m_2 = 0$ zero-$g$ box (both $\omega_0$ free). Old pre–stage-0 stage $k$ is now id $k+1$ for $k \ge 1$. Zero-gravity **equal** masses are **stage 2**.


| Id  | Was | Regime                                                             |
| --- | --- | ------------------------------------------------------------------ |
| 0   | new | like stage 1, one $\omega_0 = 0$                                   |
| 1   | new | $m_2 = 0$, $m_1 = m_{\mathrm{equal}}$, $g = g_{\mathrm{zero}} = 0$ |
| 2   | 1   | $g = 0$, equal nonzero masses                                      |
| 3   | 2   | small ICs, equal masses, $g_{\mathrm{low}}$                        |
| 4   | 3   | small ICs, equal masses, $g_{\mathrm{high}}$                       |
| 5   | 4   | full IC box, equal masses, $g \sim [g_{\min}, g_{\max}]$, `pe_min` |
| 6   | 5   | free masses / ICs, $g \sim [g_{\min}, g_{\max}]$, hardness filters |


1. **Zero second mass, zero gravity, one joint at rest.** Same as stage 1, but after drawing ICs set $\omega_{10} = 0$ or $\omega_{20} = 0$ (uniform choice). Motion stays uniform in angles with one constant $\omega_i$.
2. **Zero second mass, zero gravity.** $m_2 = 0$, $m_1 = m_{\mathrm{equal}}$, $g = g_{\mathrm{zero}}$. Draw $\theta_{10},\theta_{20},\omega_{10},\omega_{20}$ from the stage 1 box. Motion is uniform in both angles (see Stage rules). Do not turn gravity on here. Gravity first appears at stage 3.
3. Zero gravity, equal masses (nonzero $\omega_0$ so the system still moves).
4. Small ICs, somewhat stable motion, equal masses, low gravity (`g_low`).
5. Same IC box as Stage 3, equal masses, higher gravity (`g_high`).
6. Full IC box, equal masses, $g$ sampled in $[g_{\min}, g_{\max}]$ (typically $g_{\min}=0$).
7. Hard free regime: angles / $\omega_0$ / masses free in their boxes; $g$ also sampled in $[g_{\min}, g_{\max}]$; must pass **hardness filters** (below). Lengths still fixed.



## Sampler generator

Build a stage-aware sampler that draws ICs and physical parameters, then **accepts only well-behaved** sims.

### Draw fields

- angles: $\theta_{10},\ \theta_{20}$
- angular velocities: $\omega_{10},\ \omega_{20}$
- masses: $m_{1},\ m_{2}$ (stages 0–1 fix $m_2 = 0$ and $m_1 = m_{\mathrm{equal}}$; equal on stages 2–5; may differ on stage 6)
- lengths: $\ell_{1}=\ell_{2}=\ell$ fixed
- gravity: $g$



### Global safety (all stages; YAML)

- $g \in [g_{\min}, g_{\max}]$ with $g_{\min} \ge 0$ (default $g_{\min}=0$); **never negative**
- $|\omega_{10}|, |\omega_{20}| \le \omega_{0,\max}$
- Stages 2–6: $m_1, m_2 \ge m_{\min} > 0$ (keeps the coupled EOM denominator safe: $2m_1 + m_2(1-\cos 2\delta) \ge 2m_1$). Reject $m_2 \le 0$ on these stages.
- **Stages 0–1 only:** $m_2 = 0$ exactly and $m_1 = m_{\mathrm{equal}} \ge m_{\min}$. Skip `mass_below_min` and `m2_nonpositive` for that $m_2$. Reject the row if $m_2 \ne 0$ or $m_1 \ne m_{\mathrm{equal}}$. Stage 0 also requires exactly one of $\omega_{10},\omega_{20}$ to be $0$.
- Do **not** set $m_1 = 0$ on any stage. `derivatives` uses a different ODE when `m1 == 0`, and `integrate_rk4` rewrites that initial state. Stages 0–1 use only the existing `m2 == 0` branch.
- After a short probe integrate (or full traj), **discard** if any $|\omega(t)| > \omega_{\mathrm{traj,max}}$, if state has NaN/Inf, or if energy is non-finite
- **Energy drift:** total energy $E(t)$ should stay nearly constant (undamped). Discard if $\max_t |E(t)-E(0)| > \varepsilon_E$ (or relative tol); both set in sampler YAML
- Optionally reject if angle unwrap rate implies pathological spin beyond config
- Retry draw until `n` accepted rows or hit max attempts (log reject rate)



### Potential and total energy

Match `cartesian`: pivot at origin, $+y$ up, $\theta=0$ hanging down ($y_i = -\ell_i\cos\theta_i$). PE zero at the hanging equilibrium so $V\ge 0$ and `pe_min` means raised / hard:

$$
V(\theta_1,\theta_2) = (m_1+m_2)\,g\,\ell_1\,(1-\cos\theta_1) + m_2\,g\,\ell_2\,(1-\cos\theta_2)
$$

With $\ell_1=\ell_2=\ell$ from YAML. Kinetic energy (planar double pendulum):

$$
\begin{aligned}
T &= \tfrac12 (m_1+m_2)\,\ell_1^2\,\omega_1^2 + \tfrac12 m_2\,\ell_2^2\,\omega_2^2 \\
&\quad + m_2\,\ell_1\,\ell_2\,\omega_1\,\omega_2\,\cos(\theta_1-\theta_2)
\end{aligned}
$$

Total energy $E = T + V$. Sampler hardness uses $V_0 = V(\theta_{10},\theta_{20})$ at the draw (scales with $g$, so near-zero $g$ fails `pe_min`). Drift check uses $E(t)$ on the probe / full traj.

### Stage rules

- Stage 0: same as stage 1 for $(g, m_1, m_2)$ and the stage 0 IC box; one $\omega_{i0} = 0$. Same uniform-angle motion as stage 1.
- Stage 1: $g = g_{\mathrm{zero}} = 0$, $m_2 = 0$, $m_1 = m_{\mathrm{equal}}$, stage 1 IC box. Do not edit `derivatives`. The `m2 == 0` branch already returns $\ddot\theta_1 = -g\sin\theta_1 / \ell_1$ and $\ddot\theta_2 = 0$. With $g = 0$ that is $\ddot\theta_1 = \ddot\theta_2 = 0$, so $\omega_i(t) = \omega_{i0}$ and $\theta_i(t) = \theta_{i0} + \omega_{i0} t$ before wrapping. A stage-1 test must check that closed form on a short RK4 run.
- Stage 2: $g = g_{\mathrm{zero}}$, equal masses, stage 2 IC box. This is the old stage 1.
- Stage 3: $g = g_{\mathrm{low}}$, equal masses, small IC box.
- Stage 4: $g = g_{\mathrm{high}}$, equal masses, same small IC box as stage 3.
- Stage 5: $g \sim \mathrm{Unif}[g_{\min}, g_{\max}]$, equal masses, full IC box, plus $V_0 \ge V_{5,\min}$ (YAML `stage5.pe_min`) so near-zero-$g$ soft cases get filtered. This is the old stage 4.
- Stage 6: $g \sim \mathrm{Unif}[g_{\min}, g_{\max}]$, free masses / ICs in YAML ranges, plus hardness filters:
  - initial potential $V_0 \ge V_{6,\min}$ (YAML `stage6.pe_min`; with $g_{\min}=0$, low-$g$ draws often fail PE and get rejected)
  - $|m_1 - m_2| \ge \Delta m_{6,\min}$ (YAML `stage6.mass_diff_min`)
  - plus global safety / probe-sim / energy-drift checks above

API: `sample(stage, n, cfg) ->` accepted parameter rows. $m_2 = 0$ on stages 0–1 only. Stages 2–6 keep $m_2 \ge m_{\min} > 0$.

## Angles, wrapping, and loss

- Simulator may integrate unwrapped $\theta$; **before store / train**, wrap to $(-\pi, \pi]$ (YAML `angle_wrap: negpi_pi`) then store $(\sin\theta, \cos\theta)$ labels (and $\omega$).
- Never use plain $\theta$ MSE across the branch cut. Train/eval: **sin/cos MSE** + $\omega$ MSE (see Setup). Output dim 6.
- Store $\omega$ and energy series in the dataset / sim exports for plots / PE-KE.



## Data (before baseline establishment)

1. Load `configs/data/full.yaml`. Sample train / test ICs for stages 0 to 6 with accept/reject.
2. Simulate full trajectories; store wrapped $\theta$, $\omega$, energy, and params. Same $T$, $dt$ for all.
3. **On-disk format (keep simple):** NumPy `.npz` under `data/` (one file per split/stage or a small set of arrays). Keys at minimum: params/ICs, $t$, $\sin/\cos$ targets, $\omega$, $E$ (and stage id). Exact key names live in the saver code; document them once in a module docstring.
4. Train subsample stride $k$ is train-time only; disk keeps full resolution.
5. Freeze the test set. Carve a **validation** slice from the train pools (fraction in model YAML); never touch frozen test for model selection.
6. Train rows are pointwise: one sample = $(t, \mathrm{IC}, m, g) \to$ 6-D target at that $t$.



### Suggested pool sizes (starting point; budget may float)


| Stage | Train pool | Test (frozen) | Why                                                        |
| ----- | ---------- | ------------- | ---------------------------------------------------------- |
| 1     | 1024       | 128           | $m_2=0$, $g=0$; uniform $\theta(t)$                        |
| 2     | 2048       | 256           | Zero-$g$, equal masses; motion from $\omega_0$             |
| 3     | 4096       | 512           | Small IC, $g_{\mathrm{low}}$                               |
| 4     | 4096       | 512           | Small IC, $g_{\mathrm{high}}$                              |
| 5     | 8192       | 1024          | Full IC, equal masses, $g\sim[g_{\min},g_{\max}]$ + PE min |
| 6     | 16384      | 2048          | Free $m$/IC; $g\sim[g_{\min},g_{\max}]$ + hardness         |


Pools are upper caps / generation targets. Actual **training budget and epochs float** per run; log what was used.

## Baseline establishment

Under uniform / baseline training (mix from stage pools, or a floating subset):

1. Lock FC width / depth (smallest net that clears stages 1–6 under sin/cos + $\omega$ MSE). **Pass bar** = per-stage val loss $\le$ `stage_pass_mse` in model YAML (tune later).
2. Lock train subsample $k$.
3. Lock LR and batch size.

No fixed sample count. Report epochs, steps, and trajectories seen for the locked run.

## Experiments (after baseline establishment)


| #   | Name        | What it does                                                                     |
| --- | ----------- | -------------------------------------------------------------------------------- |
| 1   | Baseline    | Uniform (or mixed) data, fixed net, one shot                                     |
| 2   | Curriculum  | Stage 0 to 6 in order; train on current stage (discard old)                      |
| 3   | Active      | QbC ensemble std; mix uniform+uncertainty; $\lambda$↑ with epoch; `n_ensemble=2` |
| 4   | Progressive | Grow **depth** over training (width = locked baseline); end depth = locked       |




## Checkpoints and run folders

Many hyperparameter / strategy iterations (baseline sizes, active-learning variants, etc.). Each trained net gets its own run directory under `runs/`. Weights follow the official PyTorch pattern: save `state_dict` (recommended), or a general checkpoint dict for resume [@pytorch_saving_models].

### Layout

```text
runs/
  <run_id>/
    config.yaml          # architecture, loss, data split, seeds, LR, batch, k, ...
    metrics.jsonl        # one JSON object per logged step/epoch
    summary.json         # final eval: per-stage MSE, forgetting, epochs, wall time
    checkpoints/
      last.pt            # latest general checkpoint (resume)
      best.pt            # lowest validation loss (auto); see Policy
```

`<run_id>` example: `baseline_ann_w128_d3_k4_seed0` or `curriculum_ann_seed0`. Unique, filesystem-safe, encodes the knobs that differ.

### What goes in each checkpoint (`.pt`)

Use `torch.save` on a dict [@pytorch_saving_models]:

- `model_state_dict` — `model.state_dict()`
- `optimizer_state_dict` — for resume (optional on `best.pt` if inference-only)
- `epoch`, `global_step`
- `val_metric` — validation loss at save time
- `config` snapshot or path to `config.yaml` (architecture must be rebuildable before `load_state_dict`)

Do **not** pickle the whole `nn.Module` as the primary artifact. Rebuild the class from `config.yaml`, then `load_state_dict`. That is the portable path the docs recommend.

### Loader

One loader API for app / eval / ablations:

1. Read `runs/<run_id>/config.yaml`.
2. Build ANN from config (same code path as training).
3. `ckpt = torch.load(..., map_location=device, weights_only=True)` on `best.pt` or `last.pt`.
4. `model.load_state_dict(ckpt["model_state_dict"])`; `model.eval()` for inference.
5. Optionally restore optimizer from `last.pt` to continue training.

Index file optional: `runs/index.csv` with columns `run_id, experiment, status, best_metric, path` so sweeps stay searchable without opening every folder.

### Policy

- Every baseline establishment trial and every experiment #1–4 variant writes a run folder (including failed / dominated baselines you may still plot later).
- `best.pt` **automation:** after each epoch (or eval interval), if current validation loss is strictly lower than the best so far, overwrite `best.pt`. `last.pt` always = most recent epoch. No manual pick.
- Keep `best.pt` + `last.pt`; prune older epoch dumps if disk hurts.
- Logs (`metrics.jsonl`, `summary.json`) live beside checkpoints so plots and `results.csv` can be rebuilt without retrain.



## Pipeline

1. YAML config + sampler (safety + Stage 6 hardness).
2. Build / freeze data (wrapped $\theta$, $\omega$, energy).
3. Visualization on ground-truth data (apps, gifs, plots); stub hooks for surrogate + eval plots.
4. Baseline establishment: lock size, $k$, LR, batch; write `runs/`.
5. Run strategies 1–4 (floating budget / epochs).
6. Eval: sin/cos MSE + $\omega$ MSE per stage on frozen test; feed existing viz/plot CLIs.
  - **Error-vs-$t$ (per stage):** for each stage's frozen test set, compute mean prediction error (same loss as train: sin/cos + $\omega$) as a function of $t$. Log curves + optional scalar summaries. This is the main time-resolved forgetting / degradation view.
  - **Forgetting:** compare error-vs-$t$ (and aggregate loss) on Stage 1 **after** later-stage / full training vs a Stage-1-only reference; rise = forgetting. Stage 1 here is the $m_2 = 0$ pool. Same check can be repeated for other early stages if useful.



## Deliverables

- `configs/data/full.yaml` + `configs/models/*.yaml` + sampler module with documented stage ranges / constraints
- `runs/` tree with configs, checkpoints, metrics (PyTorch `state_dict` / checkpoint dicts [@pytorch_saving_models])
- Locked baseline note (shared FC width / depth, $k$, LR, batch size; epochs / samples used) pointing at the winning `run_id`
  - Keep failed / dominated baseline trials under `runs/` for later plots (e.g. param size vs final loss)
  - Save training / eval curves for the locked baseline (or store enough in `metrics.jsonl` / `summary.json` to rebuild them)
- `results.csv`: per-stage sin/cos MSE + $\omega$ MSE + forgetting deltas + error-vs-$t$ summary for all 4 strategies
- Per-run error-vs-$t$ curves (e.g. under `runs/<run_id>/` or rebuilt from logged arrays); one curve family per stage
- Training curves plot (from `metrics.jsonl`; uses plot helpers from Step **4b**)
- **Visualization (Step 4b, built before training):** interactive apps, gifs, time-series plots for ground-truth data; stage / split filters; sin/cos (not raw $\theta$), $\omega$, PE, KE
  - **Wiring only after NNs exist:** sim vs ANN overlay in app, surrogate trajectories from checkpoint loader (Step 8), eval error-vs-$t$ and training curves fed from Step 10 exports. No second viz stack at the end of the pipeline.
- Citations: `writeup/refs.bib`



## Build order

Check off in GitHub or any Markdown preview that supports task lists (`- [ ]` / `- [x]`).

- [x] **1. Simulator** — `srcs/` RK4, $E=T+V$, wrap angles
- [x] **2. Config YAML** — `configs/data/full.yaml`, `configs/models/*.yaml`
- [x] **3. Sampler** — five-stage accept/reject. Ids in that code are stale until step 3b.
- [x] **3b. Zero-mass stage.** New stage 1 ($m_2 = 0$), shift old stages 1–5 to 2–6 ([Zero-mass stage insertion](#zero-mass-stage-insertion))
- [x] **4. Data** — five-stage `.npz` pools. Those files are the old ids. Do not rename them. Human regenerates after 3b.
- [x] **4b. Visualization** — apps, plots, gifs on data now; ANN/metrics wiring later ([Visualization](#visualization-step-4b))
- [x] **5. Baseline establishment** — lock width, depth, $k$, LR, batch; `runs/`
- [x] **6. Models** — shared MLP from model YAML
- [ ] **7. Strategies** — baseline, curriculum, active, progressive hooks
- [ ] **8. Checkpoint I/O + loader** — `best.pt` / `last.pt`, rebuild from config
- [ ] **9. Train loop** — CLI, `metrics.jsonl`, run folders
- [ ] **10. Eval** — frozen test, error-vs-$t$, forgetting, `results.csv` hooks
- [ ] **11. Run all 4** — launch all strategies with shared locks

Linear summary: simulator → config yaml → sampler → **3b zero-mass stage** → regenerate data → **viz (4b)** → baseline → models → strategies → checkpoint I/O → train → eval → run all 4; then **wire** loader + eval into existing viz.

## Visualization (step 4b)

Build-order step **4b** immediately after Step 4 (does not block Step 5). Delivers the full visualization surface area up front; later steps only **connect** data sources (checkpoints, `metrics.jsonl`, eval arrays), not new UIs.

### Shared layer

- `srcs/viz.py`: `build_svg`, canvas constants, shared CSS (no NiceGUI). `app.py` imports this; stays the live slider sandbox.
- `srcs/plots.py` (or equivalent): time-series helpers (sin/cos, $\omega$, PE, KE, sim-vs-ref difference), stage/split filters, static export + gif writer from frame sequences.
- **Trajectory source protocol:** small interface, e.g. `GroundTruthSource` (Step 4 reader) and `SurrogateSource` (stub/no-op until Step 8). Apps and plot CLIs accept a list of sources so ANN overlay is "add `SurrogateSource`" later.



### Phase A (required before Step 5; uses frozen `.npz` only)

- `app_data.py` (port `8766`): `data_root`, stage, split, traj index; play/pause/reset/trail; stored series vs re-integrate from IC; read-only IC meta ($g$, masses, energy drift).
- Dataset reader API: `list_pools`, `open_pool`, `get_traj(i) -> TrajectoryView` with `initial_state()`, `frame_at(k)`.
- Gifs / batch export from IC sets (CLI or app action) using ground-truth trajectories.
- Time-series plots from stored trajectories; filter by stage (and split).



### Phase B (interfaces in 4b; implementation wired in Step 8)

- `SurrogateSource`: given $(t, \mathrm{IC}, m, g)$, return 6-D predictions over $t$ (same decode as training). Loader from **Checkpoints** section calls the model; viz only depends on the protocol.
- App overlay: second pendulum trace or dual time-series (sim solid, ANN dashed) when a `run_id` / checkpoint is selected. UI control can be disabled until loader exists.



### Phase C (plot entrypoints in 4b; data wired in Step 10)

- Functions/CLIs that accept paths to `metrics.jsonl`, per-run error-vs-$t$ arrays, `summary.json`, and emit training curves + eval curves + optional `results.csv` figures. Step 10 writes the arrays; Step 10 does not own matplotlib/NiceGUI layout.



### Done when (4b)

- [x] `viz.py` extracted; `app.py` behavior unchanged
- [x] `app_data.py` animates trajectories from a real pool; gifs export works for at least one IC batch
- [x] Time-series plots (sin/cos, $\omega$, PE/KE) work on stored data with stage filter
- [x] `SurrogateSource` protocol + stub documented; overlay UI present but optional/disabled without checkpoint
- [x] Eval/training plot functions exist and run on tiny fixture files (empty or synthetic) without a trained net
- [x] Focused tests: SVG smoke, reader wiring, one static plot golden or shape check



### Later wiring (not new build-order steps)


| After step                 | Connect                                                                             |
| -------------------------- | ----------------------------------------------------------------------------------- |
| 8. Checkpoint I/O + loader | Implement `SurrogateSource` with rebuild + `load_state_dict`; enable overlay in app |
| 10. Eval                   | Point Phase C CLIs at real `metrics.jsonl` / error-vs-$t$ exports                   |
| 11. Run all 4              | Same plot CLIs over four `run_id`s for comparison figures                           |




## Zero-mass stage insertion

Assigned work for build-order step **3b** only. Stop when the checklist at the end passes. Do not start step 4, 4b, or training.

Read [Curriculum stages](#curriculum-stages), [Sampler generator](#sampler-generator), and this section. Do not rewrite `project.md`.

### What changed

Old stage $k$ becomes stage $k+1$ for $k = 1..5$. New stage 1 is the same zero-gravity draw as old stage 1, except $m_2 = 0$.


| New stage | $g$                   | Masses                       | IC box                              | Extra filter                            |
| --------- | --------------------- | ---------------------------- | ----------------------------------- | --------------------------------------- |
| 1         | `g_zero` (0)          | $m_1 =$ `m_equal`, $m_2 = 0$ | new `stage1` (copy of old `stage1`) | none                                    |
| 2         | `g_zero` (0)          | both `m_equal`               | old `stage1`                        | none                                    |
| 3         | `g_low`               | both `m_equal`               | old `stage2`                        | none                                    |
| 4         | `g_high`              | both `m_equal`               | old `stage3`                        | none                                    |
| 5         | Unif `[g_min, g_max]` | both `m_equal`               | old `stage4`                        | old `stage4.pe_min`                     |
| 6         | Unif `[g_min, g_max]` | free in old `stage5.m`       | old `stage5`                        | old `stage5.pe_min` and `mass_diff_min` |


Stages 1 and 2 share `g_zero`. They differ only in $m_2$. Do not give stage 1 a positive $g$. Do not set $m_1 = 0$.

Same model inputs as every other stage, including $m_2$. Stage 1 does not drop or rename heads. Output stays 6-D sin/cos + $\omega$. Loss stays sin/cos MSE + $\omega$ MSE.

### `configs/data/full.yaml`

1. Set `num_stages: 6`.
2. Rename `g_stage1` to `g_zero` and keep the value `0.0`. Delete `g_stage1`. Stage 1 and stage 2 both read `g_zero`.
3. Shift blocks: old `stage5` → `stage6`, old `stage4` → `stage5`, old `stage3` → `stage4`, old `stage2` → `stage3`, old `stage1` → `stage2`. Copy the old `stage1` theta/omega box into the new `stage1`. Do not copy `pe_min` or `m` onto stage 1.
4. Set pools to length 6:

```yaml
pools:
  train: [1024, 2048, 4096, 4096, 8192, 16384]
  test: [128, 256, 512, 512, 1024, 2048]
```

Index 0 is stage 1. Do not leave a length-5 list.

### Sampler code

Touch `srcs/sampler.py` only for stage dispatch and the stage-1 mass exception.

1. In `draw_candidate`, add `stage == 1` before the old stage-1 branch. That branch sets `g = g_zero`, `m1 = m_equal`, `m2 = 0`. Still draw both angles and both angular velocities from `stage1`.
2. Move the old `stage == 1` body to `stage == 2`. Old stage 2 becomes 3, old 3 becomes 4, old 4 becomes 5, old 5 becomes 6. Stage 2 reads `g_zero`, not `g_stage1`.
3. Replace leftover defaults `num_stages` of `5` that mean "how many curriculum stages" with `6`. Do not change physics tolerances that happen to be the number 5.
4. In `check_static_constraints`, apply `mass_below_min` and `m2_nonpositive` only when `stage != 1`. On stage 1, reject unless `m2 == 0` and `m1 == m_equal`. Keep the $m_1 \ge m_{\min}$ check on stage 1.
5. Move the old stage-4 PE reject to stage 5 (`stage5_pe_below_min`) and the old stage-5 PE / mass-gap rejects to stage 6 (`stage6_pe_below_min`, `stage6_mass_gap`).
6. Leave the EOM denominator check in place. For $m_2 = 0$ the denominator is $2 m_1$, which clears `den_min`.
7. Do not edit `derivatives`, `kinetic_energy`, or `potential_energy`. The $m_2 = 0$ formulas already drop the $m_2$ terms.



### Tests

Update `tests/test_sampler.py` and `tests/test_configs.py`. Shift every old stage id in those tests by +1. `tests/test_configs.py` must require `g_zero`, `stage6`, and `num_stages == 6`, and must not require `g_stage1`.

Add these checks. Do not delete the shifted ones.

- A stage-1 draw has `m2 == 0`, `m1 == m_equal`, `g == 0`.
- A stage-2 draw has `g == 0` and `m1 == m2 == m_equal`.
- `sample(1, n, cfg)` returns rows. `sample(0, ...)` and `sample(7, ...)` raise.
- Integrate one accepted stage-1 row with `integrate_rk4` for a short `t_end`. Unwrapped $\theta_i(t)$ matches $\theta_{i0} + \omega_{i0} t$ within a tight tolerance. $\omega_i(t)$ stays at $\omega_{i0}$. Energy stays finite. This is the parity check against `derivatives`. Do not re-derive a new ODE in the test.
- Stages 2–6 still enforce the old stage 1–5 mass, gravity, `pe_min`, and mass-gap rules. A bit-identical RNG stream against old `.npz` files is not required.

Comment-only updates, no behavior change, in:

- `srcs/data.py` (pool path comment says stages 1..5)
- `srcs/generate_data.py` (module docstring and `--train-counts` / `--test-counts` help say 1..5)
- `srcs/plots.py` (comment that stage ids run 1..5)

If a default `num_stages` of 5 in those modules is the curriculum count, set it to 6.

### Data on disk

Do not delete `data/`. Do not rename `stage{k}_*.npz` to `stage{k+1}_*.npz`. The array inside stores the old stage id. After 3b, those files are stale. Say that in the step report. Full regeneration is human-owned. A unit test may call `generate_stage` with `train_n <= 2` into `tmp_path` only.

### Done when

- [ ] `configs/data/full.yaml` matches the table and the pool lists above. No `g_stage1` key.
- [ ] `sample(stage, n, cfg)` accepts stages 1..6 and rejects $m_2 = 0$ on stages 2..6.
- [ ] Stage-1 RK4 matches $\theta_{i0} + \omega_{i0} t$ before wrapping.
- [ ] `python -m pytest tests/test_sampler.py tests/test_configs.py` passes.
- [ ] `physics.py` ODE and energy functions are unchanged.
- [ ] `data/` was not deleted or renamed.
- [ ] Report lists stale on-disk pools, if any, and does not start the next build-order step.



## Agent scope

Agents build **one step of the build order at a time**. Human (or a thin orchestrator prompt) assigns the step, pastes this section + the relevant domain sections, and stops the agent when that step's **Done when** checks pass. Do not start the next step until the previous step is accepted.

### Global rules (every agent)

- **Do:** code, tests, and configs needed for the assigned step only.
- **Don't:** Future directions; HNN / LNN / physics-structured models; rewriting this whole `project.md`; drive-by refactors of unrelated modules; inventing new experiments or strategies; changing locked baseline knobs after they are locked (width / final depth / LR / batch / $k$).
- **Read, do not rewrite unless assigned:** `project.md`, frozen test data under `data/`, existing `runs/` artifacts from other strategies.
- **Config ownership:** sampler bounds live in `configs/data/full.yaml`; architecture / train knobs live in `configs/models/<name>.yaml`. Do not hardcode stage numbers that already belong in YAML.
- **Loss / targets stay locked:** predict 6-D $(\sin\theta_1,\cos\theta_1,\sin\theta_2,\cos\theta_2,\omega_1,\omega_2)$; MSE on sin/cos + $\omega$; no plain $\theta$ MSE.
- **Stop condition:** implement the step, add focused tests or a smoke script, report what changed and what was left undone. Do not "continue" into the next build-order step.
- **Human-owned (agents may write the CLI, not burn agent tokens running full jobs):** large pool generation, multi-epoch training, GPU sweeps, packing `runs/` from long jobs. Agents may run tiny unit tests and tiny smoke sims.



### Step scopes


| Step                       | Agent may touch                                                                                                                  | Agent must not                                                                                            | Done when                                                                                                                                                                                                                |
| -------------------------- | -------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1. Simulator               | `srcs/` physics / integrate / energy helpers; physics tests                                                                      | Sampler, train, app redesign, YAML experiment knobs                                                       | RK4 traj + $E=T+V$ match PE/KE formulas here; energy nearly conserved on short smoke; angles wrappable to $(-\pi,\pi]$                                                                                                   |
| 2. Config YAML             | `configs/data/full.yaml`, `configs/models/*.yaml` keys used later                                                                  | Invent new strategies; Future directions keys                                                             | Keys cover stages, safety, IC boxes, loss weights, seeds; loadable without code edits for those values                                                                                                                   |
| 3. Sampler                 | Sampler module + tests; reads sampler YAML                                                                                       | MLP, train loop, rewriting frozen data layout mid-flight                                                  | Done for the old five stages. Do not extend this row. Use step 3b.                                                                                                                                                       |
| 3b. Zero-mass stage        | Files listed in [Zero-mass stage insertion](#zero-mass-stage-insertion)                                                          | `physics.py` ODE branches; loss; model YAML; viz apps; deleting `data/`; rewriting this `project.md`      | Checklist in that section, including `pytest tests/test_sampler.py tests/test_configs.py`                                                                                                                                |
| 4. Data                    | Dataset builder / `.npz` writer+reader; docs of array keys                                                                       | Changing loss; training strategies; deleting frozen test without explicit ask                             | Train/val/test pools for stages 1–6 on disk after 3b; test frozen; val carved from train; pointwise $(t,\mathrm{IC},m,g)\to$ 6-D target; full-res on disk ($k$ train-time only). Full pool generation stays human-owned. |
| 4b. Visualization          | `srcs/viz.py`, `plots.py`, `app.py` refactor, `app_data.py`, gif/time-series CLIs, `SurrogateSource` stub + overlay hooks; tests | Training the MLP; implementing checkpoint loader (Step 8); changing `.npz` schema without updating reader | **Visualization (step 4b)** Done when checklist                                                                                                                                                                          |
| 5. Baseline establishment  | Baseline model YAML + short search script; `runs/` for baseline trials                                                           | Curriculum / active / progressive logic; unlocking knobs after lock                                       | Document locked width, depth, $k$, LR, batch + winning `run_id`; pass bar = per-stage val $\le$ `stage_pass_mse`                                                                                                         |
| 6. Models                  | Shared MLP builder from model YAML (width / depth); unit shape tests                                                             | Strategy-specific data policies; eval plots                                                               | Rebuild from config; input/output dims match Setup; progressive can start at `depth_start`                                                                                                                               |
| 7. Strategies              | Baseline / curriculum / active / progressive **data or schedule hooks only** (one strategy per agent run preferred)              | Other strategies; HNN; changing locked width / LR / batch / $k$                                           | Strategy matches Experiments table; Active uses ensemble std + $\lambda(t)$; Progressive grows depth only                                                                                                                |
| 8. Checkpoint I/O + loader | `runs/<run_id>/` layout; save/load `state_dict` dicts; optional `runs/index.csv`; wire `SurrogateSource` in viz                  | Retrain logic; sampler changes; new viz apps                                                              | `best.pt` / `last.pt` policy; loader rebuilds from `config.yaml` then `load_state_dict`; `weights_only=True`; ANN overlay works in existing app when `run_id` set                                                        |
| 9. Train loop              | Train entrypoint; metrics.jsonl; early stopping / epoch budget floats                                                            | Redefining loss; regenerating frozen test; new plot/UI stack                                              | One CLI/path per model YAML; writes run folder; logs epochs, samples seen, val loss                                                                                                                                      |
| 10. Eval                   | Frozen-test metrics; error-vs-$t$; forgetting vs Stage-1 reference; `results.csv` hooks; export paths consumed by 4b plot CLIs   | Retraining; changing train data; new viz layout                                                           | Per-stage sin/cos+$\omega$ MSE; error-vs-$t$ curves saved or rebuildable; forgetting delta defined; Phase C plots render from those exports                                                                              |
| 11. Run all 4              | Launch scripts / docs to run strategies 1–4 with shared locks                                                                    | Inventing 5th strategy; floating locked knobs                                                             | Four run folders + comparable `summary.json` / `results.csv` rows; comparison figures via existing 4b plot CLIs                                                                                                          |




### Prompt contract (paste into each agent)

1. Assigned step number + name from the table above.
2. **In / must not / Done when** for that row (copy verbatim).
3. Pointers to the domain sections of this file that step needs (e.g. Sampler agent: "Sampler generator" + "Curriculum stages" only).
4. Explicit: "Do not implement other build-order steps. Stop when Done when passes."



### Out of scope for all coding agents until a human reopens them

- Everything under **Future directions**
- Equalizing sample budgets across strategies
- LR / batch sweeps after lock
- Combining strategies
- Variable pendulum count / mass count as input



## Future directions to keep open

- Combining multiple training strategies together
- More masses (n-pendulum); mass count as input
- Continuous curriculum (gravity ramp, PE, Lyapunov) [@vejendla2025chaos]
- **HNN + energy regularization** (and CHNN / LNN / ModLaNet): deferred — correct $(q,p)$, parametric $m,g$, and map-vs-integrate eval [@greydanus2019hnn; @eichelsdoerfer2021physics; @finzi2020chnn; @cranmer2020lnn; @lu2022modlanets]
- Sweep LR / batch (currently locked from baseline)
- Force equal sample budgets across strategies (currently floating)
- Progressive variants: grow width instead of / in addition to depth; different depth schedules

