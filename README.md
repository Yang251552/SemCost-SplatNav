# SemCost-SplatNav

**A lightweight semantic-cost splatting prototype for visual RL.**

When a navigation policy can only perceive *geometry* (depth / occupancy), it
cannot tell apart regions that look identical but differ in *semantic risk* —
e.g. solid floor vs. a patch of unsafe terrain that is just as flat. This
prototype asks a focused question:

> When geometry/depth cannot distinguish hazardous regions, does an extra
> **semantic-cost** observation channel help a PPO policy avoid them?

The project is structured as **two stages** of evidence on this single
hypothesis, each stage progressively more realistic:

- **Stage 1 — Controlled grid baseline (done).** A geometry-ambiguous 11×11
  arena with a *ground-truth* semantic-cost field. On 200 paired evaluation
  seeds: adding the semantic-cost channel cuts hazardous-region exposure by
  **~57%** at equal (100%) success, zero collisions for both, and shorter
  paths. Cheap, fully controlled, isolates the information effect — but the
  scene and semantics are toy.
- **Stage 2 — Flagship: gsplat scene + 3-arm ablation (in progress).** A real
  pretrained Gaussian-Splatting scene rendered into RGB-D on AWS, with an
  **estimated** semantic-cost field from post-hoc DINOv2 ViT-S features. Same
  PPO net / reward / seed / steps across **three obs modes** (`depth`, `rgb`,
  `rgb+semantic`); only `obs_keys` differ (enforced by config-hash + runtime
  fairness assertion + channel-ablation unit test). Headline comparison =
  **`rgb+semantic` vs `rgb`** (RGB already encodes some semantic cues, so this
  is the honest test; `depth` is reported for completeness). Plan, budget
  (≤\$50 spot), AWS confirmation gates, and overnight automation in
  [docs/AWS_FLAGSHIP_PLAN.md](docs/AWS_FLAGSHIP_PLAN.md).

This is a *prototype toward* online semantic Gaussian Splatting + RL
(e.g. GaussGym-style pipelines), **not** a full integration of any of them
even after Stage 2. Renderer-internal VFM feature splatting (the Hero
version) is explicitly [future work](#next-steps-hero-version). See also
[Limitations](#limitations).

---

## Stage 1 — Controlled grid baseline (done)

### Motivation

Photorealistic, geometry-only simulators give an agent excellent shape cues but
no notion of *what* a surface means. A semantic Gaussian-Splatting map, by
contrast, can render a per-view **semantic-cost** field (hazard, unsafe terrain,
non-traversable-but-passable regions). The thesis direction is to feed such a
rendered semantic cost into an RL policy. Before building the full pipeline, we
isolate the core hypothesis in a small, controllable environment.

### Method

- **Algorithm:** PPO (Stable-Baselines3) with a small shared CNN feature
  extractor (`SmallNavCNN`).
- **Two observation modes**, trained and evaluated under identical conditions:
  - `depth` — geometry channel only, shape `(1, 21, 21)`.
  - `depth_semantic` — geometry **+** semantic-cost channel, shape `(2, 21, 21)`.
- The **only** observation difference between the two runs is the presence of
  the semantic-cost channel. Reward, dynamics, map distribution, horizon, PPO
  hyperparameters, training seed and the evaluation seed list are identical
  (enforced by a shared config and a runtime fairness assertion). The semantic
  policy's first convolution consequently has one extra input channel (~144
  additional weights) — an unavoidable side effect of the extra channel, not a
  capacity advantage; all other layers are identical.

### Environment

A small `11x11` walled arena (Gymnasium API):

- Fixed goal at the top-centre; the start column is randomized each episode.
- A horizontal **hazard band** crosses the arena with a single **safe gap**
  whose column is randomized each episode.
- Crucially, the hazard band is **not a wall**: it does not block motion and
  leaves **no geometric trace**. The `depth` observation is therefore *provably
  identical* across different gap locations (verified in tests) — a depth-only
  agent cannot know where it is safe to cross. The gap is revealed **only** by
  the semantic-cost channel.
- Observation is egocentric and heading-aligned (the agent sits at the centre
  facing "up"); the geometry channel encodes walls and the goal, the
  semantic-cost channel encodes hazard intensity only (it never encodes the
  goal — checked by a leakage guard and a test).
- Actions: `forward`, `turn_left`, `turn_right`, `stop`.
- Reward (shared): per-step `-0.01`, collision `-0.1`, hazard `-0.1 * intensity`
  per step on a hazard cell, goal `+1.0`, plus small potential-based distance
  shaping (`0.02`).

![Scene, depth channel, semantic-cost channel, and trajectories](figures/scene_depth_cost_trajectory.png)

*(a) the scene: a hazard band with a safe gap; (b) the geometry/depth channel —
the hazard is **invisible**; (c) the semantic-cost channel — it reveals the
hazard and the gap; (d) example trajectories: the depth-only policy crosses the
hazard, the semantic policy detours through the gap.*

### Experiments

- PPO, 500k timesteps per mode, 8 parallel envs, training seed `0`.
- Evaluation on **200 fixed seeds**, identical for both modes (so both policies
  face byte-identical maps, start columns and gaps), deterministic actions.
- Primary metrics: `success_rate`, `collision_rate`, `bad_region_time`,
  `path_length`, `average_return`.

### Results

| Metric | depth-only | depth + semantic_cost | Δ |
|---|---|---|---|
| success_rate | 1.000 | 1.000 | = |
| collision_rate | 0.000 | 0.000 | = |
| **bad_region_time** (steps in hazard) | **1.92** | **0.83** | **−57%** |
| path_length | 12.94 | 10.76 | shorter |
| average_return | 0.858 | 0.996 | higher |

*(200 paired evaluation maps; per-episode `bad_region_time` std ≈ 1.4, so the
~1.1-step paired gap is large relative to per-episode spread. Single training
seed per mode — see [Limitations](#limitations) — so this is a paired-evaluation
result, not a claim about training-seed variance.)*

![Metric comparison](figures/metrics_comparison.png)

Side-by-side rollout on the same map (depth-only left, depth+semantic right):

![Rollout comparison](figures/rollout_comparison.gif)

**Reading of the result.** Both policies solve the task (100% success, zero
collisions). With geometry only, the agent cannot see the hazard and crosses it;
adding the semantic-cost channel lets the agent route through the safe gap,
roughly **halving hazardous-region exposure** — and, because the gap-aware route
is also more direct here, it does so with *shorter* paths and higher return. The
headline claim rests on `bad_region_time` (a safety metric), not on reward alone.

---

## Stage 2 — Flagship: gsplat scene + 3-arm ablation (in progress)

> **Status:** code path and full plan are in place; training runs are gated
> on the §9 confirmation sign-off in
> [docs/AWS_FLAGSHIP_PLAN.md](docs/AWS_FLAGSHIP_PLAN.md). Result tables and
> hero video are filled in after the overnight run completes; placeholders
> below are marked `TBD`.

### Method

- **Scene.** A pretrained Gaussian-Splatting scene from HuggingFace, chosen
  so the hazard class (e.g. rug / spill / clutter patch) is **geometrically
  indistinguishable** from the surrounding floor in the depth channel but
  visually distinct in DINOv2 features (verified in D0 before any paid GPU;
  parity IoU ≥ 0.5 against a human-labelled hazard mask).
- **Renderer.** `gsplat` (CUDA) on an AWS A10G, producing per-frame RGB + Depth
  along robot poses.
- **Estimated semantic-cost.** For each rendered RGB frame, run **DINOv2 ViT-S**
  → PCA → cosine vs a hazard prototype embedding → per-pixel cost map. The
  cost map encodes hazard intensity only; it never encodes target position
  (leakage guard + channel-ablation unit test, carried over from Stage 1).
- **Three obs modes** (same PPO net / reward / seed / steps; only `obs_keys`
  differ, enforced by a config-hash check at run start):
  - `A. depth`         → `[depth]`
  - `B. rgb`           → `[rgb]`
  - `C. rgb+semantic`  → `[rgb, cost_map]`
- **Env.** A GaussGym-style vectorized env on IsaacGym / Isaac Lab (exact
  backend locked in D0); 512 → 1024 parallel envs depending on VRAM.

### Headline comparison

`rgb+semantic` vs `rgb`. RGB already encodes some semantic cues, so this is
the honest test of whether an explicit semantic-cost channel adds *further*
safety value once the policy can see colour. All three arms are reported, but
the headline claim rests on this single pair (and on `bad_region_time`, not
on average reward).

### Results (filled in post-run)

| Metric                | depth-only | rgb     | rgb + semantic | Δ (sem − rgb) |
|-----------------------|------------|---------|----------------|---------------|
| success_rate          | TBD        | TBD     | TBD            | TBD           |
| collision_rate        | TBD        | TBD     | TBD            | TBD           |
| **bad_region_time**   | TBD        | TBD     | TBD            | TBD           |
| path_length           | TBD        | TBD     | TBD            | TBD           |
| average_return        | TBD        | TBD     | TBD            | TBD           |

*(Paired evaluation across a fixed eval-seed set; target ≥3 training seeds per
arm, honestly disclosed if the AWS budget forces fewer. Hero clip
`figures/ablation_rollout.mp4` shows the same scene rolled out under each
arm side-by-side; learning curves in `figures/time_in_bad_region.png`; the
RGB / depth / cost / avoid-mask 4-up panel in
`figures/rgbd_cost_avoid_quad.png`.)*

See [Reproduce → Stage 2](#stage-2-aws-g52xlarge-spot-a10g) for the exact
commands and [docs/AWS_FLAGSHIP_PLAN.md](docs/AWS_FLAGSHIP_PLAN.md) for the
full plan, budget cap, AWS confirmation gates, and overnight automation
pipeline.

---

## Limitations

**Stage 1 (grid baseline):**
- The task is intentionally simple and geometry-ambiguous; it is designed to
  *isolate* the semantic-information effect, not to be a general benchmark.
- The "depth" channel is a top-down egocentric occupancy/geometry view, a
  simplification of true sensor depth, chosen to keep the task Markovian and
  trainable on a single CPU.
- The semantic-cost channel is a ground-truth hazard field — a **stand-in**
  for the semantic map a real system would have to *estimate*. Stage 2 lifts
  this restriction.
- Single training seed per mode, with a fixed shared evaluation seed set
  (paired comparison; not a claim about training-seed variance).

**Stage 2 (flagship):**
- Single pretrained gsplat scene with a single hazard object class; not a
  multi-scene generalization study.
- Semantic-cost comes from post-hoc DINOv2 + PCA + cosine to a hazard
  prototype — it is now an *estimated* field, but estimated **offline** from
  rendered RGB, not jointly with the splat. Renderer-internal VFM feature
  splatting (Hero version) is future work.
- 3-arm runs are budget-bounded (≤\$50 AWS spot on a single A10G; the three
  arms share that one GPU, so training is serialized — not "3× faster via
  parallel arms"). Target ≥3 training seeds per arm, honestly disclosed if
  the budget forces fewer.
- GaussGym integration is a lightweight env wrapper around the gsplat
  renderer, not a full reproduction of GaussGym's pipelines.

## Next steps (Hero version)

Stage 2 already absorbs items 1 and 2 of the original "toward GaussGym"
roadmap into the project. What remains is the **Hero version**:

1. **Renderer-internal VFM feature splatting.** Instead of running DINOv2
   post-hoc on rendered RGB, attach VFM features to the Gaussians themselves
   so semantic features can be rendered jointly with RGB-D in one rasterizer
   pass — the architecture an online semantic-splatting + RL system would
   actually use.
2. **Multi-scene generalization and on-policy splat updates.** Train across
   several gsplat scenes; study whether the policy's behaviour is sensitive
   to splat quality, hazard-prototype choice, and on-policy map refinement.
3. **Robustness to semantic-map estimation error**, partial observability,
   and distribution shift between the prototype set and deployment scenes.

The repository is structured so the env, policy, and experiment harness can
be reused as these pieces are swapped in.

## Reproduce

### Stage 1 (CPU, single machine)

```bash
pip install -r requirements.txt

# Train + evaluate both modes with a single fairness-checked command:
python scripts/run_experiment.py

# Or run the steps individually:
python scripts/train.py --obs depth
python scripts/train.py --obs depth_semantic
python scripts/evaluate.py --obs depth
python scripts/evaluate.py --obs depth_semantic

# Figures and rollout:
python scripts/plot_results.py
python scripts/render_rollout.py

# Tests:
python tests/test_env_smoke.py
```

Stage 1 outputs: `results/eval_depth.json`, `results/eval_depth_semantic.json`,
`figures/metrics_comparison.png`, `figures/rollout_comparison.gif` (+ `.mp4`),
`figures/scene_depth_cost_trajectory.png`.

### Stage 2 (AWS g5.2xlarge spot, A10G)

See [AWS_RUNBOOK.md](AWS_RUNBOOK.md) for instance setup and the
[§9 confirmation gates](docs/AWS_FLAGSHIP_PLAN.md) before any paid step.

```bash
# Pinned env (CUDA / torch / gsplat / IsaacGym-or-IsaacLab, locked in D0):
conda env create -f env.yaml && conda activate semcost-flagship

# 3-arm training (same config, only observation tokens differ; config-hash enforced):
python scripts/train_gaussgym.py --obs depth   --config configs/flagship.yaml
python scripts/train_gaussgym.py --obs rgb     --config configs/flagship.yaml
python scripts/train_gaussgym.py --obs rgb_sem --config configs/flagship.yaml

# Paired evaluation (2 seeds: spot quota denied -> on-demand cost cap; disclosed in REPORT):
python scripts/evaluate_gaussgym.py --all-modes --seeds 2

# Or the one-shot overnight pipeline (train → eval → S3 sync → shutdown):
bash scripts/run_flagship_overnight.sh
```

Stage 2 outputs: `results/eval_rgb.json`, `results/eval_rgb_sem.json`
(+ `results/eval_gaussgym_depth.json` for completeness — Stage-2 depth arm,
named so it never overwrites the Stage-1 `results/eval_depth.json`),
`results/parity_report.json`, `figures/ablation_rollout.mp4`,
`figures/time_in_bad_region.png`, `figures/rgbd_cost_avoid_quad.png`,
`renders/cost_map_overlay.mp4`.

## Repository layout

```
semcost_nav/             # Stage 1 package: env, maps, policy extractor, experiment config, metrics
scripts/                 # Stage 1: train.py, evaluate.py, run_experiment.py, plot_results.py, render_rollout.py
                         # Stage 2: train_gaussgym.py, evaluate_gaussgym.py, run_flagship_overnight.sh
configs/                 # Stage 2: flagship.yaml (shared across all 3 arms)
tests/                   # env smoke tests + Stage 2 config-hash & channel-ablation tests
results/                 # eval + training-metadata JSON (Stage 1 + Stage 2)
figures/                 # generated figures, GIFs, MP4s (Stage 1 + Stage 2 hero artifacts)
renders/                 # Stage 2: gsplat RGB-D + DINO cost-map overlay clips
docs/                    # AWS_FLAGSHIP_PLAN.md (Stage 2 plan, budget, gates)
local_safety_snapshots/  # patch snapshots before non-trivial phases (gitignored)
```

`.gitignore` covers training artifacts (`runs/`, `checkpoints/`, `*.ckpt`,
`*.pt`, large `renders/*.mp4`) — Stage 2 checkpoints live in S3, not git;
only small metrics / configs / parity reports are committed.
