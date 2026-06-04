# SemCost-SplatNav

**A lightweight semantic-cost splatting prototype for visual RL.**

When a navigation policy can only perceive *geometry* (depth / occupancy), it
cannot tell apart regions that look identical but differ in *semantic risk* —
e.g. solid floor vs. a patch of unsafe terrain that is just as flat. This
prototype asks a focused question:

> When geometry/depth cannot distinguish hazardous regions, does an extra
> **semantic-cost** observation channel help a PPO policy avoid them?

Short answer, on a deliberately geometry-ambiguous task: **yes** — adding a
semantic-cost channel cuts hazardous-region exposure by ~57% at equal (100%)
success, with collisions at zero for both, while also producing shorter paths.

This is a *prototype toward* online semantic Gaussian Splatting + RL
(e.g. GaussGym-style pipelines), **not** a full integration. See
[Limitations](#limitations) and [Next steps](#next-steps-toward-gaussgym).

---

## Motivation

Photorealistic, geometry-only simulators give an agent excellent shape cues but
no notion of *what* a surface means. A semantic Gaussian-Splatting map, by
contrast, can render a per-view **semantic-cost** field (hazard, unsafe terrain,
non-traversable-but-passable regions). The thesis direction is to feed such a
rendered semantic cost into an RL policy. Before building the full pipeline, we
isolate the core hypothesis in a small, controllable environment.

## Method

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

## Environment

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

## Experiments

- PPO, 500k timesteps per mode, 8 parallel envs, training seed `0`.
- Evaluation on **200 fixed seeds**, identical for both modes (so both policies
  face byte-identical maps, start columns and gaps), deterministic actions.
- Primary metrics: `success_rate`, `collision_rate`, `bad_region_time`,
  `path_length`, `average_return`.

## Results

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

## Limitations

- The task is intentionally simple and geometry-ambiguous; it is designed to
  *isolate* the semantic-information effect, not to be a general benchmark.
- The "depth" channel is a top-down egocentric occupancy/geometry view, a
  simplification of true sensor depth, chosen to keep the task Markovian and
  trainable on a single CPU.
- The semantic-cost channel is a ground-truth hazard field provided by the
  environment — it is a **stand-in** for the semantic map a real system would
  have to *estimate*. Estimation error is out of scope here.
- Single training seed per mode (with a fixed, shared evaluation seed set).
  Results are reported on CPU with recorded dependency versions for
  reproducibility.

## Next steps (toward GaussGym)

This prototype is the RL/observation-interface half of a larger plan:

1. Replace the hand-crafted hazard field with a **semantic-cost map rendered
   from a (semantic) Gaussian-Splatting scene**, so the semantic channel comes
   from a learned/estimated map rather than ground truth.
2. Move from the abstract grid to a **GaussGym-style photorealistic** setting
   and feed rendered RGB-D + semantic-cost to the policy.
3. Study robustness to **semantic-map estimation error** and partial
   observability.

The repository is structured so the environment, policy and experiment harness
can be reused as these pieces are swapped in.

## Reproduce

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

Outputs: `results/eval_depth.json`, `results/eval_depth_semantic.json`,
`figures/metrics_comparison.png`, `figures/rollout_comparison.gif` (+ `.mp4`),
`figures/scene_depth_cost_trajectory.png`.

## Repository layout

```
semcost_nav/        # package: env, maps, policy extractor, experiment config, metrics
scripts/            # train.py, evaluate.py, run_experiment.py, plot_results.py, render_rollout.py
tests/              # environment smoke tests (shapes, determinism, leakage guard, orientation)
results/            # evaluation + training-metadata JSON
figures/            # generated figures, GIF, MP4
```
