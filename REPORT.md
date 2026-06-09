# SemCost-SplatNav — Report

**A lightweight semantic-cost splatting prototype for visual RL.**

## TL;DR

One hypothesis, tested with increasing realism:

> When geometry/depth cannot distinguish hazardous regions, can a **semantic-cost**
> observation help a policy avoid them?

- **Stage 1 (controlled grid, ground-truth semantics).** On a geometry-ambiguous
  arena, adding a semantic-cost channel cuts hazardous-region exposure by
  **−57%** (`bad_region_time` 1.92 → 0.83) at equal 100% success and zero
  collisions, on 200 paired evaluation seeds. *Clean, controlled, but toy.*
- **Stage 2 / P2 (real gsplat scene, estimated semantics).** On a real pretrained
  Gaussian-Splatting living-room scene, a post-hoc **DINOv2** cost map localizes a
  carpet hazard (parity **IoU 0.674**, leave-one-out **0.677**) while the hazard is
  **geometrically invisible in depth at this render resolution** — its signed step
  sits below the floor-fit noise floor (`depth_invariant`), not literally zero
  thickness. *The semantic signal is real and estimable from a realistic scene.*
- **Stage 2 / P3 (3-arm RL training).** Deliberately scoped as **future work**, not
  run. See [Why P3 is future work](#why-p3-rl-training-is-future-work-not-a-gap) —
  the honest reason is methodological, not a missing capability.

The project is a *prototype toward* online semantic Gaussian-Splatting + RL, not a
full integration of any such system.

---

## Hypothesis (unchanged across stages)

When geometry/depth observations cannot distinguish hazardous regions,
semantic-cost observations can help a policy reduce time in bad regions,
collisions, or path inefficiency. Stage 1 tests this with **ground-truth**
semantics on a grid; Stage 2 tests whether the semantic signal is **estimable**
from a realistic scene with **post-hoc DINOv2** features.

---

## Stage 1 — Controlled grid baseline (done)

**Setup.** PPO (SB3) with a small shared CNN on an 11×11 walled arena. A
horizontal hazard band crosses the arena with a single safe gap whose column is
randomized per episode. The band blocks no motion and **leaves no geometric
trace** — the `depth` observation is *provably identical* across gap locations
(verified in tests), so a depth-only agent cannot know where to cross. Two obs
modes, identical in everything but the extra channel: `depth` (1×21×21) vs
`depth_semantic` (2×21×21). Reward, dynamics, map distribution, horizon, PPO
hyperparameters, training seed, and the 200-seed evaluation list are shared
(runtime fairness assertion).

**Results (200 paired evaluation seeds, deterministic actions).**

| Metric | depth-only | depth + semantic_cost | Δ |
|---|---|---|---|
| success_rate | 1.000 | 1.000 | = |
| collision_rate | 0.000 | 0.000 | = |
| **bad_region_time** | **1.92** | **0.83** | **−57%** |
| path_length | 12.94 | 10.76 | shorter |
| average_return | 0.858 | 0.996 | higher |

**Reading.** Both policies solve the task; with geometry only, the agent crosses
the invisible hazard, while the semantic channel lets it route through the safe
gap — roughly halving hazardous-region exposure. The headline rests on
`bad_region_time` (a safety metric), not reward alone. *Caveat: single training
seed per mode — a paired-evaluation result, not a training-variance claim.*

---

## Stage 2 / P2 — Semantic-cost validation on a real gsplat scene (done)

**Goal.** Before any paid RL, validate the *premise* on a realistic scene: is the
hazard (a) **estimable** from RGB via a vision foundation model, and (b) **not**
already visible in geometry/depth?

**Setup.** A pretrained gsplat scene (HuggingFace `escontra/gauss_gym_arkit`,
`training/43895956` — a living room with a patterned carpet on a wood floor),
rendered to 80 RGB-D frames (free Colab T4; $0 AWS). Carpet hand-labelled in 5
frames. Estimated semantic cost: per RGB frame, **DINOv2 ViT-S** patch features →
L2-norm → PCA(64, frozen) → cosine vs a few-shot carpet prototype → soft
per-pixel cost. All run locally on CPU.

**Result — two gates.**

| Gate | Metric | Value | Verdict |
|---|---|---|---|
| Semantics **see** the hazard | parity IoU (fixed τ=0.1) | **0.674** (per-frame 0.60–0.73) | ≥ 0.5 ✓ |
| | leave-one-out IoU (proto from N−1 frames) | **0.677** | not overfitting ✓ |
| Depth does **not** | signed carpet step vs local floor-fit noise | step/noise = 0.27 (std) / 0.72 (MAD), **sign inconsistent** across frames | `depth_invariant` ✓ |

![RGB / depth / DINO cost / avoid-mask](figures/rgbd_cost_avoid_quad.png)

**Reading & honest strength.** The DINO cost localizes the carpet, and the
localization generalizes across held-out frames. Depth shows **no systematic,
sign-consistent geometric step above the rendering noise floor** — a real step
cannot flip sign between adjacent viewpoints, and here it does (+,−,−,−,−). This
is the honest, *weaker* claim that the geometric signal is below noise — **not**
that the carpet has exactly zero thickness (unlike the Stage-1 grid, where depth
is provably identical). The parity is a solid-but-moderate 0.674, with some
false positives in the cost map.

**Review provenance.** Two independent read-only review rounds. The first caught a
genuine bug: the depth coplanarity test originally fit the floor plane on points
that *included the carpet* (self-fitting). Fixed to a floor-only fit; the verdict
then required switching from an `|residual|` threshold (noise-dominated) to a
**signed step vs noise + cross-frame sign-consistency** criterion, which the
second round confirmed is the scientifically correct test (not p-hacking) and
robust to the std-vs-MAD choice of noise scale. The estimated cost never encodes
target position (leakage guard + unit test); `pytest` 16 passed.

---

## Why P3 (RL training) is future work, not a gap

P3 would train three RL arms (`depth` / `rgb` / `rgb+semantic`) on this scene and
report whether the explicit semantic channel improves safety. We deliberately
scope it as **future work**, for a methodological reason that P2 itself surfaced:

**The DINO cost is computed *from RGB*.** P2 proving "the carpet is estimable from
RGB" is equivalent to proving "the carpet signal is already present in the RGB
image." Since the carpet is also visually salient (strong colour/texture contrast
on wood floor), a **plain RGB policy may already have enough signal to avoid it** — so
the honest headline comparison (`rgb+semantic` vs `rgb`) **may show only a small
increment**. That makes P3 *as-is* **less diagnostic**: it risks an inconclusive
near-tie rather than a clean test of whether an explicit semantic channel helps.
*(This is a hypothesis about the untrained RGB baseline, not a measured result —
which is exactly why running P3 unchanged would be low-information.)*

The scientifically honest move is therefore **not** to force a flagship RL number,
but to:

1. Report Stage 1 (a *clean* positive with ground-truth semantics) + Stage 2/P2
   (semantics are *estimable and geometry-invisible* on a real scene) as the
   evidence gradient, and
2. State precisely what would make P3 worthwhile: a hazard that is **visually
   subtle** (RGB-hard but DINO-separable) — e.g. a same-colour material/traction
   transition or low-contrast slippery patch — so an explicit semantic channel can
   add value a plain RGB policy cannot easily learn. This requires re-selecting the
   hazard/scene and re-running P2 validation first.

This is consistent with the project rule to *adjust task design honestly when
results are weak, rather than manufacture a number*.

---

## Limitations

**Stage 1.** Intentionally simple, geometry-ambiguous task to *isolate* the
information effect; top-down occupancy "depth" (not true sensor depth);
ground-truth semantics (a stand-in); single training seed per mode (paired eval).

**Stage 2 / P2.** Single scene, single hazard class; semantics estimated
**offline** from rendered RGB (post-hoc DINOv2), not jointly with the splat;
parity is moderate (IoU 0.674) over **5 labelled frames** (small-sample
cross-frame consistency, not strong generalization); `depth_invariant` is "below
the rendering noise floor," not zero-thickness coplanarity; 84×48 render
resolution has substantial depth noise (floor-fit residual std 6–14 cm).

**Not claimed.** No full GaussGym/IsaacGym integration; no renderer-internal VFM
feature splatting (the "Hero version"); no RL gain from estimated semantics
(that is P3 / future work).

---

## Deliverables

**Stage 1:** `README.md`, `PROJECT_SPEC.md`, `results/eval_depth.json`,
`results/eval_depth_semantic.json`, `figures/metrics_comparison.png`,
`figures/rollout_comparison.{gif,mp4}`, `figures/scene_depth_cost_trajectory.png`.

**Stage 2 / P2:** `results/parity_report.json` (IoU 0.674),
`results/parity_loo.json` (LOO 0.677), `results/depth_invariance.json`
(`depth_invariant`), `figures/rgbd_cost_avoid_quad.png`, 5 human carpet masks
(`assets/p2_arkit_render/masks/*.npy`), pipeline scripts
(`scripts/{fit_dino_pca,build_bad_proto,build_semantic_cost,eval_parity,loo_parity,depth_invariance_check,make_quad_figure,label_carpet_mask}.py`),
`semcost_nav/semantic/dino_cost.py`, leakage/fairness unit tests. Regenerable
binaries (`results/dino_pca64*.npy`, `bad_proto.npy`, `semantic_cost_maps.npz`)
are gitignored.

**Stage 2 / P3 (future work, not produced):** 3-arm RL training/eval, hero
ablation video, learning curves — gated and scoped above.

---

## Reproduce

Stage 1 and the P2 semantic-cost pipeline both run on CPU; see the
[Reproduce](README.md#reproduce) and
[Semantic-cost validation](README.md#semantic-cost-validation-p2--done) sections
of the README for exact commands. The full P3 AWS plan, budget cap, and
confirmation gates remain documented in
[docs/AWS_FLAGSHIP_PLAN.md](docs/AWS_FLAGSHIP_PLAN.md) for when/if P3 is pursued.
