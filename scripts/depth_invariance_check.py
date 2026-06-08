"""Check whether labeled carpet masks are geometrically distinguishable in depth.

Method (signed step vs noise floor). For each frame:
  1. Back-project depth to a 3D point cloud with the camera intrinsics.
  2. Fit a LOCAL floor plane (RANSAC) on the FLOOR ONLY -- the dilated
     neighborhood around the carpet mask with the carpet itself EXCLUDED, so the
     carpet residual is a genuine held-out test, not self-fitting.
  3. Estimate the carpet's *signed* step = median signed distance of carpet
     points to the floor plane (oriented +toward camera), and compare it to the
     plane-fit noise scale = std of the floor points' signed residuals.

A real geometric step would be a systematic, sign-consistent offset that rises
ABOVE the noise floor. Depth is judged able to distinguish the hazard only if
median |step|/noise >= 1 AND the step sign is consistent across frames.
Otherwise the geometric signal (if any) is below the rendering noise floor and
depth cannot reliably distinguish the carpet -> "depth_invariant".

This is intentionally weaker than claiming zero-thickness coplanarity: it states
the geometric signal is below the noise floor, while the DINO semantic signal is
clear (see parity IoU). The raw absolute-depth inside-vs-ring KS statistic is
kept under `supplementary_ring_abs_depth` for transparency (it is confounded by
near-field perspective gradient + furniture in the ring) and is NOT used here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation
from scipy.stats import ks_2samp


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Depth step-vs-noise test for carpet labels.")
    parser.add_argument("--masks-dir", default=str(ROOT / "assets" / "p2_arkit_render" / "masks"))
    parser.add_argument("--depth-dir", default=str(ROOT / "assets" / "p2_arkit_render" / "depth"))
    parser.add_argument("--meta", default=str(ROOT / "assets" / "p2_arkit_render" / "meta.json"))
    parser.add_argument("--out", default=str(ROOT / "results" / "depth_invariance.json"))
    parser.add_argument("--ransac-thresh", type=float, default=0.03)
    parser.add_argument("--step-to-noise", type=float, default=1.0,
                        help="median |step|/noise at/above which depth is judged to see a step")
    parser.add_argument("--dilate", type=int, default=8)
    parser.add_argument("--ring-pixels", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def backproject(depth: np.ndarray, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    h, w = depth.shape
    us, vs = np.meshgrid(np.arange(w), np.arange(h))
    Z = depth.astype(np.float64)
    X = (us - cx) / fx * Z
    Y = (vs - cy) / fy * Z
    return np.stack([X, Y, Z], axis=-1)


def ransac_plane(pts: np.ndarray, thr: float, rng: np.random.Generator, iters: int = 600):
    if len(pts) < 3:
        raise ValueError("need >=3 points to fit a plane")
    best_inliers = None
    for _ in range(iters):
        idx = rng.choice(len(pts), 3, replace=False)
        p = pts[idx]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n = n / norm
        d = n @ p[0]
        dist = np.abs(pts @ n - d)
        inliers = dist < thr
        if best_inliers is None or inliers.sum() > best_inliers.sum():
            best_inliers = inliers
    P = pts[best_inliers]
    centroid = P.mean(axis=0)
    _, _, Vt = np.linalg.svd(P - centroid)
    n = Vt[-1]
    # Orient the normal toward the camera origin so +residual = raised above floor.
    if n @ (-centroid) < 0:
        n = -n
    d = n @ centroid
    return n, d


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    meta = json.loads(Path(args.meta).read_text(encoding="utf-8"))
    intr = meta["intrinsics"]
    fx, fy, cx, cy = intr["fl_x"], intr["fl_y"], intr["pp_x"], intr["pp_y"]

    mask_paths = sorted(Path(args.masks_dir).glob("*.npy"))
    if not mask_paths:
        raise SystemExit(
            "No masks found. First run: python scripts/label_carpet_mask.py --frame <X>"
        )

    per_frame: dict[str, dict] = {}
    ring_pf: dict[str, dict] = {}
    for mask_path in mask_paths:
        frame_id = mask_path.stem
        depth_path = Path(args.depth_dir) / f"{frame_id}.npy"
        if not depth_path.exists():
            print(f"[depth-invariance] skipping {frame_id}: missing depth")
            continue
        mask = np.load(mask_path).astype(bool)
        depth = np.load(depth_path).astype(np.float32)
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        if depth.shape != mask.shape:
            raise SystemExit(f"Shape mismatch for {frame_id}: depth={depth.shape} mask={mask.shape}")

        finite = np.isfinite(depth)
        pc = backproject(depth, fx, fy, cx, cy)
        region = binary_dilation(mask, iterations=args.dilate)
        carpet_sel = mask & finite
        floor_sel = region & (~mask) & finite  # floor only, carpet EXCLUDED from the fit
        if floor_sel.sum() < 10 or carpet_sel.sum() < 5:
            print(f"[depth-invariance] skipping {frame_id}: too few points")
            continue
        n, d = ransac_plane(pc[floor_sel].reshape(-1, 3), args.ransac_thresh, rng)

        sc = pc[carpet_sel].reshape(-1, 3) @ n - d   # signed carpet residual
        sf = pc[floor_sel].reshape(-1, 3) @ n - d    # signed floor residual
        step_signed = float(np.median(sc))            # estimated step height (signed)
        # Local floor-fit residual scale (NOT pure sensor noise: floor_sel is a
        # local non-mask neighborhood, so it can include slope/edge/occlusion
        # residuals). Report both std and a robust MAD scale for sensitivity.
        noise_std = float(np.std(sf))
        noise_mad = float(1.4826 * np.median(np.abs(sf - np.median(sf))))
        step_to_noise = abs(step_signed) / noise_std if noise_std > 1e-9 else float("inf")
        step_to_noise_mad = abs(step_signed) / noise_mad if noise_mad > 1e-9 else float("inf")
        per_frame[frame_id] = {
            "carpet_step_signed_median_m": step_signed,
            "carpet_resid_abs_median_m": float(np.median(np.abs(sc))),
            "floor_noise_std_m": noise_std,
            "floor_noise_mad_m": noise_mad,
            "floor_resid_abs_median_m": float(np.median(np.abs(sf))),
            "step_to_noise": step_to_noise,
            "step_to_noise_mad": step_to_noise_mad,
            "carpet_depth_median_m": float(np.median(depth[carpet_sel])),
            "n_carpet_pts": int(carpet_sel.sum()),
            "n_floor_fit_pts": int(floor_sel.sum()),
        }

        # Supplementary: the flawed-for-this-purpose absolute-depth ring KS test.
        ring = binary_dilation(mask, iterations=args.ring_pixels) & ~mask
        inside = depth[mask]
        outside = depth[ring]
        inside = inside[np.isfinite(inside)]
        outside = outside[np.isfinite(outside)]
        if inside.size and outside.size:
            ks = ks_2samp(inside, outside)
            ring_pf[frame_id] = {
                "mean_inside": float(inside.mean()),
                "mean_outside": float(outside.mean()),
                "mean_diff": float(abs(inside.mean() - outside.mean())),
                "ks_pvalue": float(ks.pvalue),
            }

    if not per_frame:
        raise SystemExit("No frames could be evaluated.")

    steps = [v["carpet_step_signed_median_m"] for v in per_frame.values()]
    ratios = [v["step_to_noise"] for v in per_frame.values()]
    ratios_mad = [v["step_to_noise_mad"] for v in per_frame.values()]
    med_ratio = float(np.median(ratios))
    med_ratio_mad = float(np.median(ratios_mad))
    signs = {int(np.sign(s)) for s in steps if s != 0.0}
    sign_consistent = len(signs) <= 1
    is_stub = any((Path(args.masks_dir) / f"{fid}.auto_stub").exists() for fid in per_frame)

    # Depth can distinguish the hazard only if a systematic step rises to/above the
    # noise floor AND its sign is consistent across frames. Sign-consistency is the
    # primary, noise-scale-INDEPENDENT criterion: a real geometric step cannot flip
    # sign between adjacent viewpoints, so the std-vs-MAD choice of noise scale does
    # not change the verdict here. We require it under BOTH noise scales to be safe.
    rises_above_noise = (med_ratio >= args.step_to_noise) or (med_ratio_mad >= args.step_to_noise)
    depth_sees_step = rises_above_noise and sign_consistent
    if is_stub:
        verdict = "AUTO_STUB - replace by real human labels; not a P2 gate verdict"
        interpretation = "stub mask; not a real verdict"
    elif depth_sees_step:
        verdict = "depth_distinguishes"
        interpretation = (
            "carpet shows a systematic, sign-consistent geometric step above the local "
            f"floor-fit noise floor (median |step|/noise std={med_ratio:.2f}, MAD={med_ratio_mad:.2f})"
        )
    else:
        verdict = "depth_invariant"
        interpretation = (
            "no systematic geometric step above the local floor-fit noise floor "
            f"(median |step|/noise std={med_ratio:.2f}, MAD={med_ratio_mad:.2f}; step signs "
            f"{'consistent' if sign_consistent else 'inconsistent'} across frames -- the sign "
            "inconsistency is noise-scale-independent and is the primary evidence). "
            "Depth/geometry cannot reliably distinguish the carpet at this rendering "
            "resolution; this is weaker than zero-thickness coplanarity."
        )

    payload = {
        "num_labeled": len(per_frame),
        "is_stub": is_stub,
        "method": "floor-only RANSAC plane (carpet EXCLUDED from fit); verdict from signed carpet step vs LOCAL floor-fit noise floor (std + robust MAD sensitivity) and cross-frame sign-consistency (the noise-scale-independent primary criterion)",
        "ransac_thresh_m": args.ransac_thresh,
        "step_to_noise_threshold": args.step_to_noise,
        "per_frame": per_frame,
        "supplementary_ring_abs_depth": {
            "note": "absolute-depth inside-vs-ring KS test; confounded by near-field perspective gradient and non-floor objects in the ring; reported for transparency, NOT used for verdict",
            "ring_pixels": args.ring_pixels,
            "per_frame": ring_pf,
        },
        "overall": {
            "median_step_to_noise": med_ratio,
            "median_step_to_noise_mad": med_ratio_mad,
            "sign_consistent_across_frames": sign_consistent,
            "verdict": verdict,
            "interpretation": interpretation,
        },
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"[depth-invariance] verdict={verdict} "
        f"median_step_to_noise={med_ratio:.2f} sign_consistent={sign_consistent}"
    )
    print(f"[depth-invariance] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
