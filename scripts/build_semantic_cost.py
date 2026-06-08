"""Build post-hoc semantic cost maps from rollout RGB frames."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build DINO semantic cost maps.")
    group = p.add_mutually_exclusive_group(required=False)
    group.add_argument("--rollout-rgb", default=None)
    group.add_argument("--rgb-dir", default=None)
    p.add_argument("--out", default=str(ROOT / "results" / "semantic_cost_maps.npz"))
    p.add_argument("--pca", default=str(ROOT / "results" / "dino_pca64.npy"))
    p.add_argument("--bad-proto", default=str(ROOT / "results" / "bad_proto.npy"))
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--parity-report", default=str(ROOT / "results" / "parity_report.json"))
    args = p.parse_args()
    if not args.dry_run and not args.rollout_rgb and not args.rgb_dir:
        p.error("one of --rollout-rgb or --rgb-dir is required")
    return args


def write_parity_template(path: Path, num_frames: int | None = None) -> None:
    payload = {
        "IoU": None,
        "hazard_class": "carpet",
        "num_frames": num_frames,
        "bad_proto_source": None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    out_path = Path(args.out)
    report_path = Path(args.parity_report)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        import numpy as np

        np.savez(out_path, cost_maps=np.zeros((0, 1, 0, 0), dtype=np.float32))
        write_parity_template(report_path, num_frames=0)
        print(f"[semantic-cost] dry-run wrote {out_path}")
        print(f"[semantic-cost] dry-run wrote {report_path}")
        return 0

    try:
        import numpy as np
        from PIL import Image
        import torch

        from semcost_nav.semantic.dino_cost import DinoCostMapper
    except Exception as exc:
        print(f"[semantic-cost] missing dependency: {exc}", file=sys.stderr)
        return 2

    try:
        mapper = DinoCostMapper(args.pca, args.bad_proto)
    except NotImplementedError as exc:
        print(f"[semantic-cost] missing dependency: {exc}", file=sys.stderr)
        return 2

    if args.rgb_dir:
        paths = sorted(Path(args.rgb_dir).glob("*.png"))
        if not paths:
            raise SystemExit(f"No PNG frames found in {args.rgb_dir}")
        frame_names = np.asarray([path.stem for path in paths])
        batches = []
        for start in range(0, len(paths), args.batch_size):
            batch_paths = paths[start : start + args.batch_size]
            rgb_np = np.stack(
                [
                    np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
                    for path in batch_paths
                ],
                axis=0,
            )
            rgb_tensor = torch.from_numpy(rgb_np).permute(0, 3, 1, 2)
            cost = mapper(rgb_tensor).detach().cpu().numpy().astype(np.float32)
            batches.append(cost)
            done = min(start + len(batch_paths), len(paths))
            print(f"[semantic-cost] processed {done}/{len(paths)} frames")
        cost_maps = np.concatenate(batches, axis=0)
    else:
        loaded = np.load(args.rollout_rgb)
        rgb = loaded["rgb"] if isinstance(loaded, np.lib.npyio.NpzFile) else loaded
        rgb = np.asarray(rgb, dtype=np.float32)
        if rgb.ndim != 4:
            raise SystemExit("rollout RGB must be shaped (N,H,W,3) or (N,3,H,W)")
        if rgb.shape[1] == 3:
            rgb_tensor = torch.from_numpy(rgb)
        elif rgb.shape[-1] == 3:
            rgb_tensor = torch.from_numpy(rgb).permute(0, 3, 1, 2)
        else:
            raise SystemExit("rollout RGB must have 3 channels")
        if float(rgb_tensor.max()) > 1.0:
            rgb_tensor = rgb_tensor / 255.0
        frame_names = np.asarray([f"{idx:04d}" for idx in range(rgb_tensor.shape[0])])
        batches = []
        for start in range(0, rgb_tensor.shape[0], args.batch_size):
            batch = rgb_tensor[start : start + args.batch_size]
            cost = mapper(batch).detach().cpu().numpy().astype(np.float32)
            batches.append(cost)
            done = min(start + batch.shape[0], rgb_tensor.shape[0])
            print(f"[semantic-cost] processed {done}/{rgb_tensor.shape[0]} frames")
        cost_maps = np.concatenate(batches, axis=0)

    np.savez(out_path, cost_maps=cost_maps, frame_names=frame_names)
    write_parity_template(report_path, num_frames=int(cost_maps.shape[0]))
    print(f"[semantic-cost] wrote {out_path}")
    print(f"[semantic-cost] wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
