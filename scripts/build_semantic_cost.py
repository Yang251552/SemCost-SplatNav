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
    p.add_argument("--rollout-rgb", required=True)
    p.add_argument("--out", default=str(ROOT / "results" / "semantic_cost_maps.npz"))
    p.add_argument("--pca", default=str(ROOT / "results" / "dino_pca64.npy"))
    p.add_argument("--bad-proto", default=str(ROOT / "results" / "bad_proto.npy"))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--parity-report", default=str(ROOT / "results" / "parity_report.json"))
    return p.parse_args()


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

        from semcost_nav.semantic.dino_cost import DinoCostMapper
    except Exception as exc:
        print(f"# TODO: run on GPU instance ({exc})")
        return 2

    rgb = np.load(args.rollout_rgb)
    if isinstance(rgb, np.lib.npyio.NpzFile):
        rgb = rgb["rgb"]
    try:
        mapper = DinoCostMapper(args.pca, args.bad_proto)
    except NotImplementedError as exc:
        print(f"# TODO: run on GPU instance ({exc})")
        return 2
    cost = mapper(rgb)
    np.savez(out_path, cost_maps=cost.detach().cpu().numpy())
    write_parity_template(report_path, num_frames=int(rgb.shape[0]))
    print(f"[semantic-cost] wrote {out_path}")
    print(f"[semantic-cost] wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
