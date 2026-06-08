# P2 ARKit gsplat render on Colab

This folder contains the Colab-side renderer for P2. It intentionally avoids
IsaacGym and the GaussGym package import path. Local verification is static only:
real RGB/depth rendering must be run in Colab with a CUDA GPU, for example a free
T4 runtime.

## Files

- `standalone_gs_renderer.py`: standalone `BatchPLYRenderer` plus `exp_map_SE3`.
- `p2_render_arkit.py`: notebook-style script with `# %%` cells to paste into
  Colab.

## Colab run

1. Open a Colab notebook.
2. Select `Runtime > Change runtime type > T4 GPU`.
3. Upload `standalone_gs_renderer.py` beside the notebook, then paste the
   `p2_render_arkit.py` cells into the notebook. The `.py` file uses Colab/Jupyter
   magics, so it is not meant to be run with plain `python`.
4. Run the cells top to bottom.
5. When prompted, paste a Hugging Face token via `getpass`. The script passes the
   token directly to `huggingface_hub.login`, does not echo it, does not hardcode
   it, and does not write it into this repository.
6. The script downloads one ARKit scene from
   `escontra/gauss_gym_arkit` using `snapshot_download(..., repo_type="dataset",
   allow_patterns=[...])`.
7. It renders `FRAME_COUNT` frames through `BatchPLYRenderer.batch_render` and
   writes:
   - `rgb/*.png`
   - `depth/*.npy`
   - `meta.json`
   - `/content/p2_arkit_render.zip`
8. Download the zip from Colab.

## Camera and poses

The notebook uses the scene's own ARKit `lowres_wide.traj` camera trajectory
rather than a hand-authored path. Pose conversion mirrors GaussGym ARKit ingest:

1. Parse trajectory rows as timestamp, axis-angle rotation, translation.
2. Build the ARKit transform and invert it.
3. Apply the GaussGym/nerfstudio ARKit-to-OpenGL transform.
4. Right-multiply `diag(1, -1, -1)` to convert OpenGL camera axes to OpenCV.

The script then stores scene-camera `cam_trans` and `cam_quat_xyzw`, rebuilds
`c2ws` from those values, and passes the final tensor to `batch_render` with
shape `(N, 4, 4)`. The convention is OpenCV camera-to-world, and quaternion
order is xyzw.

Intrinsics default to the GaussGym A1 `config_vision.yaml` camera values with the
same downscale:

- original image: `424 x 240`
- downscale: `5.0`
- render image: `84 x 48`
- `fl_x = 303.042358398438 / 5`
- `fl_y = 303.17578125 / 5`
- `pp_x = 213.575500488281 / 5`
- `pp_y = 124.642997741699 / 5`

## Bring output back to this repo

After downloading `p2_arkit_render.zip`, unpack it under a new local data folder,
for example:

```bash
mkdir -p data/p2_arkit_render
unzip p2_arkit_render.zip -d data/p2_arkit_render
```

The expected unpacked structure is:

```text
data/p2_arkit_render/p2_arkit_render/
  rgb/
  depth/
  meta.json
```

Use that folder as the RGB/depth input for the later local DINO cost-map step.
That DINO step is outside this P2 Colab-render task.

## Known Colab risks

- `gsplat` wheel compatibility can vary with Colab's current CUDA/PyTorch image.
  If install fails, restart the runtime and reinstall, or pin a `gsplat` version
  compatible with the active Colab CUDA stack.
- Large ARKit splats can exceed T4 memory. Lower `FRAME_COUNT` or `MINIBATCH`.
- Scene choice matters. `SCENE_PATH` defaults to a GaussGym ARKit config scene;
  override it with a verified carpet/rug scene path if visual inspection shows
  the default is not suitable.
- Intrinsics are A1 config values downscaled by `5.0`, not per-frame ARKit
  `.pincam` values. If the dataset scene includes preferred intrinsics in a
  future layout, update the constants and record the change in `meta.json`.
