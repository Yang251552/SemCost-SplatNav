# %% [markdown]
# P2 ARKit gsplat render notebook script.
#
# Paste this file into a Colab notebook or upload it with
# `standalone_gs_renderer.py`, then run cells top to bottom on a T4 GPU runtime.
# Local CPU-only/static checks cannot validate the actual render.

# %%
%pip install -q gsplat plyfile einops jaxtyping huggingface_hub pillow

# %%
from getpass import getpass
from huggingface_hub import login

hf_token = getpass("Hugging Face token: ")
login(token=hf_token, add_to_git_credential=False)
del hf_token

# %%
from pathlib import Path
import json
import math
import shutil
import zipfile

import numpy as np
import torch
from huggingface_hub import HfApi, snapshot_download
from PIL import Image

from standalone_gs_renderer import BatchPLYRenderer


REPO_ID = "escontra/gauss_gym_arkit"
REPO_TYPE = "dataset"

# GaussGym's checked-in ARKit scene-generation config references these IDs.
# Override SCENE_PATH if you know a more specific carpet/rug scene in the HF repo.
SCENE_PATH = "training/43895956"
FALLBACK_SCENE_PATHS = ["training/43649417", "training/47334672"]

FRAME_COUNT = 80
OUTPUT_DIR = Path("/content/p2_arkit_render")
ZIP_PATH = Path("/content/p2_arkit_render.zip")

# A1 config_vision camera values, downscaled by the same factor used by GaussGym sensors.
ORIG_W = 424
ORIG_H = 240
DOWNSCALE = 5.0
W = int(ORIG_W / DOWNSCALE)
H = int(ORIG_H / DOWNSCALE)
FL_X = 303.042358398438 / DOWNSCALE
FL_Y = 303.17578125 / DOWNSCALE
PP_X = 213.575500488281 / DOWNSCALE
PP_Y = 124.642997741699 / DOWNSCALE

DEVICE = "cuda"
MINIBATCH = 8


# %%
def scene_has_required_files(scene_path: str) -> bool:
  api = HfApi()
  files = api.list_repo_files(repo_id=REPO_ID, repo_type=REPO_TYPE)
  prefix = scene_path.rstrip("/") + "/"
  scene_files = [f for f in files if f.startswith(prefix)]
  video_id = scene_path.rstrip("/").split("/")[-1]
  required_suffixes = [
    "lowres_wide.traj",
    "splatfacto/splat.ply",
    "splatfacto/dataparser_transforms.json",
    f"{video_id}_3dod_mesh_filled_decimated.ply",
  ]
  return all(prefix + suffix in scene_files for suffix in required_suffixes)


def choose_scene_path() -> str:
  candidates = [SCENE_PATH] + FALLBACK_SCENE_PATHS
  for scene_path in candidates:
    if scene_has_required_files(scene_path):
      return scene_path
  raise RuntimeError(
    "None of the configured scene paths has the required ARKit files. "
    "Set SCENE_PATH to a valid '<split>/<video_id>' from the dataset."
  )


scene_path = choose_scene_path()
video_id = scene_path.split("/")[-1]
allow_patterns = [
  f"{scene_path}/lowres_wide.traj",
  f"{scene_path}/splatfacto/splat.ply",
  f"{scene_path}/splatfacto/dataparser_transforms.json",
  f"{scene_path}/{video_id}_3dod_mesh_filled_decimated.ply",
]

snapshot_root = Path(
  snapshot_download(
    repo_id=REPO_ID,
    repo_type=REPO_TYPE,
    allow_patterns=allow_patterns,
  )
)
local_scene = snapshot_root / scene_path
ply_path = local_scene / "splatfacto" / "splat.ply"
traj_path = local_scene / "lowres_wide.traj"

print(f"Using scene: {scene_path}")
print(f"PLY: {ply_path}")
print(f"Trajectory: {traj_path}")


# %%
def axis_angle_to_matrix(axis_angle: np.ndarray) -> np.ndarray:
  theta = float(np.linalg.norm(axis_angle))
  if theta < 1e-12:
    return np.eye(3, dtype=np.float64)
  axis = axis_angle / theta
  x, y, z = axis
  skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float64)
  return (
    np.eye(3, dtype=np.float64)
    + math.sin(theta) * skew
    + (1.0 - math.cos(theta)) * (skew @ skew)
  )


def parse_traj_file(path: Path) -> list[np.ndarray]:
  poses = []
  with path.open("r") as f:
    for line in f:
      parts = line.strip().split()
      if len(parts) < 7:
        continue
      axis_angle = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
      translation = np.array([float(parts[4]), float(parts[5]), float(parts[6])])
      transform = np.eye(4, dtype=np.float64)
      transform[:3, :3] = axis_angle_to_matrix(axis_angle)
      transform[:3, 3] = translation
      poses.append(np.linalg.inv(transform))
  return poses


def apply_arkit_to_opengl_transform(pose: np.ndarray) -> np.ndarray:
  frame_pose = pose.copy()
  frame_pose[0:3, 1:3] *= -1
  frame_pose = frame_pose[np.array([1, 0, 2, 3]), :]
  frame_pose[2, :] *= -1
  return frame_pose


def opengl_c2w_to_opencv_c2w(opengl_c2w: np.ndarray) -> np.ndarray:
  # GaussGym math_utils.opengl_to_opencv multiplies xyzw rotation by a 180 deg X
  # rotation. Matrix form is the same right-multiplication on camera axes.
  opengl_to_opencv = np.eye(4, dtype=np.float64)
  opengl_to_opencv[:3, :3] = np.diag([1.0, -1.0, -1.0])
  return opengl_c2w @ opengl_to_opencv


def matrix_to_quaternion_xyzw(matrix: np.ndarray) -> np.ndarray:
  trace = float(np.trace(matrix))
  if trace > 0.0:
    s = math.sqrt(trace + 1.0) * 2.0
    qw = 0.25 * s
    qx = (matrix[2, 1] - matrix[1, 2]) / s
    qy = (matrix[0, 2] - matrix[2, 0]) / s
    qz = (matrix[1, 0] - matrix[0, 1]) / s
  elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
    s = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
    qw = (matrix[2, 1] - matrix[1, 2]) / s
    qx = 0.25 * s
    qy = (matrix[0, 1] + matrix[1, 0]) / s
    qz = (matrix[0, 2] + matrix[2, 0]) / s
  elif matrix[1, 1] > matrix[2, 2]:
    s = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
    qw = (matrix[0, 2] - matrix[2, 0]) / s
    qx = (matrix[0, 1] + matrix[1, 0]) / s
    qy = 0.25 * s
    qz = (matrix[1, 2] + matrix[2, 1]) / s
  else:
    s = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
    qw = (matrix[1, 0] - matrix[0, 1]) / s
    qx = (matrix[0, 2] + matrix[2, 0]) / s
    qy = (matrix[1, 2] + matrix[2, 1]) / s
    qz = 0.25 * s
  quat = np.array([qx, qy, qz, qw], dtype=np.float64)
  return quat / np.linalg.norm(quat)


def quaternion_xyzw_to_matrix(quat: np.ndarray) -> np.ndarray:
  x, y, z, w = quat / np.linalg.norm(quat)
  return np.array(
    [
      [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ],
    dtype=np.float64,
  )


def c2ws_from_trans_quat_xyzw(cam_trans: np.ndarray, cam_quat_xyzw: np.ndarray) -> np.ndarray:
  c2ws = np.tile(np.eye(4, dtype=np.float64), (cam_trans.shape[0], 1, 1))
  c2ws[:, :3, 3] = cam_trans
  for i, quat in enumerate(cam_quat_xyzw):
    c2ws[i, :3, :3] = quaternion_xyzw_to_matrix(quat)
  return c2ws.astype(np.float32)


def load_scene_camera_poses(
  traj_path: Path, frame_count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
  opengl_poses = [apply_arkit_to_opengl_transform(pose) for pose in parse_traj_file(traj_path)]
  if len(opengl_poses) < frame_count:
    raise RuntimeError(f"Only found {len(opengl_poses)} trajectory poses.")
  frame_indices = np.linspace(0, len(opengl_poses) - 1, frame_count, dtype=np.int64)
  # Use the OpenCV poses directly as c2ws. The previous matrix->quat->matrix
  # round-trip only existed to export cam_quat for meta and added an avoidable
  # numerical failure surface (a buggy quat fn would silently corrupt c2ws and
  # be hard to debug on Colab). quat/trans below are meta-only.
  opencv_poses = np.stack(
    [opengl_c2w_to_opencv_c2w(opengl_poses[i]) for i in frame_indices], axis=0
  ).astype(np.float32)
  c2ws = opencv_poses
  cam_trans = opencv_poses[:, :3, 3].astype(np.float32)
  cam_quat_xyzw = np.stack(
    [matrix_to_quaternion_xyzw(pose[:3, :3]) for pose in opencv_poses], axis=0
  ).astype(np.float32)
  return c2ws, cam_trans, cam_quat_xyzw, frame_indices.tolist()


c2ws_np, cam_trans_np, cam_quat_xyzw_np, frame_indices = load_scene_camera_poses(
  traj_path, FRAME_COUNT
)
print(f"Prepared c2ws: {c2ws_np.shape}, selected {len(frame_indices)} frames")


# %%
if torch.cuda.is_available() is False:
  raise RuntimeError("CUDA is required. In Colab, choose Runtime > Change runtime type > T4 GPU.")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "rgb").mkdir(exist_ok=True)
(OUTPUT_DIR / "depth").mkdir(exist_ok=True)

renderer = BatchPLYRenderer(ply_path, device=DEVICE)
c2ws = torch.from_numpy(c2ws_np).to(DEVICE)

# Signature check: c2ws is (N,4,4); intrinsics are scalar; returns
# rgb uint8 (N,H,W,3), depth float32 (N,H,W,1).
rgb, depth = renderer.batch_render(
  c2ws,
  fl_x=float(FL_X),
  fl_y=float(FL_Y),
  pp_x=float(PP_X),
  pp_y=float(PP_Y),
  h=H,
  w=W,
  minibatch=MINIBATCH,
  out_device="cpu",
)

print(rgb.shape, rgb.dtype, depth.shape, depth.dtype)


# %%
for i in range(rgb.shape[0]):
  Image.fromarray(rgb[i].numpy()).save(OUTPUT_DIR / "rgb" / f"{i:04d}.png")
  np.save(OUTPUT_DIR / "depth" / f"{i:04d}.npy", depth[i].numpy())

meta = {
  "repo_id": REPO_ID,
  "scene_path": scene_path,
  "ply_path": str(ply_path),
  "traj_path": str(traj_path),
  "frame_indices": frame_indices,
  "camera_model": "A1 config_vision intrinsics downscaled for gsplat render",
  "intrinsics": {
    "orig_width": ORIG_W,
    "orig_height": ORIG_H,
    "downscale": DOWNSCALE,
    "width": W,
    "height": H,
    "fl_x": FL_X,
    "fl_y": FL_Y,
    "pp_x": PP_X,
    "pp_y": PP_Y,
  },
  "pose_convention": {
    "source": "ARKit lowres_wide.traj scene camera trajectory",
    "c2ws": "OpenCV camera-to-world matrices, shape (N,4,4)",
    "conversion": "ARKit traj -> GaussGym/nerfstudio OpenGL pose -> right-multiply diag(1,-1,-1) for OpenCV camera axes",
    "quaternion_order_if_exported": "xyzw",
  },
  "cam_trans": cam_trans_np.tolist(),
  "cam_quat_xyzw": cam_quat_xyzw_np.tolist(),
  "c2ws": c2ws_np.tolist(),
}

with (OUTPUT_DIR / "meta.json").open("w") as f:
  json.dump(meta, f, indent=2)

if ZIP_PATH.exists():
  ZIP_PATH.unlink()
with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
  for path in sorted(OUTPUT_DIR.rglob("*")):
    zf.write(path, arcname=path.relative_to(OUTPUT_DIR.parent))

print(f"Wrote {OUTPUT_DIR}")
print(f"Wrote {ZIP_PATH}")


# %%
from google.colab import files

files.download(str(ZIP_PATH))
