# ruff: noqa: F722
"""Standalone gsplat PLY renderer for Colab P2 rendering.

This file is adapted from GaussGym's BatchPLYRenderer, but intentionally has no
package-level simulator dependency. It is meant to be copied next to the Colab
render script and imported directly.
"""

from pathlib import Path
import json
from typing import Tuple, Union

import numpy as np
import torch
from einops import rearrange, repeat
from gsplat.rendering import rasterization
from jaxtyping import Float
from plyfile import PlyData
from torch import Tensor


def exp_map_SE3(tangent_vector: Float[Tensor, "b 6"]) -> Float[Tensor, "b 3 4"]:
  """Compute the exponential map `se(3) -> SE(3)`."""

  tangent_vector_lin = tangent_vector[:, :3].view(-1, 3, 1)
  tangent_vector_ang = tangent_vector[:, 3:].view(-1, 3, 1)

  theta = torch.linalg.norm(tangent_vector_ang, dim=1).unsqueeze(1)
  theta2 = theta**2
  theta3 = theta**3

  near_zero = theta < 1e-2
  non_zero = torch.ones(1, dtype=tangent_vector.dtype, device=tangent_vector.device)
  theta_nz = torch.where(near_zero, non_zero, theta)
  theta2_nz = torch.where(near_zero, non_zero, theta2)
  theta3_nz = torch.where(near_zero, non_zero, theta3)

  sine = theta.sin()
  cosine = torch.where(near_zero, 8 / (4 + theta2) - 1, theta.cos())
  sine_by_theta = torch.where(near_zero, 0.5 * cosine + 0.5, sine / theta_nz)
  one_minus_cosine_by_theta2 = torch.where(
    near_zero, 0.5 * sine_by_theta, (1 - cosine) / theta2_nz
  )
  ret = torch.zeros(tangent_vector.shape[0], 3, 4).to(
    dtype=tangent_vector.dtype, device=tangent_vector.device
  )
  ret[:, :3, :3] = (
    one_minus_cosine_by_theta2 * tangent_vector_ang @ tangent_vector_ang.transpose(1, 2)
  )

  ret[:, 0, 0] += cosine.view(-1)
  ret[:, 1, 1] += cosine.view(-1)
  ret[:, 2, 2] += cosine.view(-1)
  temp = sine_by_theta.view(-1, 1) * tangent_vector_ang.view(-1, 3)
  ret[:, 0, 1] -= temp[:, 2]
  ret[:, 1, 0] += temp[:, 2]
  ret[:, 0, 2] += temp[:, 1]
  ret[:, 2, 0] -= temp[:, 1]
  ret[:, 1, 2] -= temp[:, 0]
  ret[:, 2, 1] += temp[:, 0]

  sine_by_theta = torch.where(near_zero, 1 - theta2 / 6, sine_by_theta)
  one_minus_cosine_by_theta2 = torch.where(
    near_zero, 0.5 - theta2 / 24, one_minus_cosine_by_theta2
  )
  theta_minus_sine_by_theta3_t = torch.where(
    near_zero, 1.0 / 6 - theta2 / 120, (theta - sine) / theta3_nz
  )

  ret[:, :, 3:] = sine_by_theta * tangent_vector_lin
  ret[:, :, 3:] += one_minus_cosine_by_theta2 * torch.cross(
    tangent_vector_ang, tangent_vector_lin, dim=1
  )
  ret[:, :, 3:] += theta_minus_sine_by_theta3_t * (
    tangent_vector_ang @ (tangent_vector_ang.transpose(1, 2) @ tangent_vector_lin)
  )
  ret = torch.cat([ret, torch.zeros(ret.shape[0], 1, 4, device=ret.device)], dim=1)
  ret[:, 3, 3] = 1.0
  return ret


class BatchPLYRenderer:
  def __init__(self, ply_path: Union[str, Path], device: str = "cuda"):
    """
    Initialize renderer with Gaussians loaded from a PLY file.

    Args:
        ply_path: Path to the .ply file containing Gaussian parameters.
        device: Device to run computations on ("cuda" or "cpu").
    """
    self.device = torch.device(device)
    self.ply_path = Path(ply_path)

    (
      self.means,
      self.scales,
      self.quats,
      self.colors,
      self.colors_viser,
      self.opacities,
      self.sh_degree,
    ) = self._load_ply_gaussians(self.ply_path)

    splat_json_path = self.ply_path.parent / (self.ply_path.stem + "_splat.json")
    dataparser_json_path = self.ply_path.parent / "dataparser_transforms.json"
    if splat_json_path.exists():
      json_path = splat_json_path
    elif dataparser_json_path.exists():
      json_path = dataparser_json_path
    else:
      raise RuntimeError(f"Could not find JSON file for {self.ply_path}")

    with json_path.open("r") as f:
      json_data = json.load(f)
      self.dataparser_scale = json_data["scale"]
      self.dataparser_transform = (
        torch.tensor(json_data["transform"]).float().to(self.device)
      )
      self.dataparser_transform = torch.cat(
        [
          self.dataparser_transform,
          torch.tensor([[0, 0, 0, 1]], device=self.device, dtype=torch.float32),
        ],
        dim=0,
      )

  def _load_ply_gaussians(self, ply_path: Path):
    """Load Gaussian parameters from a PLY file."""
    with ply_path.open("rb") as f:
      plydata = PlyData.read(f)
      v = plydata["vertex"]

    means = (
      torch.from_numpy(np.stack([v["x"], v["y"], v["z"]], axis=-1))
      .float()
      .to(self.device)
    )

    scales = (
      torch.from_numpy(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=-1))
      .float()
      .to(self.device)
    )
    scales = torch.exp(scales)

    quats = (
      torch.from_numpy(
        np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=-1)
      )
      .float()
      .to(self.device)
    )

    sh_c0 = 0.28209479177387814
    colors_viser = (
      torch.from_numpy(
        0.5 + sh_c0 * np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1)
      )
      .float()
      .to(self.device)
    )

    sh_components = [np.stack([v[f"f_dc_{i}"] for i in range(3)], axis=-1)]

    num_rest_coeffs = sum(1 for name in v.dtype().names if name.startswith("f_rest_"))
    num_bases = num_rest_coeffs // 3
    for i in range(num_bases):
      sh_3chan = np.zeros((len(v), 3))
      for j in range(3):
        sh_3chan[:, j] = v[f"f_rest_{i + num_bases * j}"]
      sh_components.append(sh_3chan)

    colors = torch.from_numpy(np.stack(sh_components, axis=1)).float().to(self.device)
    opacities = torch.from_numpy(v["opacity"]).float().to(self.device)
    opacities = torch.sigmoid(opacities)

    sh_degree = int(np.sqrt(colors.shape[1]) - 1)

    return means, scales, quats, colors, colors_viser, opacities, sh_degree

  @torch.no_grad()
  def batch_render(
    self,
    c2ws: Float[Tensor, "num_cameras 4 4"],
    fl_x: Union[float, Tensor],
    fl_y: Union[float, Tensor],
    pp_x: Union[float, Tensor],
    pp_y: Union[float, Tensor],
    h: int,
    w: int,
    camera_linear_velocity: Union[Float[Tensor, "num_cameras 3"], None] = None,
    camera_angular_velocity: Union[Float[Tensor, "num_cameras 3"], None] = None,
    motion_blur_frac: float = 0.0,
    blur_samples: int = 5,
    blur_dt: float = 0.03,
    minibatch: int = 15,
    out_device: str = "cpu",
  ) -> Tuple[Tensor, Tensor]:
    """
    Batch render a set of cameras.

    Args:
        c2ws: Camera-to-world matrices (4x4 homogeneous, OpenCV format).
        fl_x: X focal length for all cameras or a per-camera tensor.
        fl_y: Y focal length for all cameras or a per-camera tensor.
        pp_x: X principal point for all cameras or a per-camera tensor.
        pp_y: Y principal point for all cameras or a per-camera tensor.
        h: Image height.
        w: Image width.
        minibatch: Number of images to render in each batch.
        out_device: Device for returned tensors.

    Returns:
        Tuple containing RGB uint8 (num_cameras, h, w, 3) and depth float32
        (num_cameras, h, w, 1).
    """
    out_device = torch.device(out_device) if out_device else self.device

    c2ws = torch.bmm(self.dataparser_transform.repeat(c2ws.shape[0], 1, 1), c2ws)
    c2ws[:, :3, 3] *= self.dataparser_scale
    w2cs = torch.inverse(c2ws)
    num_images = c2ws.shape[0]
    if isinstance(fl_x, float):
      fl_x = torch.full((num_images,), fl_x, device=self.device)
    if isinstance(fl_y, float):
      fl_y = torch.full((num_images,), fl_y, device=self.device)
    if isinstance(pp_x, float):
      pp_x = torch.full((num_images,), pp_x, device=self.device)
    if isinstance(pp_y, float):
      pp_y = torch.full((num_images,), pp_y, device=self.device)

    k = torch.eye(3, device=self.device, dtype=torch.float32)[None].repeat(
      num_images, 1, 1
    )
    k[:, 0, 0] = fl_x
    k[:, 1, 1] = fl_y
    k[:, 0, 2] = pp_x
    k[:, 1, 2] = pp_y

    imgs_out = torch.empty((num_images, h, w, 3), device=out_device, dtype=torch.uint8)
    depth_out = torch.empty(
      (num_images, h, w, 1), device=out_device, dtype=torch.float32
    )

    for i in range(0, num_images, minibatch):
      batch_w2cs = w2cs[i : i + minibatch]
      batch_k = k[i : i + minibatch]

      original_b_size = len(batch_w2cs)
      blur_batch = torch.rand(1).item() < motion_blur_frac
      if blur_batch:
        if camera_linear_velocity is None or camera_angular_velocity is None:
          raise ValueError("Motion blur requires camera velocity tensors.")
        dts = -torch.linspace(0, blur_dt, blur_samples, device=self.device)
        batch_w2cs = repeat(
          batch_w2cs, "batch mat1 mat2 -> (batch r) mat1 mat2", r=blur_samples
        )
        batch_k = repeat(
          batch_k, "batch mat1 mat2 -> (batch r) mat1 mat2", r=blur_samples
        )
        batch_vels = torch.cat(
          [
            camera_linear_velocity[i : i + minibatch],
            camera_angular_velocity[i : i + minibatch],
          ],
          dim=-1,
        )
        batch_vels = repeat(batch_vels, "batch vel -> (batch r) vel", r=blur_samples)
        dts = repeat(dts, "samples -> (batch samples) 1", batch=original_b_size)
        batch_delta_mats = exp_map_SE3(batch_vels * dts)
        batch_w2cs = torch.bmm(batch_w2cs, torch.inverse(batch_delta_mats))

      with torch.no_grad():
        colors, _, _ = rasterization(
          self.means,
          self.quats,
          self.scales,
          self.opacities,
          self.colors,
          batch_w2cs,
          batch_k,
          w,
          h,
          radius_clip=3.0,
          sh_degree=self.sh_degree,
          render_mode="RGB+D",
        )
      depths = colors[..., 3:]
      colors = colors[..., :3]
      colors.clamp_(0, 1)
      if blur_batch:
        colors = rearrange(colors, "(b r) h w c -> b r h w c", r=blur_samples)
        depths = rearrange(depths, "(b r) h w c -> b r h w c", r=blur_samples)
        colors = colors.mean(dim=1)
        depths = depths[:, 0]
      assert colors.shape == (original_b_size, h, w, 3), (
        f"Colors shape is {colors.shape} but should be {original_b_size, h, w, 3}"
      )

      imgs_out[i : i + minibatch] = (
        255 * colors[..., :3].to(out_device, non_blocking=True)
      ).to(torch.uint8)
      depth_out[i : i + minibatch] = (depths / self.dataparser_scale).to(
        out_device, non_blocking=True
      )

    return imgs_out, depth_out
