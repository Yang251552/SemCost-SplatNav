# D0.1 — GaussGym 依赖栈核实结论（2026-05-31）

来源：直接 clone `github.com/escontra/gauss_gym`(default branch `main`) 读
`setup_dev.sh` / `pyproject.toml` / `uv.lock` / `environment.yml` / `docker/Dockerfile` /
`gauss_gym/envs/*/config_vision.yaml`。非搜索猜测，是仓库权威文件 + 在线探测。

## 1. 命门答案：是 IsaacGym Preview 4（legacy），不是 Isaac Lab

- `setup_dev.sh` L36–39：`wget https://developer.nvidia.com/isaac-gym-preview-4 -O IsaacGym_Preview_4_Package.tar.gz` 后 `tar -xzf`。
- `uv.lock`：`isaacgym  version = "1.0rc4"  source = editable "isaacgym/python"`。
- `pyproject.toml`：`description = "Photorealistic Isaac Gym Environments"`，`requires-python = ">=3.8,<3.9"`。
- → 确认依赖**已被 NVIDIA 弃用的 IsaacGym Preview 4**。这是预期中的最大风险点。

## 2. 精确版本锁（直接抄进 env.yaml / AMI）

| 组件 | 版本 | 来源 |
|---|---|---|
| Python | **3.8**（`>=3.8,<3.9` 强约束） | pyproject |
| CUDA toolkit | **12.1**（conda `nvidia/label/cuda-12.1.0`） | environment.yml |
| PyTorch | **2.4.1** + torchvision 0.19.1（cu121 轮子） | pyproject / Dockerfile |
| gsplat | **1.5.3**（pin git commit `2323de5`） | uv.lock |
| IsaacGym | **Preview 4 / 1.0rc4** | setup_dev.sh / uv.lock |
| warp-lang | 1.7.1 | pyproject |
| numpy | 1.23.5 | uv.lock |
| RL / PPO | **仓库自带 PPO**（`gauss_gym/rl/loss.py: learn_ppo`，非外部 rsl_rl/SB3） | 源码 |
| gcc/g++ | 11（conda gxx_linux-64=11） | environment.yml |

两条安装路径：
- **conda**：`bash setup_dev.sh` → 装到 `~/.gauss_gym_deps`（README 官方）。
- **Docker**：base `nvcr.io/nvidia/pytorch:23.04-py3`，`docker/build.sh` 自动按 `nvidia-smi` 填 `TORCH_CUDA_ARCH_LIST`。AWS 上更可复现，**推荐**。

## 3. 重大利好（去掉最大安装阻塞）

**IsaacGym Preview 4 现在可免登录直接下载。** 实测 `curl -IL https://developer.nvidia.com/isaac-gym-preview-4`：
301→302 重定向到带 token 的 `developer.download.nvidia.com/.../IsaacGym_Preview_4_Package.tar.gz`，
返回 `HTTP 200 / application/x-gzip`。
→ 历史上 IsaacGym 要 NVIDIA 开发者账号 + 手动门控；现在 `setup_dev.sh` 的自动 `wget` 能直接跑通。
**plan §5 里"IsaacGym 装不上"的最大不确定项从『高』降到『中』**（仍需在 A10G 上验证编译/运行）。

## 4. A10G 适配评估（24GB）

- vision config 默认 `num_envs: 2048`、`terrain.max_num_scenes: 10` → A10G 上按显存降到 ~512–1024。
- 图像观测极小：`camera_image_shape: [3, 58, 87]`（RGB），相机 `resize: [58, 87]`、`far_clip: 10m`。
  → 渲染负载轻，24GB 充裕。A10G = Ampere(compute 8.6)，CUDA 12.1 完全支持。
- `gs_render`/`force_renderer` 可开关；blind 策略默认不渲染高斯。

## 5. 对本项目（三路消融）的关键影响 — 需你知晓

1. **观测开关天然适配三路消融**：vision config 里已有
   `pixel_obs: True`、`use_depth: False`、`use_depth_only: False`、`camera_image_shape:[3,58,87]`。
   → 我们的三路 = **depth-only**(`use_depth_only:True`) / **RGB**(`pixel_obs RGB`) / **RGB+semantic**(再加一路 semantic_cost 通道)，
   切的就是这几个 flag + 加通道，改动点清晰，公平性容易保证。
2. **机器人是腿足运动，不是 P1 的轮式导航**：已验证配置 = `a1`/`go1`(四足)、`t1`(人形)、`anymal_c`。
   README 注明"只有 a1 和 t1 在硬件上验证过"。
   → 叙事从"轮式避障导航"微调为"**腿足在含 hazard 地形上运动 + 语义代价**"。仍贴 GaussGym 本行，
   但和 P1 网格不是同一形态。**这点要你确认接受。**
3. **场景从 HF 自动下载，真实仓库名（读源码 `scene_ingest.py` 得到，全部实测 HTTP 200 公开）**：
   - `escontra/gauss_gym_data`（**当前默认开**，`iphone_data` split_frac=1.0，iPhone/Polycam 扫描，**以楼梯/台阶为主**：bww_stairs / long_stairs 等）
   - `escontra/gauss_gym_arkit`（室内房间 ARKitScenes，config 里 split_frac=0 默认关；**最适合"室内地毯/材质过渡"hazard**）
   - `escontra/veo_scenes`（生成式视频场景）；户外走 `leggedrobotics/grand_tour_dataset`（GrandTour，config 里默认关）
   - gating：API 层全部 200（公开可列），**实际下载需 `huggingface-cli login` token**；GrandTour 可能要一次性同意条款。不是硬墙。
   → 见 D0.3。**关键：默认场景是楼梯=几何丰富，对"几何不可区分 hazard"实验不合适，需 override 场景。**

## 6. 结论

- 依赖栈完全查清，可进入 D0.2 写 `env.yaml` / 选 AMI。
- 最大风险（IsaacGym 安装）因免登录下载而显著下降，但**仍须在真实 A10G 上跑通 smoke** 才算落地（D3 付费步骤，有 6h timebox + 降级路径兜底）。
- **新决策点（要你拍板）**：
  a. 接受"**腿足运动 + 语义地形**"形态（而非 P1 的轮式导航）？
  b. 场景族优先 **arkit 室内**（配地毯/水渍/杂物 hazard）还是 grand_tour 户外？
  这两项会和 D0.3 的 hazard 物体一起定。
