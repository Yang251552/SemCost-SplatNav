# HANDOFF — Stage-2 P2：渲染已成功 → 下一步本机 DINO cost map

> 新窗口先读这段。这是个**干净切点**：P2 渲染（Colab 免费 GPU）已成功出图，80 帧 RGB+depth 已下载到本机；下一步是 **P2 后半：本机跑 DINO 语义 cost map**（免 GPU，在 Mac 上跑）。
> 工作目录：`/Users/yangchenghan/Downloads/SP申请3/SemCost-SplatNav`
> 上游：`docs/AWS_FLAGSHIP_PLAN.md`(§4 P2/P3/P4 分阶段)、`docs/HANDOFF_STAGE2_EXECUTE.md`(骨架阶段)、`REVIEW_NOTES.md`/`EXPERIMENT_LOG.md` 的 2026-06-01 条目。

---

## 进度快照（2026-06-01）

- Stage-1 ✅（grid RL，ready_for_application）
- Stage-2 骨架 ✅（commit `95db91c`：configs/flagship.yaml、train/evaluate_gaussgym、stage2_config、observation_tokens、dino_cost 骨架、tests，pytest 5 绿）
- Stage-2 **P2 渲染 ✅ 刚完成**（本文件主题）
- Stage-2 **P2 后半（DINO cost map）← 你在这，本机免费可做**
- Stage-2 P3（三路 RL，付费 GPU）← 未做，需用户决策（且 P2 的 parity IoU 是 P3 的决策闸）

---

## P2 渲染产出（已验证成功）

- **方式**：走 Colab 免费 T4 GPU（**没烧 AWS 钱**，spot 没批所以避开 on-demand）。用 `scripts/colab/standalone_gs_renderer.py`（从 GaussGym `batch_gs_renderer.py` 抽的独立 gsplat 渲染器，已 source-grounded 审过）+ `scripts/colab/p2_render_arkit.py`。
- **场景**：`escontra/gauss_gym_arkit` 的 **`training/43895956`**（HF dataset，公开但下载需 HF token）。
- **结果**：80 帧 RGB(84×48 png) + depth(84×48 npy) + meta.json，打包成 `p2_arkit_render.zip`，**已下载到 `~/Downloads/`**。
- **关键验证（看图确认）**：
  - 渲染管线**完全正确**——RGB 是清晰真实客厅（黑沙发/窗/木地板/书架），坐标系(ARKit→OpenGL→OpenCV)+内参全对。
  - **场景含地毯** ✅——RGB 第 60 帧木地板上有花纹地毯，正是 hazard（地毯 vs 木地板：depth 几乎齐平、视觉材质可分，完美的"几何看不见/语义看得见"案例）。
  - **小瑕疵**：第 0 帧（RGB 0）全黑——相机轨迹起点 edge case，80 帧就这 1 帧，DINO 时跳过纯黑帧即可，不影响。
- **数据已就位**：解压到 `assets/p2_arkit_render/`（rgb/ depth/ meta.json）。新窗口直接从这里读，不用再去 Colab。

---

## 下一步：P2 后半 — 本机 DINO 语义 cost map（免 GPU）

目标产出（CLAUDE.md Stage-2 artifacts 的 P2 部分）：
1. **DINO cost map**：对 `assets/p2_arkit_render/rgb/*.png` 跑 → RGB→DINOv2 ViT-S→PCA64(冻结)→坏原型 cosine→软 cost map（方法 `docs/D0_3_scene_hazard.md §3`）。`semcost_nav/semantic/dino_cost.py` 已有骨架（之前是 import-guard NotImplementedError，**现在要让它本机真跑**——DINOv2 `facebook/dinov2-small` 公开免 token，CPU 能跑只是慢）。
2. **parity_report.json**：cost map vs **人标地毯 mask** 的 **IoU≥0.5**（需手标 3-5 帧地毯掩码；含地毯的帧如 RGB 60）。`scripts/build_semantic_cost.py` 有骨架。
3. **depth 不变性检验**：地毯区 vs 周围 depth 统计无差异（证"depth 看不见地毯"）——用下载的 depth/*.npy。
4. **四联图** `figures/rgbd_cost_avoid_quad.png`（RGB / depth / cost / avoid-mask 4-up）。
5. （可选）`renders/cost_map_overlay.mp4`。

**P2 验收 gate**：IoU≥0.5 + depth 不变性成立 → P2 完成，且支持"值得冲 P3"。若 IoU 擦边/depth 能看见地毯 → 按 CLAUDE.md stop condition，重选场景 or 诚实报告负面结论（**不许伪造 IoU**）。

**实施分工**：按 teammate 规则，DINO pipeline 真代码是 EXECUTE→派 Codex 主写（`serial --intent implement`，前台+超时兜底），Claude source-grounded 审。先 Claude source-ground：dino_cost.py 现状 + DINOv2 transformers API。

---

## 环境坑 + 解法（Colab gsplat，未来若重跑 Colab 必看）

Colab 现状：**python 3.12 + torch 2.11 + cu128**（太新）。gsplat 装不上的连环坑，最终解法（已验证成功）：
1. 降 torch：`pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121` → **重启 runtime**。
2. 装 gsplat **源码版**：`pip install gsplat==1.5.3 plyfile einops jaxtyping huggingface_hub pillow matplotlib`（预编译 wheel 只有 cp310，python 3.12 装不了 → 必须源码编译）。
3. **一次性现场编译**（~13 分钟，**设 `TORCH_CUDA_ARCH_LIST=7.5`(T4) + `MAX_JOBS=4`，绝不中断**；中断会留半成品缓存导致 `gsplat_cuda.so not found`，要 `shutil.rmtree("/root/.cache/torch_extensions")` 清掉重编）。编译成功后缓存住，后续秒过。
4. `standalone_gs_renderer.py` 要**上传成文件**到 /content/（`from standalone_gs_renderer import` 用）；`Disconnect and delete runtime` 会清掉它要重传。
5. 完整可跑的 6-cell 序列在本次对话历史 + `scripts/colab/README.md`。

---

## 关键文件 / git / 闸

- **新增已 commit**（本 handoff 同批）：`scripts/colab/{standalone_gs_renderer.py, p2_render_arkit.py, README.md}` + 本文件。
- **数据（不入库）**：`assets/p2_arkit_render/`（80 帧，gitignore）。
- **待办文档**：`REVIEW_NOTES.md` 需补 P2 渲染审查记录（坐标转换逐行核对 GaussGym 一致、standalone 抽取忠实、c2ws 去 quat 往返的 fix、Colab 环境坑）——Claude 之前承诺补但上下文满了未补，新窗口补。
- **GPU 实例**：`i-06966fdfc4eefd3ea`(eu-north-1) 仍 STOPPED，本轮 $0 AWS。
- **付费闸**：P3 三路 RL 训练才需付费 GPU + 用户两道确认（§9）；P2 后半全本机免费，无需闸。
- **HF token**：P2 渲染用过（在 Colab login，未经聊天）；用户已被提醒泄露的旧 key 要轮换。

---

## 新窗口第一步

1. 读本文件 + 记忆 `stage2-execute-review-done` + `REVIEW_NOTES.md`/`EXPERIMENT_LOG.md` 的 2026-06-01 条目。
2. 确认 `assets/p2_arkit_render/` 有 80 帧（`ls assets/p2_arkit_render/rgb | wc -l` = 80）。
3. source-ground `semcost_nav/semantic/dino_cost.py` 现状 + `docs/D0_3_scene_hazard.md §3` DINO 方法。
4. 派 Codex 写 DINO cost map 本机 pipeline（读 assets 渲染帧 → cost map → parity → 四联图），Claude 审。
5. 先在含地毯帧（如 frame 60 对应的 rgb/00xx.png）上验证 cost map 能高亮地毯，再算 IoU。

*本文件由 Claude 在 P2 渲染成功的干净切点撰写，用于新窗口接续。*
