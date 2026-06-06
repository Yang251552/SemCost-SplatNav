# AWS 稳健旗舰版执行 plan — SemCost-SplatNav

> 范围：在已完成的 P1 低配版基础上，升级到**真实 gsplat 场景 + GaussGym 三路消融**，
> 语义走 **post-hoc DINOv2**。目标产出 = 给 RAI/ETH 申请用的 hero artifact：
> 三路并排 rollout 视频 + “time-in-bad-region” 学习曲线。
>
> ⚠️ 本文为**计划**，不含任何已执行的付费步骤。所有上 AWS 的动作前都有 §9 confirmation gate。
> 协作来源：Claude 主笔 + Codex `max` 独立产出一份平行 plan 做交叉验证（见 §10）。两份高度一致；
> 本文已合并 Codex 的具体验收阈值，并补上 Codex 漏掉的 3 个 P0。

---

## §0 假设与未决（执行前必须确认）

**已确认**
- 版本：稳健旗舰版（真实场景 + GaussGym 三路消融 + post-hoc 语义）。Hero 版（renderer 内部特征 splatting）= future work。
- 硬件：AWS g5.2xlarge = 1× NVIDIA A10G 24GB / 8 vCPU / 32GB RAM。

**已确认（2026-05-31，用户拍板）**
- 预算：**硬上限 ≤\$50**（旗舰版估 \$20–40，spot）。设 AWS Budgets \$50 告警 + \$45 自动停机硬闸。
- 时间：**deadline 紧，但每天可投入多、含夜间无人值守自动化**。→ 策略 = 白天最少人工、把训练塞进**过夜自动化 pipeline**（一条命令跑完三路 + 自动 checkpoint S3 + 自动关机）。
- 场景来源：**HuggingFace `escontra/gauss_gym_arkit`（室内 ARKit）**，override 默认楼梯场景（自采集仅后备）。
- hazard 语义：**地毯/地垫**（D0.3 三 agent 裁决：DINO 可分度最强、数据最易得；已排除水/湿面这一 VFM 失效模式）。
- **机器人形态：a1 四足狗**（GaussGym 官方验证过）。项目从 P1 轮式网格导航 → a1 四足在真实房间运动 + 语义代价。
- 本机为 Mac（无 NVIDIA 卡）→ 任何 `gsplat`/IsaacGym 渲染与训练都在云端或免费 Colab/Kaggle GPU 上跑。

**仍未决（不阻塞 D0）**
- W&B vs 纯本地 TensorBoard（默认纯本地，省事省钱）。
- 具体 HF 场景 + 具体 hazard 物体类：D0 由 Claude 挑 2–3 个候选、验证“几何不可区分但 DINO 可分”后再定。

---

## §1 目标与非目标

**目标**
- 环境从 11×11 网格 → **真实感 gsplat 场景**渲染的 RGB-D。
- 语义从“环境给的真值” → **估计值**：post-hoc 渲染 RGB 跑 DINOv2 → PCA → cosine cost map。
- GaussGym（依赖 IsaacGym）中跑**三路消融**：`depth-only` / `RGB` / `RGB+semantic`，同网络、同 reward、同 seed，**只改观测**。
- 产出 hero artifact + 可复现实验记录 + 申请向 README/报告。

**非目标**
- 不做 renderer 内部 VFM 特征 splatting（Hero 版）。
- 不声称“完整 GaussGym 集成 / 论文复现”；定位为轻量集成 demo。
- 不做大规模超参搜索、不追 SOTA。

---

## §2 现状基线（P1，已完成）

- SB3 PPO + 小 CNN（`SmallNavCNN`），11×11 walled arena，CPU。
- 两路 `depth` vs `depth+semantic`，`semantic_cost` = 环境给的危险区**真值**（非估计）。
- 结果：`bad_region_time` 1.92→0.83（**−57%**），success 100%、collision 0、路径更短。
- 价值：已证明“多一路语义信息能降低危险区暴露”——但**场景假、语义假、只两路**。旗舰版把这三样变真/升级。

---

## §3 总体架构与数据流

```
真实 gsplat 场景 (.ply/checkpoint)
        │  相机位姿(机器人) ──► gsplat 渲染 (CUDA, A10G)
        ▼
   RGB  +  Depth
   │         │
   │         └──────────────────────────►  obs: depth 通道
   │
   ├─► (post-hoc) DINOv2 ViT-S ─► PCA→~64d ─► 与“危险”原型 cosine ─► cost map ─► obs: semantic 通道
   │
   └──────────────────────────────────────►  obs: RGB 通道

obs 组合（三路消融，同 net / 同 reward / 同 seed / 同 steps）：
   A. depth-only      → [depth]
   B. RGB             → [rgb]
   C. RGB + semantic  → [rgb, cost_map]

  → 统一 PPO（GaussGym/IsaacGym 向量化环境，A10G 上 512→1024 envs）
  → 指标：bad_region_time / collision_rate / success_rate / path_length / return
```

**接口约定**：环境暴露统一 obs dict（`rgb|depth|cost_map`），PPO/网络三路共享，只切换 obs key 组合 → 公平性靠这一点保证。三路共享同一 config，仅 `obs_keys` 不同，**config hash 入库校验**（Codex 建议，采纳）。

---

## §4 分阶段执行计划

每阶段有**可验证产出**与**验收 gate**；未过 gate 不进下一阶段。

### P2 — 真实 gsplat 场景 + post-hoc 语义
**做什么**
- 选定/获取真实 gsplat 场景（决策②）。
- 写可控相机渲染封装：给位姿出 RGB-D。
- post-hoc 语义：RGB → DINOv2 ViT-S → PCA→~64d → 选定“危险”原型 → cosine → cost map。
- 避障可视化：RGB/depth/cost/avoid-mask 四联图 + 一条脚本轨迹。

**可验证产出**：`renders/`（RGB/depth/cost 三联图 ≥20 帧）、`cost_map_overlay.mp4`、`parity_report.json`。

**验收 gate**
- [ ] 渲染 RGB-D 正常，多视角一致。
- [ ] cost map 在“危险”语义区高响应：与人工标注 hazard **IoU ≥ 0.5**（Codex 阈值，采纳）。
- [ ] **修正**：`gsplat` 光栅化 CUDA 专用，**Mac 本机跑不了渲染**。DINO 部分可本机(MPS/CPU)，渲染 smoke 走免费 Colab/Kaggle GPU 或一小段廉价 spot（见 §5）。
- [ ] cost map 不编码目标位置（leakage guard，延续 P1）。

### P3 — GaussGym + IsaacGym 集成 + 三路消融（核心，付费 GPU）
**做什么**
- **D0 前置（免费）**：读 GaussGym README，确认依赖 **IsaacGym(legacy preview) 还是 Isaac Lab/Sim**（NVIDIA 已弃用 IsaacGym），锁定精确 CUDA/torch/gsplat 版本。参考基线 **CUDA 11.8 / torch 2.1 / 对应 gsplat wheel**，以 README 为准，用 conda `env.yaml` 固化。
- A10G 上装 GaussGym + 物理后端（timebox，见 §5）。
- 接 P2 的 RGB-D + cost map 为 obs 通道；加 semantic cost reward 项（与 P1 同形 `-0.1 × intensity`/步）。
- envs：512（保守）→ 视显存升 1024。
- 跑三路 A/B/C，**同 net/reward/seed/steps**，checkpoint 到 S3。

**可验证产出**：`runs/{depth,rgb,rgb_sem}/`（TensorBoard、ckpt、eval.json）、三路 `metrics.csv`。

**验收 gate**
- [ ] 三路除 `obs_keys` 外 config diff 为空（**config hash 校验**）。
- [ ] semantic 不泄露目标（通道 mask + channel-ablation 单测，Codex 建议，采纳）。
- [ ] 训练收敛、success 合理、指标可复现（固定 eval seed 集，沿用 P1 的 200 seeds）。
- [ ] 过夜训练能从 spot 中断 `--resume` 续跑（S3 checkpoint）。

### P4 — 结果 / 视频 / 曲线 / 报告 polish
**做什么**
- hero 视频：三路同地图并排 rollout `ablation_rollout.mp4`。
- 学习曲线：`time_in_bad_region.png` 三条线 + 指标柱状对比。
- 报告：`REPORT.md`、README 升级、devlog、STATUS/PROGRESS/SUMMARY。
- 诚实写 Limitations 与 future work（Hero 版 in-renderer splatting）。

**验收 gate**
- [ ] hero 视频 + 曲线生成、复现命令可跑。
- [ ] headline 结论挂在 `bad_region_time`（安全指标），核心对比 **RGB+semantic vs RGB**（见 §7）。
- [ ] 显著性：尽量 **≥3 seeds**（Codex 建议）；预算/时间不够时至少 paired-eval + 诚实标注单 seed。
- [ ] 申请向一句话 claim 有证据、无夸大。

---

## §5 风险与回退

| 风险 | 等级 | 缓解 / 回退 |
|---|---|---|
| **IsaacGym/GaussGym 在 A10G 装不上**（最大时间黑洞） | 高 | D0 先核实依赖栈(IsaacGym vs Isaac Lab)；用已知可用容器/AMI；**timebox 6h 付费 GPU**（Codex 值，采纳）；超时直接降级 → **自写轻量 gsplat batched env**（复用 P1 env 抽象）跑三路消融，hero 图仍成立 |
| `gsplat` 需 CUDA，Mac 跑不了 | 中 | P2 渲染走 Colab/Kaggle 免费 GPU 或短 spot；DINO 本机 |
| 显存 OOM | 中 | 降 env 数 / 分辨率(如 256×256) / fp16 |
| 真实场景 hazard 几何上可区分 / DINO 分不出 | 中 | 先选**几何不可区分、语义清晰**的 hazard（决策③）；选场景时验证 |
| g5 quota 需工单（1–2 天异步） | 中 | **D0 第一件事**提工单，并行等待 |
| spot 中断丢进度 | 中 | 每 N step checkpoint→S3，`--resume` 续跑 |
| 版本不兼容 | 高 | CUDA/torch/gsplat 全 pin，conda `env.yaml` 固化，版本写进 REPORT |
| 预算超支 | 低 | 闲时自动停机 + Budgets 告警 \$100/\$150 |

---

## §6 AWS 工程要点

- **实例**：g5.2xlarge spot（us-east-1 / us-west-2 比价）；Deep Learning AMI（Ubuntu 22.04，预装 CUDA）。
- **存储**：gp3 EBS ~100GB；S3 桶存场景/ckpt/视频，训练每 N step 同步。
- **自动停机**：训练结束/异常 → `shutdown -h +5`；CloudWatch 闲置告警（最大省钱项）。
- **持久化**：tmux 持久会话 + SSH/VSCode remote；spot 中断后新实例拉 S3 续跑。
- **省钱**：不训练就 `stop` 实例；COLMAP 等 CPU 活放本机/廉价 CPU 实例。
- **护栏**：AWS Budgets \$150 阈值 80%/100% 告警。
- **版本对齐**：以 GaussGym README 锁定版本为准（D0），全程 pin，记录 `nvidia-smi`/torch/gsplat 到 REPORT。

---

## §7 公平性与研究诚实性

- **三路一致**：reward/seed/steps/eval seed 集完全相同，唯一差异是 obs 组合（config hash + 运行时断言强制）。
- **headline = RGB+semantic vs RGB**：RGB 已隐含大量语义，“depth-only 输给 RGB+semantic”几乎必然、说服力弱。真正诚实且有力的主张是：**已经能看彩色图(RGB)时，显式语义 cost 通道能否进一步降 bad_region_time**。三路都报，但核心结论挂这一对。
- **hazard 语义定义（已确认=某类物体）**：真实场景 hazard = DINO 与某“坏”物体原型 cosine 匹配区域（地毯/水渍/杂物等）。该类物体必须**几何上与地面不可区分**（不挡路、深度无痕，延续 P1 思想），否则 depth-only 也能看出来、实验失去意义。D0 选场景时验证：depth 通道里该物体不可见、DINO cosine 图里它高亮。
- **不泄露目标**：cost map 只编码 hazard 强度，绝不含目标位置（leakage guard + channel-ablation 单测）。
- 主张以 `bad_region_time` / `collision_rate` 为准，return 仅辅助；结果弱则诚实调任务设计，不操纵指标。

---

## §8 时间线 / 成本 / 交付物

**时间线（deadline 紧 → 压缩 active 人工，重过夜自动化）**
- **D0**（免费，今天就能做）：提 g5 quota 工单；读 GaussGym 依赖、锁版本写 `env.yaml`；Claude 挑 2–3 个 HF 场景 + hazard 物体类候选并验证“几何不可区分/DINO 可分”；本机过 P1 parity sanity；**写好过夜自动化 pipeline 脚本**（见下）。
- **D1**（少量付费 GPU）：P2 渲染封装 + post-hoc 语义 + 四联图 + `parity_report.json`（渲染走 Colab/短 spot）；P3 代码接好、CPU/本机 dry-run。**Gate P2**
- **D1 夜 / D2**（付费 GPU，timebox 6h）：GaussGym/IsaacGym 安装 + smoke；装不通 → 立即降级（§5）。当晚启动**自动化过夜训练**。**Gate P3**
- **过夜（无人值守）**：一条命令顺跑三路 `depth→rgb→rgb_sem`，每 N step checkpoint→S3，全部跑完自动 `aws s3 sync` + `shutdown`。利用“可大量过夜自动化”把人工压到最少。
- **D2/D3 白天**（本地，免 GPU）：从 S3 拉结果 → hero 视频 + 曲线 + 报告。**Gate P4**

**过夜自动化 pipeline（关键适配“时间紧+可自动化”）**
```bash
# run_flagship_overnight.sh （D0 写好，D1 夜一键启动）
set -euo pipefail
for mode in depth rgb rgb_sem; do
  python scripts/train_gaussgym.py --obs $mode --config configs/flagship.yaml \
    --resume-from-s3 --checkpoint-every 5000 --s3 s3://<bucket>/runs/$mode || true
done
python scripts/evaluate_gaussgym.py --all-modes --seeds 2 --s3 s3://<bucket>/runs
aws s3 sync runs/ s3://<bucket>/runs/   # 最终回传
sudo shutdown -h +2                      # 跑完即关机，省钱硬闸
```
- 配 EC2 user-data 或 systemd，开机自动 `--resume-from-s3` 续跑 → spot 中断也能无人值守接力。

**GPU 小时 / 成本（spot ≈ \$0.45/hr，硬上限 \$50）**
- P2 渲染 smoke ~2–4 GPU-hr；P3 安装+调试 ~4–8 GPU-hr（timebox 严格）；三路过夜训练 ~25–45 GPU-hr。
- 合计 ~30–55 GPU-hr ≈ **\$14–25 spot**，含 buffer **< \$50**。
- **省钱硬措施**：纯 spot；不训练立即 `stop`；跑完自动 `shutdown`；AWS Budgets \$45/\$50 两档告警；若逼近 \$45 自动停机。on-demand 不用（除非最终不可中断跑，且预算允许）。
- 若要进一步压成本：P4 把 seeds 从 3 降到 2（诚实标注），或缩训练步数——以保住三路 hero 图为最高优先。

**交付物清单**
- `figures/` / `renders/`：三路并排 rollout 视频、time-in-bad-region 曲线、指标对比图、场景/depth/cost/avoid-mask 四联图。
- `results/` `runs/`：三路 eval/train JSON + `metrics.csv`。
- `checkpoints/`：三路策略。
- 文档：README（升级）、`REPORT.md`、EXPERIMENT_LOG、devlog、STATUS/PROGRESS/SUMMARY、`env.yaml`、AWS runbook。
- 一句话申请 claim（有证据）。

**Git milestone（本地，配合 [../CLAUDE.md](../CLAUDE.md) Git Rules + Automation Safety Git Rules）**

每阶段开始前打 `wip: before <phase>` 空 commit 做 safety point；过 gate 后打 stable commit 并把 hash 写入 `EXPERIMENT_LOG.md`。

| 阶段 | safety point (前) | stable commit (后) | 备注 |
|---|---|---|---|
| D0（免费） | `wip: before flagship D0` | `chore: lock flagship env + scene candidates` | 含 `env.yaml`、HF 场景候选 report、过夜脚本草稿、AWS quota 工单截图 |
| P2 渲染 + post-hoc 语义 | `wip: before P2 gsplat render` | `feat: gsplat render + DINO cost map (parity IoU=X)` | 过 gate 后；IoU 数字写进 commit message；含 `parity_report.json` |
| P3 GaussGym 集成 | `wip: before P3 gaussgym install` | `feat: gaussgym 3-arm training scaffold (config hash=X)` | 装通后；含 config-hash 校验 + channel-ablation 单测 |
| P3 三路过夜训练完成 | — | `exp: 3-arm flagship results (seed 0..N)` | 从 S3 拉回 metrics/eval 后；**不直接 commit ckpt** |
| P4 hero artifacts | `wip: before P4 polish` | `docs: flagship report + hero video + curves` | README/REPORT/figures |

**两个坑（提前规避）**：
1. 过夜 pipeline 跑完只做 `aws s3 sync` + `shutdown`，**不自动 commit**。D2/D3 白天从 S3 拉回结果后必须手动走 stable commit 流程，hash 进 `EXPERIMENT_LOG.md`。
2. ckpt 文件大、走 S3 + `.gitignore`，**不进 git**；只 commit `metrics.csv` / `eval.json` / `config.yaml` / `parity_report.json` 这类小而可复现的产物。D0 第一件事检 `.gitignore` 是否覆盖 `runs/`、`checkpoints/`、`*.ckpt`、`*.pt`、`renders/*.mp4` 等大文件模式。

---

## §9 Confirmation Gate（任何付费步骤前必须确认）

**已确认（2026-05-31）**
- [x] 预算 ≤\$50、spot 优先、deadline 紧+重过夜自动化。
- [x] 场景来源 = HF 预训练 gsplat。
- [x] hazard = 某类物体（地毯/水渍/杂物，D0 定具体类）。

**仍待确认（两道闸，阻塞付费 GPU）**
- [ ] **第一道闸（上 P2 短 GPU 前）**：D0 全免费产出（场景候选 + 版本锁 + parity 计划 + 过夜脚本草稿 + `.gitignore` 检查 + D0 stable commit 已落）出来后，由你看一眼候选场景/hazard 是否 OK，再开 spot 做 P2 渲染 smoke。
- [ ] **第二道闸（上 P3 过夜训练 spot 前）**：P2 stable commit 已落（含 `parity_report.json`、IoU≥0.5），三路 config-hash 校验 + channel-ablation 单测在本地/CPU dry-run 通过，过夜脚本（`run_flagship_overnight.sh` + EC2 user-data `--resume-from-s3`）已验证可启动，S3 桶可写、AWS Budgets \$45/\$50 告警已配。

> 现在即可开始 **D0（全部免费、无付费动作）**：提 quota 工单 + 锁版本 + 选 HF 场景/hazard 候选 + 本机 P1 parity + 写过夜自动化脚本。
> D0 完成后我把候选结果给你看 → 你点头 → 才开第一台 spot。

---

## §10 Codex 交叉验证记录

- 动作：`codex_workflow.py max --intent research`（只读，独立产出平行 plan）。
- 状态：Codex 完成会话（`019e7d1c…`）并产出**完整 plan + ===HANDOFF=== 块**，但 workflow 的 handoff 抽取器报 “not found”（工具解析 glitch）。Claude 已从 session 日志**恢复全文**并合并，非纯降级。
- 两份一致点：三阶段拆解、IsaacGym 为最大风险 + 降级到自写 batched env、spot+S3 checkpoint+自动停机、三路同 config 仅观测不同、主张挂 bad_region_time、成本远低于 \$150。
- 采纳自 Codex 的强化：IoU≥0.5 的 cost-map parity 阈值、config-hash 公平性校验、channel-ablation 单测、CUDA11.8/torch2.1 版本基线、IsaacGym timebox 6h、P4 ≥3 seeds、`env.yaml` 固化。
- Claude 补上 Codex 漏掉的 P0：① `gsplat` 需 CUDA、Mac 跑不了 P2 渲染（Codex 误判“P2 本地 0 付费”）；② IsaacGym 已被 NVIDIA 弃用、需 D0 核实 vs Isaac Lab；③ g5 quota 工单需 D0 最先发起。
