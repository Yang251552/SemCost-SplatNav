# D0 总结（全免费阶段完成）— 2026-05-31

AWS 稳健旗舰版的 Day-0 前置工作全部完成，**未花一分钱、未开任何云资源**。
下面是产出清单 + 唯一剩下的付费 gate + D1 开工清单。

## D0 五项产出

| # | 任务 | 状态 | 产物 |
|---|---|---|---|
| D0.1 | 查清 GaussGym 依赖栈 | ✅ | `docs/D0_1_gaussgym_deps.md` |
| D0.2 | 锁版本 / env.yaml | ✅ | `docs/aws/gauss_gym_env.yml` |
| D0.3 | 选场景 + hazard | ✅ | `docs/D0_3_scene_hazard.md` |
| D0.4 | 本机 P1 parity | ✅ | 见下 |
| D0.5 | 自动化脚本 | ✅ | `scripts/aws/*`（Claude 已审查 + 修 4 问题） |

## 关键结论

1. **依赖栈（权威，读 clone 的 repo 文件）**：IsaacGym **Preview 4 / 1.0rc4**（legacy，非 Isaac Lab）·
   Python 3.8 · CUDA 12.1 · torch 2.4.1 · gsplat 1.5.3(git pin) · 仓库自带 PPO。
   **最大利好**：IsaacGym Preview 4 实测可免登录下载（200/x-gzip）→ 安装风险从「高」降到「中」。
2. **形态已定**：a1 四足狗 + 地毯/地垫 hazard（室内 ARKit 场景，需 override 默认楼梯场景）。
   叙事 = 「a1 四足在含隐形地毯-hazard 的真实房间运动 + 语义代价」。
3. **DINO 方法已定**：RGB→224²→DINOv2 ViT-S→PCA64(冻结)→坏原型 cosine→软 cost map；
   地毯做成零高度纯材质保证 depth-only 真盲。已排除水/湿面（VFM 失效模式）。
4. **三路消融接口清晰**：GaussGym vision config 已有观测分组开关，三路 = 切 flag + 加一路 cost 通道，
   同 task/seed/steps，公平性靠 config-hash + 运行时断言。

## D0.4 — P1 parity sanity（本机 CPU，已验证）

- `tests/test_env_smoke.py` → **All smoke tests passed**。
- 依赖可导入：gymnasium 1.0.0 / sb3 2.4.1 / numpy 1.26.4 / torch 2.2.2（Python 3.12）。
- P1 基线锚点（200 paired seeds，作为旗舰版对照）：
  - depth-only：success 1.0 / collision 0 / **bad_region_time 1.92** / path 12.94 / return 0.858
  - depth+semantic：success 1.0 / collision 0 / **bad_region_time 0.825** / path 10.76 / return 0.996
  - → **−57% bad_region_time**，旗舰版要在真实场景 + 三路上复现/超越这个故事。

## D0.5 — AWS 自动化脚本（Claude 审查通过）

`scripts/aws/`：`run_flagship_overnight.sh`（三路顺跑+S3 周期同步+trap 退出必关机）、
`ec2_user_data.sh`（开机装环境+clone+S3 resume+systemd 启动，spot 被抢自动接力）、
`setup_budget_guard.sh`（\$50 Budgets 告警 + 会话增量成本 watchdog + GPU 空闲 watchdog 双关机）、`README.md`。

Claude 审查后修正 4 处（详见 REVIEW_NOTES）：
- 降级环境不再误 pip 装 gsplat（git pin / Docker 才对）。
- watchdog 关机阈值改为**本次会话增量成本**（baseline 机制），严格对应"单次 ≤\$50"。
- README 补 EBS `DeleteOnTermination=false` 续跑说明。
- 占位注释换成真实 `gauss_train --task=a1_vision` / `gauss_play` 命令形式。

> 真实训练命令仍是占位（接 GaussGym 后替换 `train_one`/`evaluate_all`），脚本不会误启动长任务。

## 唯一剩下的付费 gate（要你做）

- **g5 quota 工单**：你已选择自己去 AWS Console 提（首次用 g 系列常需，可能 1–2 天异步）。
  提的是 **"Running On-Demand G and VT instances"** 的 vCPU 配额（g5.2xlarge = 8 vCPU；建议申请到 ≥8，spot 另有 "All G and VT Spot Instance Requests" 配额也要 ≥8）。区域建议 us-east-1 或 us-west-2。
- 批下来后告诉我区域，我们进 **D1**。

## D1 开工清单（上 GPU 后，按序）

1. 起第一台 g5.2xlarge spot（用 `ec2_user_data.sh`，先 `USE_LIGHTWEIGHT_ENV=1` 验证关机/同步链路，几分钱）。
2. 跑 `setup_budget_guard.sh` 建预算 + watchdog。
3. 切完整路径：GaussGym `bash setup_dev.sh`（timebox 6h）→ smoke `gauss_train --task=a1_vision --env.num_envs 512`。
4. P2 渲染 + post-hoc 语义：override arkit 场景、渲一条跨地毯轨迹、跑 DINO cost map、四联图、parity_report（depth 不变性 + IoU≥0.5）。**Gate P2**。
5. 接三路 obs + reward，启动过夜自动化训练。**Gate P3**。

## 风险位（不变）

GaussGym/IsaacGym 在 A10G 上的真实安装仍需 D1 在付费 GPU 上验证（6h timebox，超时→降级到自写轻量 gsplat env 跑三路，hero 图仍成立）。
