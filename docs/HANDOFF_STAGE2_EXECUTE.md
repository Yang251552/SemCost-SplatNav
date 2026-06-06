# HANDOFF — Stage 2 EXECUTE：Codex 主写真代码 → Claude source-grounded 审查

> 这份工单定义**实现阶段的协作翻转**：Codex 拥有 workspace、主写 Stage-2 真代码；
> 写完翻 `next-actor` 给 Claude，Claude 做**贴源码**审查。对应 teammate 状态机 `EXECUTE → REVIEW`。
> 工作目录：`/Users/yangchenghan/Downloads/SP申请3/SemCost-SplatNav`
> 上游：`docs/AWS_FLAGSHIP_PLAN.md`(总plan)、`docs/D0_SUMMARY.md`(已定决策)、`docs/D0_1_gaussgym_deps.md`(依赖真相)、`docs/D0_3_scene_hazard.md`(场景/hazard/DINO法)。

---

## 进度快照（2026-06-01，新窗口先读这段）

**走到哪了：** Stage-1 ✅ → D0 全免费前置 ✅ → D1 起机器+链路验证 ✅（含 Step C）→ **现在站在「写 Stage-2 真代码」起步线**。

- **D1 GPU 机器：已 STOPPED，不烧钱。** `i-06966fdfc4eefd3ea`，g5.2xlarge，**eu-north-1**，**on-demand ≈\$1.4/hr**（spot 配额没批，卡 0 → \$50≈35h，比原计划紧）。续用：`aws ec2 start-instances --instance-ids i-06966fdfc4eefd3ea --region eu-north-1`（会拿新 IP，需对 SG `sg-0ea8c67a94d49d1b8` 重放行你的 IP）。
- **Step C 已过：** 无人值守链路全通（三路占位训练→S3 同步 9 对象→自动关机→停机后仍能从 S3 读回日志）。证明的是「壳」能用，**不是任何科研结论**——三路目前还是 `echo+sleep` 占位。
- **repo 缺口已补：** commit `2e9bf7d` 已 push 到 GitHub `Yang251552/SemCost-SplatNav` main（scripts/aws/* + docs/* + handoffs）。无密钥/账号ID/桶名/实例ID（已扫描，handoff 用占位符）。→ 开机自启可从 GitHub clone，systemd 续跑前提满足。
- **下一步 = 本文件 §2/§3：** Codex 主写 Stage-2 真代码骨架（**本机/CPU dry-run，别动那台已停的 GPU**）→ 翻 Claude 贴源码审。
- **仍缺 / 闸：** Stage-2 真代码（train_gaussgym/evaluate/DINO cost map/configs/flagship.yaml）一行没写；`docs/aws/gauss_gym_env.yml` 已在本地；Step D 完整 GaussGym 要 **HF_TOKEN（用户到 Step D 才给）** + 6h timebox + 单开专注会话（付费 GPU 闸2，启动前必须回到用户确认）。
- **环境坑：** AMI 自带 torch2.11/py3.13/cu130 在 venv `/opt/pytorch`，与 GaussGym 锁定的 py3.8/torch2.4.1/cu121 **不兼容** → GaussGym 必须装自己的 conda（`setup_dev.sh`），别混用 AMI venv。

> 详细活状态见记忆 `aws-d1-launch-state`；本快照与之对齐，冲突以记忆为准（记忆更新更频繁）。

---

## 0. 为什么是这个流程（读一次就够）

- 全局 CLAUDE.md：**Claude 不负责大段实现代码**；Codex 负责编码/重构/测试/调试。
- 之前 D0 是 PLAN 阶段（决策/调研），所以 Claude 主导——那是**阶段**决定的，不是规则。
- 现在要写 Stage-2 真代码（RL/渲染/DINO），进入 **EXECUTE**：**Codex 主导，Claude 退到 REVIEW**。
- **铁律**：Claude 的 REVIEW 必须 **source-grounded**（读真源码 / 真跑），不是"读起来对"。
  证据：D0 里 Codex 首版有 2 个会烧钱的 P1（误装 gsplat、关机阈值用月累计），是 Claude **读了 clone 的 GaussGym 源码**才拦下的。审查若退化成 vibes，翻转反而更危险。

---

## 1. 活状态（必须先知道，否则会做错/烧钱）

> 来源：记忆 `aws-d1-launch-state`。执行前用 `aws`/SSH 复核，别全信本节。

- **GPU 机器已在跑、在烧钱**：`i-06966fdfc4eefd3ea`，g5.2xlarge，**on-demand ≈ \$1.4/hr**（spot 配额没批，卡 0）。**\$50 预算 ≈ 35 小时**——比原计划紧得多。不写代码时 **stop 实例**。
- 区域 **eu-north-1**（不是 handoff 写的 us-east-1）。实例在 AZ 1b。
- AMI 自带 **PyTorch 2.11 / py3.13 / cu130** 在 venv `/opt/pytorch`（`source /opt/pytorch/bin/activate`），**无 conda**。
  ⚠️ 这与 GaussGym 锁定的 **py3.8 / torch2.4.1 / cu121** **不兼容**——GaussGym 必须装在**自己的 conda env**（`bash setup_dev.sh` → `~/.gauss_gym_deps`），不要用 AMI 的 venv 跑 GaussGym。
- 预算 watchdog 活着（\$45 或 90min GPU 空闲自动关机）。仓库已 clone 到实例 `~/SemCost-SplatNav`。
- **缺口**：`docs/aws/gauss_gym_env.yml` 仍未在仓库（D2 要补/核）；`HF_TOKEN` 用户还没给（下 ARKit 场景必需）；Step C 轻量链路、Step D GaussGym 都没做。
- 账号用 **root 凭证**（已标记风险，未修）；wandb key 曾贴进聊天（用户已被提醒事后轮换）。

---

## 2. EXECUTE 要交付什么（Stage-2 真代码，对齐 CLAUDE.md Required Artifacts）

Codex 主写以下文件（**先写代码骨架 + 能在 CPU/小规模 dry-run 的版本，付费 GPU 全量跑前要过 Claude 审查 + 用户闸**）：

1. `scripts/train_gaussgym.py` — 三路统一训练入口，`--obs {depth,rgb,rgb_sem} --config configs/flagship.yaml`。
   - 三路**只差 `obs_keys`**；同 net/reward/seed/steps。
   - 启动时打印 **config hash**；三路 hash 除 obs_keys 外必须一致（fairness 断言）。
2. `scripts/evaluate_gaussgym.py` — `--all-modes --seeds 3`，产 `results/eval_{depth,rgb,rgb_sem}.json`。
3. `configs/flagship.yaml` — 基于 GaussGym `a1_vision`，override：场景=ARKit（`terrain.scenes.arkit.split_frac=1.0`，其余 0）、`num_envs` 保守 512、加 semantic_cost 通道与 reward 项。
4. **post-hoc 语义 cost map 模块** — RGB→DINOv2 ViT-S→PCA64(冻结)→坏原型 cosine→软 cost map（方法见 `D0_3`）。含 `results/parity_report.json`（cost vs 人标 hazard，IoU≥0.5）。
5. `docs/aws/gauss_gym_env.yml` — 补上 D0.2 版本锁文件（当前仓库缺）。

**公平性/诚实性硬约束**（来自 CLAUDE.md，违反即 P0）：
- 三路 config 除 `obs_keys` 外 diff 必须为空（config-hash 校验 + 运行时断言）。
- semantic_cost **不得编码目标位置**（channel-ablation 单测）。
- headline 对比 = **RGB+semantic vs RGB**（depth-only 只为完整性报，不当主张）。
- hazard 若在 depth 通道里可区分（depth-only 自己就拿到近零 bad_region_time）→ **停，重选场景/hazard，不许伪造 IoU**。

---

## 3. 协作接线（怎么"从 Codex 开始、翻给 Claude 审"）

> 你（用户）只跟 Claude 对话；Codex 是 `codex_workflow.py` 派发的 CLI，不能直接跟你说话。
> 所以"第一脚派发"由 Claude 做，**实现主导权**交给 Codex，**审查**翻回 Claude。

**EXECUTE（Codex 主写）** — Claude 发起一次：
```bash
python3.12 ~/.claude/scripts/codex_workflow.py max \
  --task "按 docs/HANDOFF_STAGE2_EXECUTE.md §2 实现 Stage-2 真代码（先骨架+CPU dry-run 版，不跑付费 GPU 全量）" \
  --workspace "/Users/yangchenghan/Downloads/SP申请3/SemCost-SplatNav" --zh
```
- Codex 拥有 workspace，主写代码 + 自测（CPU dry-run / 单测）。
- 写完产出 `===HANDOFF===` 块，翻 `next-actor` → Claude。

**REVIEW（Claude 贴源码审）** — 不可简化为"读一眼"：
- [ ] 对照 clone 的 GaussGym 真源码（`/tmp/gg` 或重新 clone）核 API/命令/config 字段**真实存在**，不是猜的。
- [ ] 跑 fairness：三路 config-hash 校验、channel-ablation 单测**真的执行并通过**。
- [ ] 版本：py/torch/gsplat/IsaacGym 与 `D0_1` 锁定一致；GaussGym 走自己的 conda，不混 AMI venv。
- [ ] 找"看起来对实则猜"的点（D0 教训）：库名、CLI flag、文件路径、reward 形状。
- [ ] P0/P1 必修后再翻回 Codex；修完**重跑相关测试**才算过。
- 结果写 `REVIEW_NOTES.md`；`codex_workflow.py` 工具若再误报 handoff，从 session 日志捞回并记降级。

**付费 GPU 闸**：全量三路过夜训练属 P3，**启动前必须回到用户确认**（机器在 on-demand 烧钱，\$50≈35h）。Codex/Claude 都不能自己决定开长任务。

---

## 4. 完成 / 返回条件

- **EXECUTE 完成**：§2 文件骨架齐、CPU/小规模 dry-run 通过、自测绿、handoff 翻给 Claude。
- **REVIEW 完成**：贴源码审过、P0/P1 清零、fairness 单测通过、记录入 REVIEW_NOTES。
- **任一不确定是否花钱 / 逼近 \$45 软闸 / hazard 在 depth 可见** → 停，回报用户。
- 始终：不写代码就 stop 实例省钱；不要把完整日志灌回主上下文。

---

## 5. 给用户的回报模板（Claude 汇总，简短）

```markdown
### Stage-2 EXECUTE/REVIEW 回报
- 阶段: EXECUTE(Codex主写) / REVIEW(Claude审) / 待用户付费闸
- 新增/改动文件: <list>
- fairness: config-hash[✓/✗] channel-ablation单测[✓/✗] 三路仅obs_keys差异[✓/✗]
- source-grounded 审查发现: P0<n> P1<n>（列关键项）/ 已修[✓/✗] 重测[✓/✗]
- 版本一致(py3.8/torch2.4.1/gsplat1.5.3/IsaacGym P4): [✓/✗]
- GPU 机器: <running/stopped> / 本次累计花费 $<x>
- 阻塞 / 需用户决定: <...>
- 下一步: <...>
```

---

*本文件由 Claude 撰写（定义的是 Claude 自己的审查职责 + 协议接线，故 Claude 主笔）；可让 Codex 反向挑刺后定稿。与活状态 `aws-d1-launch-state` 对齐。*
