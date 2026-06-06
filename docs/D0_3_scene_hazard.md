# D0.3 — HF 场景 + hazard 物体候选（2026-05-31）

来源：3 个并行 subagent（2× 场景调研 Explore + 1× DINO 方法 general-purpose）+ 读 cloned repo `/tmp/gg`。
诚实声明：**未做像素级真验**（本机无 GPU、数据需 HF 登录）；以下是调研+推理排序，真验在 P2 GPU 阶段。

## 0. gating 实测（真实仓库名）

| repo | HTTP | 状态 |
|---|---|---|
| `escontra/gauss_gym_data` | 200 | 公开（下载需 `huggingface-cli login` token） |
| `escontra/gauss_gym_arkit` | 200 | 公开（同上） |
| `escontra/veo_scenes` | 200 | 公开 |
| `leggedrobotics/grand_tour_dataset` | 200 | 公开；大数据集，**可能需一次性同意条款** |

结论：没有硬访问墙，`huggingface-cli login` 一个 token 即可。

## 1. 关键发现：默认场景不能用

GaussGym 默认开的 `gauss_gym_data`(iphone_data) **以楼梯/台阶为主**——几何信息恰恰是任务核心，
和我们"hazard 在几何上不可区分"的前提**直接冲突**。→ 必须 override 场景（改 `terrain.scenes` split_frac）。

## 2. hazard 候选三选一（核心权衡）

三个 agent 在"哪个 hazard"上各有侧重，归纳成一个清晰权衡：

| 候选 | 几何齐平度（depth 看不见？） | DINO 可分度 | 数据可得性 | 综合 |
|---|---|---|---|---|
| **A. 室内地毯/地垫**（ARKit 客厅/卧室） | 厚 5–15mm，对腿足≈不可见（需 P2 验证） | **最强**（织物 vs 木/瓷砖，DINO 材质信号最干净；2/3 agent 都排第一） | ARKit 房间里地毯**很常见** | **DINO 最稳、叙事最直观** |
| **B. 室内地面材质过渡**（瓷砖↔木地板/油毡，或地面涂装/标线） | **0mm 完全齐平**（和 P1 "无几何痕迹"最像） | 良好但略难（都是"地面"） | ARKit 厨房/卫生间/门厅常见 | **实验最严谨（可证 depth 不变）** |
| **C. 户外水洼/积水**（GrandTour 步道） | **0mm 完全齐平** | **不稳**：高光随视角/光照变，DINO 特征跨帧不一致（方法 agent 警告） | GrandTour 下载重 + 可能要同意条款 | flushness 满分但 DINO 不稳、下载重 |

**我的技术负责人推荐**：**A（室内地毯）为主**——DINO 可分度是这条 pipeline 的命门，地毯在这点上 2/3 agent 一致最优、且数据最易得、叙事最好懂（"机器人几何上感觉不到地毯边缘，但视觉上看得出这是另一种表面"）。
若你更看重"像 P1 那样可证明 depth 完全不变"的严谨性，选 **B**。C 暂作 future work/备选。

## 3. DINO 方法定稿（方法 agent，采纳）

- 管线：渲染 RGB → resize **224×224**（patch=14 → 16×16 grid）→ DINOv2 ViT-S patch features → **L2 normalize** → PCA→64d（**在多帧/多场景上拟合一次、冻结复用**）→ 与"坏原型"cosine → **软 cost**（`relu(cos−margin)`，非硬阈值）→ 双线性上采样成 per-pixel cost map。
- 坏原型：**few-shot 掩码特征平均**（3–5 帧手标 hazard 掩码 → 掩码内 patch 特征平均 → L2 normalize）。比 crop 平均更干净，比 CLIP/文本对齐更简单，~20 行代码。
- 公平性（不泄露几何/目标）：hazard 渲成纯反照率/材质变化、**零高度**；自动测 `depth_with_hazard ≈ depth_without`；cost map 只编码 hazard 强度、与目标位置独立 → 保证 depth-only 基线对 hazard 真盲。

## 4. P2 GPU 真验步骤（落地清单）

1. `huggingface-cli login`；override 场景（如 `terrain.scenes.arkit.split_frac=1.0`，其余 0）；载 1–2 个含地毯场景。
2. 渲一条跨 hazard 边界的轨迹，dump RGB+depth ≥50 帧。
3. **depth 不变性检验**：hazard 区 vs 地面在 depth/heightmap 上统计无差异（无边缘）→ 证伪条件 (a)。
4. **DINO 可分检验**：cost map 与手标掩码 IoU ≥0.5、地面低响应 → 证条件 (b)。
5. 若地毯太弱 → 退 B（材质过渡，掩码更脆）；再不行才 C。

## 5. 已拍板（2026-05-31）

- **机器人形态 = a1 四足狗**（GaussGym 官方验证过、最稳）。项目从 P1 轮式网格导航转为「a1 四足在真实房间运动 + 语义代价」。
- **hazard = 地毯/地垫**（候选 A）。理由见 §2：DINO 可分度最强、数据最易得、叙事最直观；方法 agent 的硬证据排除了水/湿面（VFM 失效模式）。
- 后续锁定：robot config = `gauss_gym/envs/a1/config_vision.yaml`；场景 override 为 `terrain.scenes.arkit.split_frac=1.0`（iphone_data/grand_tour=0），载含地毯的 ARKit 室内场景。
- 候选 B（材质过渡）作为地毯太弱时的回退，C（水洼）弃用。
