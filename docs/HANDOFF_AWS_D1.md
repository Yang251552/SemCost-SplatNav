# AWS D1 运维 Handoff — 起第一台 g5.2xlarge spot

> 读者：零上下文子 agent。
> 目标：只负责把 D1 第一台 AWS GPU 机器起起来、装好环境、验证省钱与续跑链路，并把简短状态回报给主 agent。
> 相关仓库文件：`docs/AWS_FLAGSHIP_PLAN.md`、`docs/D0_SUMMARY.md`、`docs/D0_1_gaussgym_deps.md`、`docs/D0_3_scene_hazard.md`、`scripts/aws/README.md`、`scripts/aws/ec2_user_data.sh`、`scripts/aws/run_flagship_overnight.sh`、`scripts/aws/setup_budget_guard.sh`。若 `docs/aws/gauss_gym_env.yml` 存在，也要核验；若不存在，按阻塞点回报，不要擅自补文件。

## 0. 你的唯一任务 / 不该做的事

唯一任务：

- 把 D1 第一台 `g5.2xlarge` spot 实例起起来。
- 装好环境。
- 验证省钱与续跑链路。
- 把简短状态回报给主 agent。

不负责：

- 科研内容。
- 训练调参。
- 写论文。
- 改 RL 代码。
- 设计新实验。
- 擅自扩展预算、区域、实例规格或训练范围。

## 1. 硬性护栏（最重要，放最前）

- 这是付费动作。预算硬上限 `50 USD`，软闸 `45 USD`，spot 优先，跑完自动关机。
- 两道确认闸来自 `docs/AWS_FLAGSHIP_PLAN.md` §9：
  - 闸1：起任何付费实例前停下，向用户确认前置清单全部就绪。
  - 闸2：从 `USE_LIGHTWEIGHT_ENV=1` 廉价验证切到完整 GaussGym 付费训练前再停一次。
- 绝不把 AWS 密钥、真实桶名、HF token 写进仓库文件或 commit；用环境变量、实例环境、启动模板或 SSM Parameter Store。
- 子 agent 没有用户 AWS 凭证，不能替用户 `aws configure`、过 MFA、在 Console 点批准。这些只回报给用户做。
- 回报要简短：实例 id、区域、S3 路径、本次花费、状态、阻塞点。不要把完整日志灌回主上下文。
- 不确定就停，宁可问，不要烧钱。

## 2. 背景（一段话）

SemCost-SplatNav 旗舰版 Stage2：`a1` 四足 + 地毯 hazard + 室内 ARKit gsplat 场景 + GaussGym 三路消融（`depth` / `rgb` / `rgb_sem`）。D0 全免费前置已完成，详见 `docs/D0_SUMMARY.md`。本 handoff 只覆盖 D1 的“把机器起起来”。

## 3. 前置核验清单（闸1，逐条 check，缺一不可）

逐条执行。任一缺失就停下问用户，不要硬上。

### 3.1 g5 quota 已批

必须确认目标区域为 `us-east-1` 或 `us-west-2`，且两个 quota 都至少 `8` vCPU：

- `Running On-Demand G and VT instances >= 8 vCPU`
- `All G and VT Spot Instance Requests >= 8 vCPU`

验证命令：

```bash
export AWS_REGION="<us-east-1-or-us-west-2>"

aws service-quotas list-service-quotas \
  --region "$AWS_REGION" \
  --service-code ec2 \
  --query "Quotas[?contains(QuotaName, 'Running On-Demand G and VT instances') || contains(QuotaName, 'All G and VT Spot Instance Requests')].[QuotaName,Value,QuotaCode]" \
  --output table
```

若命令查不到或值小于 `8`，让用户到 AWS Console 申请/确认。不要自己替用户提交 Console/MFA 操作。

### 3.2 本机 aws cli 已配置

验证命令：

```bash
aws sts get-caller-identity
aws configure get region
```

要求：

- `aws sts get-caller-identity` 成功返回 `Account` / `Arn` / `UserId`。
- CLI 区域与 D1 区域一致，或命令显式带 `--region "$AWS_REGION"`。

如果需要 `aws configure`、MFA、SSO 登录或权限批准，停下请用户处理。

### 3.3 S3 桶、EC2 key pair、subnet、security group、AMI 就绪

需要用户提供：

- 实验 S3 桶 URI，例如 `s3://<bucket>/<prefix>`。
- EC2 key pair 名称。
- subnet id。
- security group id。
- AMI id，优先 Ubuntu 22.04 GPU/NVIDIA driver AMI 或 Deep Learning AMI GPU Ubuntu。
- IAM instance profile 名称。

验证命令：

```bash
export AWS_REGION="<us-east-1-or-us-west-2>"
export S3_URI="s3://<bucket>/<prefix>"
export EC2_KEY_NAME="<key-pair-name>"
export SUBNET_ID="<subnet-id>"
export SECURITY_GROUP_ID="<sg-id>"
export AMI_ID="<ami-id>"
export IAM_INSTANCE_PROFILE_NAME="<instance-profile-name>"

aws s3 ls "$S3_URI" --region "$AWS_REGION"

aws ec2 describe-key-pairs \
  --region "$AWS_REGION" \
  --key-names "$EC2_KEY_NAME"

aws ec2 describe-subnets \
  --region "$AWS_REGION" \
  --subnet-ids "$SUBNET_ID" \
  --query "Subnets[0].[SubnetId,VpcId,AvailabilityZone,MapPublicIpOnLaunch]" \
  --output table

aws ec2 describe-security-groups \
  --region "$AWS_REGION" \
  --group-ids "$SECURITY_GROUP_ID" \
  --query "SecurityGroups[0].[GroupId,VpcId,GroupName]" \
  --output table

aws ec2 describe-images \
  --region "$AWS_REGION" \
  --image-ids "$AMI_ID" \
  --query "Images[0].[ImageId,Name,State,CreationDate]" \
  --output table

aws iam get-instance-profile \
  --instance-profile-name "$IAM_INSTANCE_PROFILE_NAME" \
  --query "InstanceProfile.[InstanceProfileName,Arn]" \
  --output table
```

如果桶未建，用户可执行：

```bash
aws s3 mb "s3://<bucket>" --region "$AWS_REGION"
```

不要把真实桶名提交进仓库。

### 3.4 IAM instance profile 权限就绪

instance profile 至少需要：

- 对实验桶的 `s3:GetObject`、`s3:PutObject`、`s3:ListBucket`。
- `ce:GetCostAndUsage`。
- AWS Budgets 创建/更新/查询预算所需权限，例如 `budgets:CreateBudget`、`budgets:UpdateBudget`、`budgets:DescribeBudget`。
- `sts:GetCallerIdentity`。

验证命令：

```bash
aws iam get-instance-profile \
  --instance-profile-name "$IAM_INSTANCE_PROFILE_NAME" \
  --query "InstanceProfile.Roles[].RoleName" \
  --output text

export IAM_ROLE_NAME="<role-name-from-command-above>"

aws iam list-attached-role-policies \
  --role-name "$IAM_ROLE_NAME" \
  --output table

aws iam list-role-policies \
  --role-name "$IAM_ROLE_NAME" \
  --output table
```

权限是否足够通常要结合 policy 内容人工确认。若看不到 policy 或权限不足，停下问用户。

### 3.5 HF token 准备好

用途：下载 `escontra/gauss_gym_arkit` 场景。

验证命令只检查本机或安全 secret 注入是否存在，不打印 token：

```bash
test -n "${HF_TOKEN:-}" && echo "HF_TOKEN is set" || echo "HF_TOKEN is missing"
```

要求：

- 用环境变量、启动模板、SSM Parameter Store 或运行时注入。
- 不写入仓库文件。
- 不回显 token。

## 4. 执行步骤（按序，标注哪步付费）

### Step A（免费）：核验清单 + 整理待填环境变量

先完成 §3 全部核验。然后把 `scripts/aws/` 三个脚本里的占位变量整理成待填清单给用户确认。

待填环境变量清单：

```bash
export AWS_REGION="<us-east-1-or-us-west-2>"
export REGION="$AWS_REGION"
export AWS_ACCOUNT_ID="<12-digit-account-id>"
export BUDGET_EMAIL="<user-email>"
export BUDGET_USD="50"
export COST_SHUTDOWN_USD="45"

export S3_URI="s3://<bucket>/<prefix>"
export BUCKET="$S3_URI"
export RUN_NAME="<flagship-d1-run-name>"
export S3_RUN_URI="${BUCKET%/}/runs/${RUN_NAME}"

export AMI_ID="<ami-id>"
export INSTANCE_TYPE="g5.2xlarge"
export EC2_KEY_NAME="<key-pair-name>"
export SUBNET_ID="<subnet-id>"
export SECURITY_GROUP_ID="<sg-id>"
export IAM_INSTANCE_PROFILE_NAME="<instance-profile-name>"

export PROJECT_REPO_URL="<repo-url>"
export PROJECT_BRANCH="<branch>"
export GAUSSGYM_REPO_URL="<gaussgym-repo-url>"
export GAUSSGYM_BRANCH="<branch>"
export PROJECT_DIR="/opt/SemCost-SplatNav"
export GAUSSGYM_DIR="/opt/GaussGym"
export SERVICE_USER="ubuntu"

export USE_LIGHTWEIGHT_ENV="1"
export HF_TOKEN="<do-not-print-or-commit>"
```

需要注意：

- `scripts/aws/ec2_user_data.sh` 默认值里有 `YOUR_BUCKET_NAME`、`YOUR_GITHUB_USER` 等占位符；真实值通过启动模板或环境注入，不要改仓库文件。
- `scripts/aws/run_flagship_overnight.sh` 当前 `train_one()` / `evaluate_all()` 是占位命令；`USE_LIGHTWEIGHT_ENV=1` 用于先验证 S3、日志、预算 watchdog、自动关机链路。
- `scripts/aws/setup_budget_guard.sh` 需要 `AWS_ACCOUNT_ID`、`BUDGET_EMAIL`、`BUDGET_USD`、`COST_SHUTDOWN_USD`。

### Step B（免费）：先建预算护栏

在有权限的控制机上运行：

```bash
export REGION="$AWS_REGION"
export AWS_ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
export BUDGET_EMAIL="<user-email>"
export BUDGET_USD="50"
export COST_SHUTDOWN_USD="45"

bash scripts/aws/setup_budget_guard.sh
```

验证预算与 watchdog 相关资源：

```bash
aws budgets describe-budget \
  --account-id "$AWS_ACCOUNT_ID" \
  --budget-name "SemCost-SplatNav-Flagship-50USD"
```

说明：

- AWS Budgets 告警不是实时硬断电。
- `setup_budget_guard.sh` 还会安装本地 cost/GPU idle watchdog；若在控制机运行，只能完成预算部分或因 systemd 权限失败。实际 EC2 上仍要验证 watchdog timer。

### Step C（闸1 → 付费，几分钱）：用轻量环境起 spot 验证链路

闸1：在执行任何 `aws ec2 run-instances` 之前，停下向用户确认：

- §3 前置核验全部通过。
- §4 Step A 环境变量全部已填。
- §4 Step B 预算护栏已创建或明确由用户接受缺口。
- 用户确认可以产生少量 AWS 费用。

确认后，用 `USE_LIGHTWEIGHT_ENV=1` 起 `g5.2xlarge` spot。目标只验证：

- 开机自启动。
- systemd service 正常。
- S3 恢复/同步。
- GPU 空闲/预算 watchdog。
- 自动关机链路。
- root/EBS 持久化与 S3 续跑兜底。

实例启动后验证：

```bash
export INSTANCE_ID="<instance-id>"

aws ec2 describe-instances \
  --region "$AWS_REGION" \
  --instance-ids "$INSTANCE_ID" \
  --query "Reservations[0].Instances[0].[InstanceId,State.Name,InstanceType,Placement.AvailabilityZone,SpotInstanceRequestId,PublicDnsName]" \
  --output table
```

SSH 到实例后验证：

```bash
journalctl -u semcost-splatnav-overnight.service -n 200 --no-pager
journalctl -u semcost-splatnav-overnight.service -f

sudo systemctl status semcost-splatnav-overnight.service --no-pager
sudo systemctl status semcost-budget-idle-watchdog.timer --no-pager
journalctl -u semcost-budget-idle-watchdog.service -n 100 --no-pager

nvidia-smi
aws sts get-caller-identity
aws s3 ls "$S3_RUN_URI/" --recursive --summarize --region "$AWS_REGION"
```

验证自动关机：

```bash
sudo systemctl stop semcost-splatnav-overnight.service
aws s3 sync /opt/SemCost-SplatNav/logs/aws_flagship "$S3_RUN_URI/logs" --region "$AWS_REGION" --only-show-errors
sudo shutdown -h now
```

关机后在控制机确认：

```bash
aws ec2 describe-instances \
  --region "$AWS_REGION" \
  --instance-ids "$INSTANCE_ID" \
  --query "Reservations[0].Instances[0].[InstanceId,State.Name]" \
  --output table
```

### Step D（闸2 → 付费，主要花费）：切完整 GaussGym 路径

闸2：从 `USE_LIGHTWEIGHT_ENV=1` 廉价验证切到 `USE_LIGHTWEIGHT_ENV=0` 前，停下向用户确认：

- Step C 轻量链路通过。
- S3 checkpoint/results/logs 同步可见。
- watchdog 与自动关机已验证。
- 当前花费低于软闸 `45 USD`。
- 用户确认可以进入完整 GaussGym 安装与 smoke。

确认后：

```bash
export USE_LIGHTWEIGHT_ENV="0"
```

按 `docs/AWS_FLAGSHIP_PLAN.md` §5 处理 GaussGym 风险：完整 GaussGym / IsaacGym 安装 timebox `6h`。优先走 GaussGym 的 `setup_dev.sh`：

```bash
cd "$GAUSSGYM_DIR"
bash setup_dev.sh
```

smoke 命令目标：

```bash
gauss_train --task=a1_vision --env.num_envs 512
```

若装不上或超过 6h：

- 立刻停止长时间付费排错。
- 按 `docs/AWS_FLAGSHIP_PLAN.md` §5 降级：转自写轻量 gsplat batched env / 本仓库轻量路径。
- 同步日志到 S3。
- 自动或手动关机。
- 简短回报阻塞点。

锁定版本来自 `docs/D0_SUMMARY.md` 与目标 `docs/aws/gauss_gym_env.yml`：

- Python `3.8`
- CUDA `12.1`
- torch `2.4.1`
- torchvision `0.19.1`
- gsplat `1.5.3`，GaussGym git pin
- IsaacGym `Preview 4 / 1.0rc4`
- warp-lang `1.7.1`
- numpy `1.23.5`

若 `docs/aws/gauss_gym_env.yml` 当前不存在，记录为阻塞/仓库缺口，不要新建其它文件。

## 5. 起 spot 的精确命令模板

先填这些环境变量。所有真实值都来自用户或安全环境，不写进仓库：

```bash
export AWS_REGION="<us-east-1-or-us-west-2>"
export AMI_ID="<ubuntu-22.04-gpu-or-dlami-ami-id>"
export EC2_KEY_NAME="<key-pair-name>"
export SUBNET_ID="<subnet-id>"
export SECURITY_GROUP_ID="<sg-id>"
export IAM_INSTANCE_PROFILE_NAME="<instance-profile-name>"
export RUN_NAME="<flagship-d1-run-name>"
```

起 spot 模板：

```bash
aws ec2 run-instances \
  --region "$AWS_REGION" \
  --image-id "$AMI_ID" \
  --instance-type g5.2xlarge \
  --key-name "$EC2_KEY_NAME" \
  --subnet-id "$SUBNET_ID" \
  --security-group-ids "$SECURITY_GROUP_ID" \
  --iam-instance-profile "Name=$IAM_INSTANCE_PROFILE_NAME" \
  --instance-market-options 'MarketType=spot,SpotOptions={SpotInstanceType=persistent,InstanceInterruptionBehavior=stop}' \
  --user-data file://scripts/aws/ec2_user_data.sh \
  --block-device-mappings '[
    {
      "DeviceName": "/dev/sda1",
      "Ebs": {
        "VolumeSize": 150,
        "VolumeType": "gp3",
        "DeleteOnTermination": false
      }
    }
  ]' \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=semcost-splatnav-d1},{Key=Project,Value=SemCost-SplatNav},{Key=RunName,Value=$RUN_NAME}]"
```

必须确认：

- `SpotInstanceType=persistent`。
- `InstanceInterruptionBehavior=stop`。
- root/EBS `DeleteOnTermination=false`。
- 启动前已经通过安全方式把 `REGION`、`BUCKET`、`RUN_NAME`、`PROJECT_REPO_URL`、`GAUSSGYM_REPO_URL`、`USE_LIGHTWEIGHT_ENV=1`、`HF_TOKEN` 等运行变量注入到 user-data 或启动模板。不要把真实桶名或 token commit 到仓库。

如果只用当前 `scripts/aws/ec2_user_data.sh` 原文件作为 user-data，它内部默认仍有占位值；必须在实际启动方式里覆盖这些值，否则会 clone/sync 到错误目标。

## 6. 完成 / 返回条件

成功条件：

- 实例起来。
- `USE_LIGHTWEIGHT_ENV=1` 链路验证通过：开机自启动、S3 同步、watchdog、自动关机。
- 闸2 后，GaussGym smoke `gauss_train --task=a1_vision --env.num_envs 512` 通过，或已按 `docs/AWS_FLAGSHIP_PLAN.md` §5 降级路径处理。
- 简短状态已回报给主 agent。

失败/阻塞条件：

- 任一前置核验缺失。
- 需要用户 AWS 凭证、MFA、Console 批准或 quota 操作。
- 预算护栏无法创建且用户未确认继续。
- spot 无容量、AMI 不可用、权限不足、S3 不可写。
- GaussGym/IsaacGym 安装超过 6h timebox。

失败/阻塞时：

- 记录关键事实到 `docs/devlog`（如果已有约定文件）或直接回报主 agent。
- 先同步 S3，再停机。
- 停下问用户。

始终：

- 不确定就停，宁可问不要烧钱。
- 不要把完整日志回灌主上下文。

## 7. 给主 agent 的回报模板

```markdown
AWS D1 status:

- Instance id: `<i-... or None>`
- Region: `<us-east-1/us-west-2>`
- S3 path: `<s3://bucket/prefix/runs/run-name>`
- Spend this session: `<USD estimate or unknown>`
- Step A preflight: `<pass/fail/block>`
- Step B budget guard: `<pass/fail/block>`
- Step C lightweight spot chain: `<pass/fail/block>`
- Step D GaussGym smoke: `<not-started/pass/fail/degraded>`
- Current state: `<running/stopped/terminated/not-launched>`
- Blockers: `<short list or None>`
- Next step: `<one line>`
```

本文件由主 agent 审查后定稿。
