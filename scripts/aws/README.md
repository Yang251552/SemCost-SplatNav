# SemCost-SplatNav AWS Flagship Overnight

这些脚本把 `g5.2xlarge` spot 上的过夜 GPU 训练做成无人值守、可续跑、带预算保护的流程。所有真实值都用环境变量或启动模板注入，不要把 AWS 密钥、真实桶名或 HF token 写进仓库。

## 文件

- `run_flagship_overnight.sh`: 顺序运行 `depth-only` / `rgb` / `rgb_semantic` 三路消融，周期同步 checkpoint 到 S3，最后 evaluate、回传、关机。
- `ec2_user_data.sh`: EC2 user-data。安装 Docker、NVIDIA container toolkit、AWS CLI，clone 本项目和 GaussGym，从 S3 恢复后用 systemd 后台启动训练。
- `setup_budget_guard.sh`: 创建 50 USD AWS Budgets 告警，并安装 cost/GPU idle watchdog。费用接近 45 USD 或 GPU 空闲过久时关机。

## 版本锁定

目标环境固定为：

- Python `3.8`
- CUDA `12.1`
- PyTorch `2.4.1` + torchvision `0.19.1` (`cu121`)
- gsplat `1.5.3`
- IsaacGym `Preview 4 / 1.0rc4`
- warp-lang `1.7.1`
- numpy `1.23.5`

当前训练和评估命令是占位命令。接入 GaussGym 时，替换 `run_flagship_overnight.sh` 里的 `train_one()` 和 `evaluate_all()` 函数为真实 `gauss_train` / evaluate 命令。

## 前置

本机或控制机需要：

```bash
aws configure
aws sts get-caller-identity
aws s3 mb s3://YOUR_BUCKET_NAME --region us-east-1
```

EC2 instance profile 至少需要：

- `s3:GetObject`, `s3:PutObject`, `s3:ListBucket` 到你的实验桶
- `ce:GetCostAndUsage` 用于 watchdog 查成本
- `budgets:*Budget*` 用于创建/更新预算
- `sts:GetCallerIdentity`

如需 Hugging Face 私有模型或数据，在启动模板或 SSM Parameter Store 注入 `HF_TOKEN`，不要写入脚本。

## 创建预算与 watchdog

在 EC2 上或有权限的控制机上运行：

```bash
export AWS_ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
export BUDGET_EMAIL="you@example.com"
export BUDGET_USD="50"
export COST_SHUTDOWN_USD="45"
bash scripts/aws/setup_budget_guard.sh
```

预算告警默认在 80% 和 100% 发邮件。watchdog 默认每 5 分钟检查一次，GPU 平均利用率低于 10% 且持续 30 分钟会关机。

## 启动 spot

推荐实例：

- Instance type: `g5.2xlarge`
- GPU: NVIDIA A10G 24GB
- Market: Spot
- AMI: Ubuntu 22.04 GPU/NVIDIA driver AMI，或 Deep Learning AMI GPU Ubuntu
- Root volume: 建议 150GB+
- IAM role: 绑定上面的 S3/Cost/Budget 权限
- **EBS 持久化（spot 续跑相关）**: 用 `InstanceInterruptionBehavior=stop` + `persistent` 时，确保 root/EBS 卷
  `DeleteOnTermination=false`，这样 spot 被回收（stop）时卷不被删、重启后本地数据还在，省一次重下/重装。
  即使忘了设，S3 兜底（checkpoints/results 已实现周期同步）仍能恢复进度，只是要重新下数据。

示例命令需要先替换 subnet、security group、key、AMI：

```bash
aws ec2 run-instances \
  --region us-east-1 \
  --image-id ami-REPLACE_ME \
  --instance-type g5.2xlarge \
  --key-name YOUR_KEY \
  --subnet-id subnet-REPLACE_ME \
  --security-group-ids sg-REPLACE_ME \
  --iam-instance-profile Name=YOUR_INSTANCE_PROFILE \
  --instance-market-options 'MarketType=spot,SpotOptions={SpotInstanceType=persistent,InstanceInterruptionBehavior=stop}' \
  --user-data file://scripts/aws/ec2_user_data.sh \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=semcost-splatnav-flagship}]'
```

常用启动变量可放进启动模板 user-data 顶部：

```bash
export REGION="us-east-1"
export BUCKET="s3://YOUR_BUCKET_NAME/semcost-splatnav"
export RUN_NAME="flagship-overnight-001"
export PROJECT_REPO_URL="https://github.com/YOUR_GITHUB_USER/SemCost-SplatNav.git"
export GAUSSGYM_REPO_URL="https://github.com/YOUR_GITHUB_USER/GaussGym.git"
export HF_TOKEN="hf_xxx"
```

## 监控

SSH 到实例后：

```bash
journalctl -u semcost-splatnav-overnight.service -f
tail -f /opt/SemCost-SplatNav/logs/aws_flagship/*.log
nvidia-smi
aws s3 ls s3://YOUR_BUCKET_NAME/semcost-splatnav/runs/flagship-overnight-001/ --recursive --summarize
```

watchdog 日志：

```bash
journalctl -u semcost-budget-idle-watchdog.service -n 100
systemctl list-timers | grep semcost
```

## 手动停机

先同步，再关机：

```bash
sudo systemctl stop semcost-splatnav-overnight.service
aws s3 sync /opt/SemCost-SplatNav/checkpoints/aws_flagship s3://YOUR_BUCKET_NAME/semcost-splatnav/runs/flagship-overnight-001/checkpoints
aws s3 sync /opt/SemCost-SplatNav/results/aws_flagship s3://YOUR_BUCKET_NAME/semcost-splatnav/runs/flagship-overnight-001/results
sudo shutdown -h now
```

## 续跑

Spot 被抢后重新启动同一个 `RUN_NAME` 即可。`ec2_user_data.sh` 会从：

```text
s3://YOUR_BUCKET_NAME/semcost-splatnav/runs/RUN_NAME/checkpoints
s3://YOUR_BUCKET_NAME/semcost-splatnav/runs/RUN_NAME/results
```

恢复到本地，然后启动 `run_flagship_overnight.sh` 接力。真实 GaussGym 命令需要使用对应的 `--resume` / `--checkpoint-dir` 参数。

## 成本预期

`g5.2xlarge` spot 价格随区域波动。50 USD 通常足够跑一个短过夜窗口，但不保证覆盖长训练、失败重试和大规模 S3 传输。预算告警不是实时硬断电；本仓库额外安装的 watchdog 用 Cost Explorer 月累计费用接近 45 USD 时主动关机，仍可能有数小时账单延迟。

## 降级开关

GaussGym 或 IsaacGym 安装失败时，把：

```bash
export USE_LIGHTWEIGHT_ENV=1
```

写入 user-data 或 `/etc/semcost-splatnav-aws.env`，然后重启服务：

```bash
sudo systemctl restart semcost-splatnav-overnight.service
```

该模式只保留本仓库轻量环境和占位训练流程，适合验证 S3 同步、日志、预算守护和自动关机链路。

