#!/usr/bin/env bash
# EC2 user-data bootstrap for SemCost-SplatNav AWS flagship overnight run.
#
# 用法：启动 g5.2xlarge spot 时把本文件作为 user-data。脚本会安装 Docker /
# NVIDIA container toolkit / AWS CLI / git，clone 本项目与 GaussGym，从 S3 恢复
# checkpoint，然后通过 systemd 在后台启动 run_flagship_overnight.sh。

set -euo pipefail

#######################################
# User variables. Override by editing user-data or exporting in launch template.
#######################################
REGION="${REGION:-us-east-1}"
BUCKET="${BUCKET:-s3://YOUR_BUCKET_NAME/semcost-splatnav}"
RUN_NAME="${RUN_NAME:-flagship-resume}"
PROJECT_REPO_URL="${PROJECT_REPO_URL:-https://github.com/YOUR_GITHUB_USER/SemCost-SplatNav.git}"
GAUSSGYM_REPO_URL="${GAUSSGYM_REPO_URL:-https://github.com/YOUR_GITHUB_USER/GaussGym.git}"
PROJECT_BRANCH="${PROJECT_BRANCH:-main}"
GAUSSGYM_BRANCH="${GAUSSGYM_BRANCH:-main}"
PROJECT_DIR="${PROJECT_DIR:-/opt/SemCost-SplatNav}"
GAUSSGYM_DIR="${GAUSSGYM_DIR:-/opt/GaussGym}"
SERVICE_USER="${SERVICE_USER:-ubuntu}"
USE_LIGHTWEIGHT_ENV="${USE_LIGHTWEIGHT_ENV:-0}"  # GaussGym 装不上时设为 1，仅使用本仓库轻量环境。
HF_TOKEN="${HF_TOKEN:-}"                         # 如需 Hugging Face 登录，用实例环境变量注入，不要写死。
CONDA_DIR="${CONDA_DIR:-/opt/conda}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-semcost-flagship}"

# Locked versions required by the task/GaussGym target.
PYTHON_VERSION="3.8"
CUDA_VERSION="12.1"
PYTORCH_VERSION="2.4.1"
TORCHVISION_VERSION="0.19.1"
TORCH_CUDA_TAG="cu121"
GSPLAT_VERSION="1.5.3"
ISAACGYM_VERSION="Preview 4 / 1.0rc4"
WARP_LANG_VERSION="1.7.1"
NUMPY_VERSION="1.23.5"

LOG_FILE="/var/log/semcost-splatnav-user-data.log"
exec > >(tee -a "$LOG_FILE") 2>&1

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

apt_install_base() {
  export DEBIAN_FRONTEND=noninteractive
  apt-get update
  apt-get install -y ca-certificates curl gnupg lsb-release git unzip jq python3-pip
}

install_aws_cli() {
  if command -v aws >/dev/null 2>&1; then
    return
  fi
  local arch
  arch="$(uname -m)"
  if [[ "$arch" == "x86_64" ]]; then
    curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip
  else
    curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-aarch64.zip" -o /tmp/awscliv2.zip
  fi
  unzip -q /tmp/awscliv2.zip -d /tmp
  /tmp/aws/install
}

install_docker_and_nvidia_runtime() {
  if ! command -v docker >/dev/null 2>&1; then
    install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    chmod a+r /etc/apt/keyrings/docker.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" > /etc/apt/sources.list.d/docker.list
    apt-get update
    apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
  fi

  distribution=$(. /etc/os-release; echo "${ID}${VERSION_ID}")
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL "https://nvidia.github.io/libnvidia-container/${distribution}/libnvidia-container.list" \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    > /etc/apt/sources.list.d/nvidia-container-toolkit.list
  apt-get update
  apt-get install -y nvidia-container-toolkit
  nvidia-ctk runtime configure --runtime=docker || true
  systemctl restart docker || true
  usermod -aG docker "$SERVICE_USER" || true
}

install_conda_env() {
  if [[ ! -x "${CONDA_DIR}/bin/conda" ]]; then
    local installer="/tmp/miniconda.sh"
    curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o "$installer"
    bash "$installer" -b -p "$CONDA_DIR"
  fi

  # 这是【轻量降级环境】(USE_LIGHTWEIGHT_ENV=1)：仅供本仓库自写的轻量 env / 占位流程，
  # 用来验证 S3 同步、日志、预算守护、自动关机链路。它【不】安装 gsplat / IsaacGym。
  #
  # 关于 gsplat 与 IsaacGym（完整 GaussGym 路径，非本函数职责）：
  # - gsplat 在 GaussGym 的 pyproject 里是 git pin（commit 2323de5905d5e90e035f792fe65bad0fedd413e7），
  #   不是 PyPI 版本，且 cu121 下需现场编译 —— 不要在这里 `pip install gsplat==x`，会装错或编译失败。
  # - IsaacGym Preview 4 / 1.0rc4 由 GaussGym 自己的 setup_dev.sh（wget 免登录下载）或官方 Docker 安装。
  # - 因此完整路径请走 GaussGym 的 setup_dev.sh / docker，而非本函数。
  if ! "${CONDA_DIR}/bin/conda" env list | awk '{print $1}' | grep -qx "$CONDA_ENV_NAME"; then
    "${CONDA_DIR}/bin/conda" create -y -n "$CONDA_ENV_NAME" "python=${PYTHON_VERSION}"
  fi
  "${CONDA_DIR}/bin/conda" run -n "$CONDA_ENV_NAME" python -m pip install --upgrade pip
  # 仅装轻量环境跑占位/降级流程所需的最小依赖（torch + numpy）。gsplat 故意不装。
  "${CONDA_DIR}/bin/conda" run -n "$CONDA_ENV_NAME" python -m pip install \
    "numpy==${NUMPY_VERSION}" \
    "torch==${PYTORCH_VERSION}+${TORCH_CUDA_TAG}" \
    "torchvision==${TORCHVISION_VERSION}+${TORCH_CUDA_TAG}" \
    --extra-index-url "https://download.pytorch.org/whl/${TORCH_CUDA_TAG}"

  chown -R "$SERVICE_USER:$SERVICE_USER" "$CONDA_DIR" || true
}

clone_or_update() {
  local repo_url="$1"
  local branch="$2"
  local dir="$3"
  if [[ -d "$dir/.git" ]]; then
    git -C "$dir" fetch --all --prune
    git -C "$dir" checkout "$branch"
    git -C "$dir" pull --ff-only origin "$branch" || true
  else
    rm -rf "$dir"
    git clone --branch "$branch" "$repo_url" "$dir"
  fi
}

restore_from_s3() {
  local s3_run_uri="${BUCKET%/}/runs/${RUN_NAME}"
  mkdir -p "$PROJECT_DIR/checkpoints/aws_flagship" "$PROJECT_DIR/results/aws_flagship" "$PROJECT_DIR/logs/aws_flagship"
  aws s3 sync "$s3_run_uri/checkpoints" "$PROJECT_DIR/checkpoints/aws_flagship" --region "$REGION" --only-show-errors || true
  aws s3 sync "$s3_run_uri/results" "$PROJECT_DIR/results/aws_flagship" --region "$REGION" --only-show-errors || true
}

write_runtime_env() {
  cat >/etc/semcost-splatnav-aws.env <<EOF
REGION=${REGION}
BUCKET=${BUCKET}
RUN_NAME=${RUN_NAME}
PROJECT_DIR=${PROJECT_DIR}
GAUSSGYM_DIR=${GAUSSGYM_DIR}
USE_LIGHTWEIGHT_ENV=${USE_LIGHTWEIGHT_ENV}
PYTHON_VERSION=${PYTHON_VERSION}
CUDA_VERSION=${CUDA_VERSION}
PYTORCH_VERSION=${PYTORCH_VERSION}
TORCHVISION_VERSION=${TORCHVISION_VERSION}
TORCH_CUDA_TAG=${TORCH_CUDA_TAG}
GSPLAT_VERSION=${GSPLAT_VERSION}
ISAACGYM_VERSION=${ISAACGYM_VERSION}
WARP_LANG_VERSION=${WARP_LANG_VERSION}
NUMPY_VERSION=${NUMPY_VERSION}
HF_TOKEN=${HF_TOKEN}
CONDA_DIR=${CONDA_DIR}
CONDA_ENV_NAME=${CONDA_ENV_NAME}
PATH=${CONDA_DIR}/envs/${CONDA_ENV_NAME}/bin:${CONDA_DIR}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
EOF
  chmod 0600 /etc/semcost-splatnav-aws.env
}

write_systemd_service() {
  cat >/etc/systemd/system/semcost-splatnav-overnight.service <<EOF
[Unit]
Description=SemCost-SplatNav flagship overnight AWS run
After=network-online.target docker.service
Wants=network-online.target

[Service]
Type=simple
User=${SERVICE_USER}
WorkingDirectory=${PROJECT_DIR}
EnvironmentFile=/etc/semcost-splatnav-aws.env
ExecStart=/usr/bin/env bash ${PROJECT_DIR}/scripts/aws/run_flagship_overnight.sh
Restart=on-failure
RestartSec=60
StandardOutput=append:${PROJECT_DIR}/logs/aws_flagship/systemd.out.log
StandardError=append:${PROJECT_DIR}/logs/aws_flagship/systemd.err.log

[Install]
WantedBy=multi-user.target
EOF
}

main() {
  log "Starting SemCost-SplatNav EC2 bootstrap"
  apt_install_base
  install_aws_cli
  install_docker_and_nvidia_runtime
  install_conda_env

  clone_or_update "$PROJECT_REPO_URL" "$PROJECT_BRANCH" "$PROJECT_DIR"
  if [[ "$USE_LIGHTWEIGHT_ENV" != "1" ]]; then
    clone_or_update "$GAUSSGYM_REPO_URL" "$GAUSSGYM_BRANCH" "$GAUSSGYM_DIR" || {
      log "GaussGym clone failed; switching USE_LIGHTWEIGHT_ENV=1"
      USE_LIGHTWEIGHT_ENV="1"
    }
  fi

  restore_from_s3
  mkdir -p "$PROJECT_DIR/logs/aws_flagship"
  chown -R "$SERVICE_USER:$SERVICE_USER" "$PROJECT_DIR" "$GAUSSGYM_DIR" 2>/dev/null || true
  chmod +x "$PROJECT_DIR/scripts/aws/run_flagship_overnight.sh"

  write_runtime_env
  write_systemd_service
  systemctl daemon-reload
  systemctl enable semcost-splatnav-overnight.service
  systemctl start semcost-splatnav-overnight.service
  log "Bootstrap complete. Follow logs with: journalctl -u semcost-splatnav-overnight.service -f"
}

main "$@"
