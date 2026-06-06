#!/usr/bin/env bash
# SemCost-SplatNav AWS overnight flagship runner.
#
# 目标：
# - 顺序跑三路消融：depth-only / rgb / rgb_semantic
# - 每隔一段时间把 checkpoint / results / logs 同步到 S3，支持 spot 中断后续跑
# - 三路失败互不阻塞，最后统一 evaluate、回传产物并自动关机
#
# 注意：下面的 train_one/evaluate_all 目前是占位命令。接入 GaussGym 后，只需要
# 替换这两个函数中的 echo/sleep 为真实 gauss_train / evaluate 命令即可。

set -euo pipefail

#######################################
# User variables. Override with env vars.
#######################################
BUCKET="${BUCKET:-s3://YOUR_BUCKET_NAME/semcost-splatnav}"  # 不要写真实密钥；桶名用环境变量覆盖。
REGION="${REGION:-us-east-1}"
PROJECT_DIR="${PROJECT_DIR:-$HOME/SemCost-SplatNav}"
GAUSSGYM_DIR="${GAUSSGYM_DIR:-$HOME/GaussGym}"
RUN_NAME="${RUN_NAME:-flagship-$(date -u +%Y%m%dT%H%M%SZ)}"
S3_RUN_URI="${S3_RUN_URI:-${BUCKET%/}/runs/${RUN_NAME}}"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-$PROJECT_DIR/checkpoints/aws_flagship}"
RESULTS_DIR="${RESULTS_DIR:-$PROJECT_DIR/results/aws_flagship}"
LOG_DIR="${LOG_DIR:-$PROJECT_DIR/logs/aws_flagship}"
SYNC_INTERVAL_SECONDS="${SYNC_INTERVAL_SECONDS:-300}"  # 每 N 秒同步一次 checkpoint。
SHUTDOWN_ON_FINISH="${SHUTDOWN_ON_FINISH:-1}"

# Locked versions from GaussGym authoritative environment target.
PYTHON_VERSION="3.8"
CUDA_VERSION="12.1"
PYTORCH_VERSION="2.4.1"
TORCHVISION_VERSION="0.19.1"
TORCH_CUDA_TAG="cu121"
GSPLAT_VERSION="1.5.3"
ISAACGYM_VERSION="Preview 4 / 1.0rc4"
WARP_LANG_VERSION="1.7.1"
NUMPY_VERSION="1.23.5"

ABLATIONS=("depth-only" "rgb" "rgb_semantic")
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
MASTER_LOG="${LOG_DIR}/overnight_${RUN_NAME}_${TIMESTAMP}.log"

#######################################
# Helpers
#######################################
log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$MASTER_LOG"
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || {
    log "ERROR: required command not found: $1"
    exit 127
  }
}

sync_to_s3() {
  log "Syncing local artifacts to ${S3_RUN_URI}"
  aws s3 sync "$CHECKPOINT_DIR" "$S3_RUN_URI/checkpoints" --region "$REGION" --only-show-errors || true
  aws s3 sync "$RESULTS_DIR" "$S3_RUN_URI/results" --region "$REGION" --only-show-errors || true
  aws s3 sync "$LOG_DIR" "$S3_RUN_URI/logs" --region "$REGION" --only-show-errors || true
}

sync_from_s3() {
  log "Restoring prior checkpoints/results from ${S3_RUN_URI} if present"
  aws s3 sync "$S3_RUN_URI/checkpoints" "$CHECKPOINT_DIR" --region "$REGION" --only-show-errors || true
  aws s3 sync "$S3_RUN_URI/results" "$RESULTS_DIR" --region "$REGION" --only-show-errors || true
}

start_periodic_sync() {
  while true; do
    sleep "$SYNC_INTERVAL_SECONDS"
    sync_to_s3
  done
}

stop_periodic_sync() {
  local pid="${1:-}"
  if [[ -n "$pid" ]] && kill -0 "$pid" >/dev/null 2>&1; then
    kill "$pid" >/dev/null 2>&1 || true
    wait "$pid" >/dev/null 2>&1 || true
  fi
}

train_one() {
  local mode="$1"
  local mode_log="${LOG_DIR}/train_${mode}_${TIMESTAMP}.log"
  local mode_ckpt="${CHECKPOINT_DIR}/${mode}"
  mkdir -p "$mode_ckpt"

  log "Starting ablation=${mode}; log=${mode_log}"

  # 占位训练命令：
  # 后续替换为 GaussGym 实际训练命令。真实入口（来自 GaussGym README / pyproject scripts）：
  #   source ~/.gauss_gym_deps/miniconda3/bin/activate gauss_gym
  #   cd "$GAUSSGYM_DIR"
  #   gauss_train --task=a1_vision --env.num_envs 512   # A10G 24GB 保守值，OK 再升 1024
  #     # 三路消融通过切换观测相关 flag 实现（同 task / 同 seed / 同 max_iterations）：
  #     #   depth-only    : 仅深度通道
  #     #   rgb           : 渲染 RGB
  #     #   rgb_semantic  : RGB + post-hoc 语义 cost 通道（地毯 hazard）
  #     # 场景 override：--terrain.scenes.arkit.split_frac=1.0（iphone_data/grand_tour=0）
  #     # checkpoint/resume 走 config 的 runner.* 与本脚本的 S3 sync 兜底
  #
  # 这里保留轻量占位，避免脚本在未接入 GaussGym 时误启动长任务。
  {
    echo "PLACEHOLDER: train ablation=${mode}"
    echo "Pinned env: Python ${PYTHON_VERSION}, CUDA ${CUDA_VERSION}, PyTorch ${PYTORCH_VERSION}+${TORCH_CUDA_TAG}, torchvision ${TORCHVISION_VERSION}+${TORCH_CUDA_TAG}, gsplat ${GSPLAT_VERSION}, IsaacGym ${ISAACGYM_VERSION}, warp-lang ${WARP_LANG_VERSION}, numpy ${NUMPY_VERSION}"
    echo "Replace train_one() placeholder with gauss_train command."
    sleep 5
    touch "${mode_ckpt}/PLACEHOLDER_${mode}_checkpoint.txt"
  } 2>&1 | tee -a "$mode_log"

  log "Finished ablation=${mode}"
}

evaluate_all() {
  local eval_log="${LOG_DIR}/evaluate_${TIMESTAMP}.log"
  log "Starting final evaluation; log=${eval_log}"

  # 占位评估命令：
  # 后续替换为 GaussGym 实际评估命令（来自 README）：
  #   gauss_play --runner.load_run=<RUN_NAME>     # RUN_NAME = logs/ 下的运行名或 wandb 名
  # 三路各自 eval 后汇总到三路对比（time-in-bad-region 曲线 + 指标表）。
  {
    echo "PLACEHOLDER: evaluate all ablations"
    echo "Expected ablations: ${ABLATIONS[*]}"
    touch "${RESULTS_DIR}/PLACEHOLDER_eval_done.txt"
  } 2>&1 | tee -a "$eval_log"
}

finalize() {
  local exit_code=$?
  log "Final sync before exit; exit_code=${exit_code}"
  sync_to_s3
  if [[ "$SHUTDOWN_ON_FINISH" == "1" ]]; then
    log "SHUTDOWN_ON_FINISH=1; shutting down instance now"
    sudo shutdown -h now || true
  else
    log "SHUTDOWN_ON_FINISH!=1; leaving instance running"
  fi
}

#######################################
# Main
#######################################
mkdir -p "$CHECKPOINT_DIR" "$RESULTS_DIR" "$LOG_DIR"
touch "$MASTER_LOG"
trap finalize EXIT

require_command aws
log "Run name: ${RUN_NAME}"
log "S3 run URI: ${S3_RUN_URI}"
log "Project dir: ${PROJECT_DIR}"
log "GaussGym dir: ${GAUSSGYM_DIR}"

sync_from_s3

start_periodic_sync &
SYNC_PID="$!"
log "Started periodic S3 sync pid=${SYNC_PID}, interval=${SYNC_INTERVAL_SECONDS}s"

for mode in "${ABLATIONS[@]}"; do
  train_one "$mode" || true
  sync_to_s3
done

evaluate_all || true
stop_periodic_sync "$SYNC_PID"
sync_to_s3

