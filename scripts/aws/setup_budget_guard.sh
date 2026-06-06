#!/usr/bin/env bash
# Create a hard budget guard for the AWS flagship run.
#
# Includes:
# - AWS Budgets monthly budget at BUDGET_USD with 80% / 100% alerts
# - local cost/GPU-idle watchdog systemd timer that shuts down near COST_SHUTDOWN_USD
#   or after sustained GPU idleness

set -euo pipefail

#######################################
# User variables. Override with env vars.
#######################################
REGION="${REGION:-us-east-1}"
AWS_ACCOUNT_ID="${AWS_ACCOUNT_ID:-}"              # 必填，可用 aws sts get-caller-identity 查询。
BUDGET_NAME="${BUDGET_NAME:-SemCost-SplatNav-Flagship-50USD}"
BUDGET_USD="${BUDGET_USD:-50}"
BUDGET_EMAIL="${BUDGET_EMAIL:-you@example.com}"   # 改成自己的邮箱；预算告警会发到这里。
COST_SHUTDOWN_USD="${COST_SHUTDOWN_USD:-45}"
GPU_IDLE_THRESHOLD_PERCENT="${GPU_IDLE_THRESHOLD_PERCENT:-10}"
GPU_IDLE_MINUTES="${GPU_IDLE_MINUTES:-30}"
WATCHDOG_INTERVAL_MINUTES="${WATCHDOG_INTERVAL_MINUTES:-5}"
WATCHDOG_PATH="${WATCHDOG_PATH:-/usr/local/bin/semcost_budget_idle_watchdog.sh}"
STATE_DIR="${STATE_DIR:-/var/lib/semcost-splatnav}"

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

ensure_account_id() {
  if [[ -z "$AWS_ACCOUNT_ID" ]]; then
    AWS_ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
  fi
}

create_budget() {
  local budget_file notifications_file
  budget_file="$(mktemp)"
  notifications_file="$(mktemp)"

  cat >"$budget_file" <<EOF
{
  "BudgetName": "${BUDGET_NAME}",
  "BudgetLimit": {
    "Amount": "${BUDGET_USD}",
    "Unit": "USD"
  },
  "TimeUnit": "MONTHLY",
  "BudgetType": "COST",
  "CostFilters": {},
  "CostTypes": {
    "IncludeTax": true,
    "IncludeSubscription": true,
    "UseBlended": false,
    "IncludeRefund": false,
    "IncludeCredit": false,
    "IncludeUpfront": true,
    "IncludeRecurring": true,
    "IncludeOtherSubscription": true,
    "IncludeSupport": true,
    "IncludeDiscount": true,
    "UseAmortized": false
  }
}
EOF

  cat >"$notifications_file" <<EOF
[
  {
    "Notification": {
      "NotificationType": "ACTUAL",
      "ComparisonOperator": "GREATER_THAN",
      "Threshold": 80,
      "ThresholdType": "PERCENTAGE"
    },
    "Subscribers": [
      {
        "SubscriptionType": "EMAIL",
        "Address": "${BUDGET_EMAIL}"
      }
    ]
  },
  {
    "Notification": {
      "NotificationType": "ACTUAL",
      "ComparisonOperator": "GREATER_THAN",
      "Threshold": 100,
      "ThresholdType": "PERCENTAGE"
    },
    "Subscribers": [
      {
        "SubscriptionType": "EMAIL",
        "Address": "${BUDGET_EMAIL}"
      }
    ]
  }
]
EOF

  if aws budgets describe-budget --account-id "$AWS_ACCOUNT_ID" --budget-name "$BUDGET_NAME" >/dev/null 2>&1; then
    log "Budget exists; updating ${BUDGET_NAME}"
    aws budgets update-budget --account-id "$AWS_ACCOUNT_ID" --new-budget "file://${budget_file}"
  else
    log "Creating budget ${BUDGET_NAME}"
    aws budgets create-budget \
      --account-id "$AWS_ACCOUNT_ID" \
      --budget "file://${budget_file}" \
      --notifications-with-subscribers "file://${notifications_file}"
  fi

  rm -f "$budget_file" "$notifications_file"
}

install_watchdog() {
  mkdir -p "$STATE_DIR"

  cat >"$WATCHDOG_PATH" <<'WATCHDOG'
#!/usr/bin/env bash
set -euo pipefail

REGION="${REGION:-us-east-1}"
COST_SHUTDOWN_USD="${COST_SHUTDOWN_USD:-45}"
GPU_IDLE_THRESHOLD_PERCENT="${GPU_IDLE_THRESHOLD_PERCENT:-10}"
GPU_IDLE_MINUTES="${GPU_IDLE_MINUTES:-30}"
STATE_DIR="${STATE_DIR:-/var/lib/semcost-splatnav}"
IDLE_STATE_FILE="${STATE_DIR}/gpu_idle_since"
COST_BASELINE_FILE="${STATE_DIR}/cost_baseline"

log() {
  logger -t semcost-budget-idle-watchdog "$*"
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

mkdir -p "$STATE_DIR"

month_start="$(date -u +%Y-%m-01)"
tomorrow="$(date -u -d tomorrow +%Y-%m-%d 2>/dev/null || date -u -v+1d +%Y-%m-%d)"

current_cost="$(aws ce get-cost-and-usage \
  --region "$REGION" \
  --time-period Start="$month_start",End="$tomorrow" \
  --granularity MONTHLY \
  --metrics UnblendedCost \
  --query 'ResultsByTime[0].Total.UnblendedCost.Amount' \
  --output text 2>/dev/null || echo 0)"

# COST_SHUTDOWN_USD 是【本次实验会话的增量】上限，不是月累计。
# 首次运行时把当前月累计成本记为 baseline；之后用 (当前月累计 - baseline) 作为本次花费，
# 这样月初已有其它账单也不会误触发，且严格对应用户『单次实验 ≤50 美元』的预期。
if [[ ! -f "$COST_BASELINE_FILE" ]]; then
  echo "$current_cost" > "$COST_BASELINE_FILE"
  log "Recorded cost baseline (month-to-date at session start) = ${current_cost} USD"
fi
cost_baseline="$(cat "$COST_BASELINE_FILE" 2>/dev/null || echo 0)"
session_cost="$(awk -v c="$current_cost" -v b="$cost_baseline" 'BEGIN { d = c - b; if (d < 0) d = 0; print d }')"

awk -v cost="$session_cost" -v limit="$COST_SHUTDOWN_USD" 'BEGIN { exit !(cost + 0 >= limit + 0) }' && {
  log "Session cost ${session_cost} USD (month-to-date ${current_cost} - baseline ${cost_baseline}) >= shutdown threshold ${COST_SHUTDOWN_USD}; shutting down"
  shutdown -h now
  exit 0
}

if command -v nvidia-smi >/dev/null 2>&1; then
  gpu_util="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | awk '{sum+=$1; n+=1} END {if (n==0) print 0; else print int(sum/n)}')"
else
  gpu_util="0"
fi

now_epoch="$(date -u +%s)"
if awk -v util="$gpu_util" -v threshold="$GPU_IDLE_THRESHOLD_PERCENT" 'BEGIN { exit !(util + 0 < threshold + 0) }'; then
  if [[ ! -f "$IDLE_STATE_FILE" ]]; then
    echo "$now_epoch" > "$IDLE_STATE_FILE"
    log "GPU idle started; avg utilization=${gpu_util}%"
    exit 0
  fi
  idle_since="$(cat "$IDLE_STATE_FILE")"
  idle_minutes="$(( (now_epoch - idle_since) / 60 ))"
  if (( idle_minutes >= GPU_IDLE_MINUTES )); then
    log "GPU idle for ${idle_minutes} minutes with utilization ${gpu_util}% < ${GPU_IDLE_THRESHOLD_PERCENT}%; shutting down"
    shutdown -h now
  else
    log "GPU idle for ${idle_minutes}/${GPU_IDLE_MINUTES} minutes; avg utilization=${gpu_util}%"
  fi
else
  rm -f "$IDLE_STATE_FILE"
  log "GPU active; avg utilization=${gpu_util}%"
fi
WATCHDOG

  chmod +x "$WATCHDOG_PATH"

  cat >/etc/systemd/system/semcost-budget-idle-watchdog.service <<EOF
[Unit]
Description=SemCost-SplatNav cost and GPU idle watchdog

[Service]
Type=oneshot
Environment=REGION=${REGION}
Environment=COST_SHUTDOWN_USD=${COST_SHUTDOWN_USD}
Environment=GPU_IDLE_THRESHOLD_PERCENT=${GPU_IDLE_THRESHOLD_PERCENT}
Environment=GPU_IDLE_MINUTES=${GPU_IDLE_MINUTES}
Environment=STATE_DIR=${STATE_DIR}
ExecStart=${WATCHDOG_PATH}
EOF

  cat >/etc/systemd/system/semcost-budget-idle-watchdog.timer <<EOF
[Unit]
Description=Run SemCost-SplatNav cost and GPU idle watchdog periodically

[Timer]
OnBootSec=5min
OnUnitActiveSec=${WATCHDOG_INTERVAL_MINUTES}min
AccuracySec=30s
Persistent=true

[Install]
WantedBy=timers.target
EOF

  systemctl daemon-reload
  systemctl enable --now semcost-budget-idle-watchdog.timer
}

main() {
  command -v aws >/dev/null 2>&1 || {
    echo "aws CLI is required" >&2
    exit 127
  }
  ensure_account_id
  create_budget
  install_watchdog
  log "Budget guard installed: budget=${BUDGET_USD} USD, shutdown threshold=${COST_SHUTDOWN_USD} USD, GPU idle=${GPU_IDLE_MINUTES} minutes"
}

main "$@"

