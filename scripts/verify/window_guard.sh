#!/usr/bin/env bash
# Window guard: host-side auto-rollback that survives operator-session death.
#
# Lesson (2026-11-08, twice): a rollback sequence launched from an operator ssh
# died with the session (SIGHUP) and left production down. This guard runs
# detached on the head under setsid, polls the container, and executes the
# image-only rollback itself when the experiment line is down and nobody is
# bringing it back. It arms ONLY while its guard file exists — remove the file
# to disarm at window end.
#
# Usage (head node, during a flip window):
#   setsid nohup bash window_guard.sh <rollback-tag> <timeout-min> > /tmp/window-guard.log 2>&1 &
#   e.g.: setsid nohup bash scripts/verify/window_guard.sh canary-mig-bp5 90 ... &
# Disarm: rm -f /tmp/dsv41-window-guard
set -uo pipefail
TAG="${1:?usage: window_guard.sh <rollback-image-tag> <timeout-minutes>}"
TIMEOUT_MIN="${2:?usage: window_guard.sh <rollback-image-tag> <timeout-minutes>}"
GUARD_FILE=/tmp/dsv41-window-guard
REPO="$HOME/nvmodels/DeepSeek-v4.1-Flash-DGX-Sparks"
touch "$GUARD_FILE"
echo "[guard] armed $(date '+%F %T') tag=$TAG timeout=${TIMEOUT_MIN}m pid=$$"

start_ts=$(date +%s)
while [ -f "$GUARD_FILE" ]; do
  now=$(date +%s)
  elapsed=$(( (now - start_ts) / 60 ))
  if [ "$elapsed" -ge "$TIMEOUT_MIN" ]; then
    echo "[guard] timeout ${TIMEOUT_MIN}m reached; disarming without action"
    rm -f "$GUARD_FILE"; exit 0
  fi
  sleep 30
  # healthy -> nothing to do
  code=$(curl -s -m 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/health 2>/dev/null || echo 000)
  [ "$code" = "200" ] && continue
  # a serve attempt is in progress -> give it time
  pgrep -f 'start-tp4.sh serve' >/dev/null 2>&1 && { echo "[guard] $(date '+%T') down but serve starting; waiting"; continue; }
  # container up but unhealthy and not starting: allow one more cycle
  state=$(docker inspect dsv41-head --format '{{.State.Status}}' 2>/dev/null || echo gone)
  if [ "$state" = "running" ]; then echo "[guard] $(date '+%T') container running but unhealthy ($code); waiting"; continue; fi
  # DOWN with no recovery running -> execute rollback ourselves
  echo "[guard] $(date '+%T') PRODUCTION DOWN (state=$state health=$code) and no serve in flight -> ROLLING BACK to $TAG"
  cd "$REPO"
  ./start-tp4.sh stop >/dev/null 2>&1 || true
  for h in spark-a89f spark-141e spark-2fe7; do
    ssh -n "$h" 'docker ps -aq --filter name=dsv41 | xargs -r docker rm -f; pkill -9 -f "launch_serve[r]" || true' 2>/dev/null || true
  done
  sleep 60
  for h in spark-eb1a spark-a89f spark-141e spark-2fe7; do
    ssh -n "$h" "docker tag dsv41-4x-spark:$TAG dsv41-4x-spark:canary-mig" 2>/dev/null || true
  done
  nohup python3 scripts/verify/boot_hygiene.py reap --minutes 16 >> /tmp/window-guard.log 2>&1 || true
  setsid nohup ./start-tp4.sh serve >> /tmp/window-guard.log 2>&1 < /dev/null &
  echo "[guard] rollback serve launched; disarming"
  rm -f "$GUARD_FILE"
  exit 0
done
echo "[guard] disarmed $(date '+%F %T')"
