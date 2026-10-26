#!/usr/bin/env bash
# dsv41 serve watchdog: auto-recover the TP4 line after node reboots / engine deaths.
# Runs as a systemd --user service on the head only. Skips when:
#   - /tmp/dsv41-maintenance exists (operator window marker)
#   - an active serve orchestration is already running
#   - the engine answers /v1/models
# Recovery = reaper + start-tp4.sh serve (production .env.tp4, whatever it holds).
set -u
ROOT=/home/colin/nvmodels/DeepSeek-v4.1-Flash-DGX-Sparks
LOG=/tmp/dsv41-watchdog.log
mark() { echo "[$(date '+%F %T')] $*" >>"$LOG"; }

while true; do
  sleep 60
  [ -e /tmp/dsv41-maintenance ] && { mark "maintenance marker present, skip"; continue; }
  code=$(curl -s -o /dev/null -w '%{http_code}' -m 4 http://127.0.0.1:8000/v1/models || true)
  [ "$code" = "200" ] && continue
  if pgrep -f 'start-tp4[.]sh serve|start[.]sh serve' >/dev/null 2>&1; then
    mark "engine down but serve orchestration active ($code), waiting"
    continue
  fi
  mark "engine down ($code) and no serve running -> auto-recover"
  # Proven recovery sequence (2026-10-26 adoption): stop + clean + DRAIN before
  # serve — skipping the drain caused slab-cap thrash and a 4-rank barrier timeout.
  "$ROOT/start-tp4.sh" stop >>"$LOG" 2>&1 || true
  for h in spark-a89f spark-141e spark-2fe7; do
    ssh -n "$h" 'docker ps -aq --filter name=dsv41-worker | xargs -r docker rm -f' >/dev/null 2>&1 || true
  done
  nohup python3 "$ROOT/scripts/verify/boot_hygiene.py" reap --minutes 16 >>"$LOG" 2>&1 &
  mark "draining 60s for swap/page release"
  sleep 60
  cd "$ROOT" && nohup ./start-tp4.sh serve >>/tmp/serve-watchdog.log 2>&1 &
  mark "recovery serve dispatched (see /tmp/serve-watchdog.log)"
done
