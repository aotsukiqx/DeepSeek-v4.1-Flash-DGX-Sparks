#!/usr/bin/env bash
# mig measurement window (v3, recreated 2026-10-25 after /tmp cleanup ate v2).
# Phase 0 build mig image (env swap during build only) → Phase 1 stop+prewarm+serve
# with image-proof gate → Phase 2 gates/smokes/qeval/probes(cold+decode+conc6) →
# Phase 3 ALWAYS rollback. Probes come from /tmp (repo copies re-deployed).
set -u
ROOT=/home/colin/nvmodels/DeepSeek-v4.1-Flash-DGX-Sparks
LOG=/tmp/mig-window.log
RES=/tmp/mig-window-results
MARK() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
mkdir -p "$RES"

rollback() {
  MARK "ROLLBACK to production line"
  # Capture worker container logs BEFORE start-tp4.sh stop deletes the containers
  # (worker crash evidence; containers are named exactly dsv41-worker).
  for h in spark-a89f spark-141e spark-2fe7; do ssh -n "$h" 'docker logs --tail 400 dsv41-worker > /tmp/worker-crash.log 2>&1' >/dev/null 2>&1 || true; done
  cp /tmp/env.prod.live "$ROOT/.env.tp4" 2>/dev/null || true
  "$ROOT/start-tp4.sh" stop >>"$LOG" 2>&1 || true
  for h in spark-a89f spark-141e spark-2fe7; do ssh -n "$h" 'docker ps -aq --filter name=dsv41-worker | xargs -r docker rm -f' >/dev/null 2>&1; done
  nohup python3 "$ROOT/scripts/verify/boot_hygiene.py" reap --minutes 16 >>/tmp/reaper-w.log 2>&1 &
  nohup "$ROOT/start-tp4.sh" serve >>/tmp/serve-w-restore.log 2>&1 &
  for i in $(seq 1 40); do
    sleep 15
    code=$(curl -s -o /dev/null -w '%{http_code}' -m 4 http://127.0.0.1:8000/v1/models || true)
    [ "$code" = 200 ] && { MARK "PRODUCTION RESTORED (200)"; return 0; }
  done
  MARK "!!! RESTORE FAILED — manual intervention required"
  return 1
}

cd "$ROOT"
cp .env.tp4 /tmp/env.prod.live
trap 'rollback' EXIT

MARK "PHASE0 build mig image"
cp /tmp/env.mig .env.tp4
if ! ./start-tp4.sh build >>"$LOG" 2>&1; then
  MARK "PHASE0 build FAILED"; exit 1
fi
cp /tmp/env.prod.live .env.tp4
MARK "PHASE0 done (production env restored)"

MARK "PHASE1 mig serve"
cp /tmp/env.mig .env.tp4
nohup python3 scripts/verify/boot_hygiene.py reap --minutes 40 >>/tmp/reaper-w.log 2>&1 &
./start-tp4.sh stop >>"$LOG" 2>&1 || true
sleep 5
for h in spark-a89f spark-141e spark-2fe7; do ssh -n "$h" 'docker ps -aq --filter name=dsv41-worker | xargs -r docker rm -f' >/dev/null 2>&1; done
sleep 3
MARK "PHASE1 pre-serve: GPU pids on head: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)"
nohup ./start-tp4.sh serve >>/tmp/serve-w-mig.log 2>&1 &
healthy=0
for i in $(seq 1 175); do
  sleep 30
  code=$(curl -s -o /dev/null -w '%{http_code}' -m 4 http://127.0.0.1:8000/v1/models || true)
  [ "$code" = 200 ] && { healthy=1; break; }
  st=$(docker ps -a --format '{{.Status}}' | head -1)
  echo "$st" | grep -q '^Exited' && {
    MARK "PHASE1 container exited: $st"; docker logs --tail 80 dsv41-head >"$RES/boot-crash.log" 2>&1; break; }
done
if [ "$healthy" != 1 ]; then MARK "PHASE1 FAILED (no health)"; exit 1; fi
MARK "PHASE1 HEALTHY"
img=$(docker ps --format '{{.Image}}' | head -1)
[ "$img" = "dsv41-4x-spark:canary-mig" ] || { MARK "PHASE1 IMAGE PROOF FAILED (running: $img)"; exit 1; }
MARK "PHASE1 image proof OK: $img"

MARK "PHASE2 gates"
L="$ROOT/logs-tp4/dsv41.log"
for pat in "verify_num_draft_tokens" "max_total_num_tokens" "RoCEnante ready" "sm120_candidates ARMED" "mm_tokenizer_compat ARMED" "loop abort: ngram" "Engram prefetch ARMED" "shared-expert" "empty_cache hook installed"; do
  n=$(grep -ac "$pat" "$L" || true); MARK "gate [$n] $pat"
done
docker logs dsv41-head >"$RES/mig-docker.log" 2>&1
MARK "PHASE2 warm+smoke"
curl -s -m 60 http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4.1-flash","temperature":0,"max_tokens":16,"chat_template_kwargs":{"thinking":false},"messages":[{"role":"user","content":"Say OK."}]}' >"$RES/warm.json"
python3 /tmp/svc_smoke.py http://127.0.0.1:8000 >"$RES/smoke.txt" 2>&1; MARK "smoke rc=$?"

MARK "PHASE2 qeval"
python3 /tmp/qeval.py run after --url http://127.0.0.1:8000/v1/chat/completions >"$RES/qeval-run.log" 2>&1 || true
cp /tmp/qeval-after.json "$RES/qeval-mig.json" 2>/dev/null || true

MARK "PHASE2 probes (traffic-gated)"
gap() { local last; last=$(grep 'POST /v1' "$L" | tail -1 | grep -oE '^\[[0-9-]+ [0-9:]+\]')
  [ -z "$last" ] && return 0
  local t=$(date -d "${last:1:-1}" +%s) now=$(date +%s); [ $((now - t)) -ge 45 ]; }
w=0; until gap; do sleep 15; w=$((w+15)); [ $w -ge 900 ] && { MARK "no gap"; break; }; done
{ python3 /tmp/probe600k.py cold 1
  python3 /tmp/probe600k.py cold 3
  python3 /tmp/decode600k.py 7
  python3 /tmp/decode600k.py 8
  python3 /tmp/probe600k.py conc 6 10; } >"$RES/probes.txt" 2>&1
MARK "PHASE2 done"

MARK "PHASE3 end of window (rollback via trap)"
exit 0
