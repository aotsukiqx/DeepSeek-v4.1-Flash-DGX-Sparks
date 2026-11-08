#!/usr/bin/env bash
# Bounded decode profiler (bp6r2 window tool): profile N scheduler steps only.
#
# Lesson (2026-11-08): profiling a full decode600k run on 600k context blows
# host memory on GB10 UMA (scheduler SIGKILLed, traces lost). num_steps bounds
# the capture to ~N decode steps (~N*25ms of engine time) — enough for kernel
# attribution, 3 orders of magnitude fewer events.
#
# Usage (head node, after the target line is healthy):
#   bash decode_profile_bounded.sh [steps]     # default 40
set -euo pipefail
STEPS="${1:-40}"
OUT=/tmp/decprof
PORT=8000

echo "== starting bounded profile: $STEPS steps"
curl -s -X POST "http://127.0.0.1:$PORT/start_profile" -H 'Content-Type: application/json' \
  -d "{\"output_dir\":\"$OUT\",\"activities\":[\"CPU\",\"GPU\"],\"num_steps\":$STEPS}"
echo

echo "== driving decode traffic (small decode probe; profiler auto-stops at $STEPS steps)"
python3 /tmp/decode600k.py 4 2>&1 | tail -1 || true

sleep 10  # let the trace flush
echo "== collecting traces BEFORE any stop (docker cp race rule)"
docker exec dsv41-head bash -c "ls -la $OUT/ | head -4"
rm -rf /tmp/decprof-traces
docker cp "dsv41-head:$OUT" /tmp/decprof-traces
echo "traces: $(du -sh /tmp/decprof-traces | cut -f1) at /tmp/decprof-traces"
echo "analyze: python3 /tmp/analyze_prof.py /tmp/decprof-traces/<TP-0...>.trace.json.gz 25"
