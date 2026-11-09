#!/usr/bin/env bash
# upstream_watch.sh — automatic upstream follow-up for the DSV41 fleet.
# Checks (GitHub REST, unauthenticated): issue activity on our two filed issues
# + new sglang main commits touching the six DSV41 axes. Appends one line per
# run to $LOG; when action is needed, writes $ALERT (any session should check
# for its existence and act; the file body says what to do).
set -uo pipefail
ISSUES="49173 49174"
LOG=/tmp/upstream-watch.log
ALERT=/tmp/dsv41-upstream-alert
BASE_CUTOFF="2026-11-08T14:00:00Z"   # when our watch began
AXES='python/sglang/srt/layers/engram.py python/sglang/srt/speculative/ python/sglang/srt/mem_cache/ python/sglang/srt/layers/attention/deepseek_v4_backend.py python/sglang/srt/models/deepseek_v4.py'
ts() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }

need_alert=0
: > "$ALERT.tmp"

for n in $ISSUES; do
  resp=$(curl -sf -m 20 "https://api.github.com/repos/sgl-project/sglang/issues/$n" 2>/dev/null) || { echo "$(ts) issue#$n fetch-failed" >> "$LOG"; continue; }
  comments=$(echo "$resp" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("comments",0))' 2>/dev/null || echo "?")
  state=$(echo "$resp" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("state","?"))' 2>/dev/null)
  updated=$(echo "$resp" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("updated_at","?"))' 2>/dev/null)
  echo "$(ts) issue#$n state=$state comments=$comments updated=$updated" >> "$LOG"
  if [ "$comments" != "0" ] && [ "$comments" != "?" ] && [ "$comments" != "0" ]; then
    echo "ISSUE #$n has $comments comments (updated $updated) — read maintainer response: https://github.com/sgl-project/sglang/issues/$n" >> "$ALERT.tmp"
    need_alert=1
  fi
  if [ "$state" = "closed" ]; then
    echo "ISSUE #$n CLOSED — check resolution (fixed? wontfix?) and whether the bp6 patch chain can collapse" >> "$ALERT.tmp"
    need_alert=1
  fi
done

# new commits on main touching DSV41 axes since cutoff
newc=$(curl -sf -m 20 "https://api.github.com/repos/sgl-project/sglang/commits?since=$BASE_CUTOFF&per_page=100" 2>/dev/null \
  | python3 -c '
import json,sys
try: data=json.load(sys.stdin)
except Exception: sys.exit(0)
axes=("engram.py","speculative/","mem_cache/","deepseek_v4_backend.py","deepseek_v4.py","deepseek_v4_dspark")
n=0
for c in data:
    msg=c["commit"]["message"].splitlines()[0]
    # keyword census: include the naming variants actually used upstream
    keys = ("dsv4","dsv4.1","dsv-v4","dspark","engram","indexer","verify",
            "sm120","sm121","deepseek-v4","dgx","spark")
    if any(k in msg.lower() for k in keys):
        n+=1
print(n)' 2>/dev/null || echo "?")
echo "$(ts) sglang-main dsv41-relevant-since-watch=$newc" >> "$LOG"
if [ "$newc" != "?" ] && [ "$newc" -gt 0 ]; then
  echo "$newc DSV41-relevant commit(s) landed on sglang main since watch began — run the settle sweep" >> "$ALERT.tmp"
  need_alert=1
fi

if [ "$need_alert" = "1" ]; then
  mv "$ALERT.tmp" "$ALERT"
  echo "$(ts) ALERT raised" >> "$LOG"
else
  rm -f "$ALERT.tmp" "$ALERT"
fi
