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

# New commits on main touching DSV4 axes since cutoff. The window has outgrown
# 100 commits, so page until exhausted; counting only page 1 under-reported the
# census (14 vs 18 on 2026-11-10) and is never a complete list anyway.
newc=$(python3 - "$BASE_CUTOFF" <<'PY' 2>/dev/null || echo "? ?"
import json, sys, urllib.parse, urllib.request

cutoff = sys.argv[1]
keys = ("dsv4", "dsv4.1", "dsv-v4", "dspark", "engram", "indexer", "verify",
        "sm120", "sm121", "deepseek-v4", "dgx", "spark")
hits = total = page = 0
while page < 10:
    page += 1
    url = ("https://api.github.com/repos/sgl-project/sglang/commits?"
           + urllib.parse.urlencode({"since": cutoff, "per_page": 100,
                                     "page": page}))
    req = urllib.request.Request(url, headers={"User-Agent": "dsv41-upstream-watch"})
    try:
        batch = json.load(urllib.request.urlopen(req, timeout=25))
    except Exception:
        sys.exit(1)
    for c in batch:
        total += 1
        msg = c["commit"]["message"].splitlines()[0]
        if any(k in msg.lower() for k in keys):
            hits += 1
    if len(batch) < 100:
        break
print(hits, total)
PY
)
echo "$(ts) sglang-main dsv41-relevant-since-watch=${newc%% *} window-commits=${newc##* }" >> "$LOG"
hits=${newc%% *}
if [ "$hits" != "?" ] && [ "$hits" -gt 0 ] 2>/dev/null; then
  echo "$hits DSV41-relevant commit(s) (of ${newc##* } in the window, since watch began) — run the settle sweep" >> "$ALERT.tmp"
  need_alert=1
fi

if [ "$need_alert" = "1" ]; then
  mv "$ALERT.tmp" "$ALERT"
  echo "$(ts) ALERT raised" >> "$LOG"
else
  rm -f "$ALERT.tmp" "$ALERT"
fi
