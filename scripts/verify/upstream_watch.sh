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

# TLS/connection drops on some networks kill a single-shot curl (rc 35 mid
# handshake, rc 28 timeout) even while the API is healthy — 2026-11-17 lost a
# whole run to it. Retry each leg over plain curl then --tlsv1.2, which is
# enough in practice; a leg that still fails sets FETCH_RC so the run is loud
# instead of silently reporting "nothing new".
FETCH_RC=0
get() {  # get <url> — body on stdout, 0 on 2xx, 1 after all retries
  local url="$1" out code body attempt extra
  for attempt in 1 2 3 4; do
    for extra in "" "--tlsv1.2"; do
      out=$(curl -sS --http1.1 --compressed -m 25 -A dsv41-upstream-watch \
                 -w $'\n%{http_code}' $extra "$url" 2>/dev/null) || continue
      code=${out##*$'\n'}
      body=${out%$'\n'*}
      case "$code" in 2??) printf '%s' "$body"; return 0 ;; esac
    done
    # curl's LibreSSL stack drops the handshake on some networks while python's
    # OpenSSL works (and vice versa), so try urllib before the next round.
    if body=$(python3 - "$url" <<'PY' 2>/dev/null
import sys, urllib.request
try:
    r = urllib.request.urlopen(urllib.request.Request(
        sys.argv[1], headers={"User-Agent": "dsv41-upstream-watch",
                              "Accept": "application/vnd.github+json"}), timeout=25)
    if 200 <= r.status < 300:
        sys.stdout.write(r.read().decode())
        sys.exit(0)
except Exception:
    pass
sys.exit(1)
PY
    ); then
      printf '%s' "$body"
      return 0
    fi
    sleep 2
  done
  return 1
}

need_alert=0
: > "$ALERT.tmp"

for n in $ISSUES; do
  resp=$(get "https://api.github.com/repos/sgl-project/sglang/issues/$n") || { echo "$(ts) issue#$n fetch-failed" >> "$LOG"; FETCH_RC=1; continue; }
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
if ! newc=$(python3 - "$BASE_CUTOFF" <<'PY' 2>/dev/null
import json, subprocess, sys, time, urllib.parse, urllib.request

cutoff = sys.argv[1]
keys = ("dsv4", "dsv4.1", "dsv-v4", "dspark", "engram", "indexer", "verify",
        "sm120", "sm121", "deepseek-v4", "dgx", "spark")
UA = "dsv41-upstream-watch"


def fetch(url):
    """urllib, then curl (plain and --tlsv1.2), 3 rounds — this leg used to die
    silently at a healthy budget whenever the network dropped a handshake."""
    for _round in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            return json.load(urllib.request.urlopen(req, timeout=25))
        except Exception:
            pass
        for extra in ([], ["--tlsv1.2"]):
            r = subprocess.run(["curl", "-sS", "--http1.1", "--compressed",
                                "-m", "25", "-A", UA] + extra + [url],
                               capture_output=True, text=True)
            if r.returncode == 0:
                try:
                    return json.loads(r.stdout)
                except json.JSONDecodeError:
                    pass
        time.sleep(2)
    return None


hits = total = page = 0
while page < 10:
    page += 1
    url = ("https://api.github.com/repos/sgl-project/sglang/commits?"
           + urllib.parse.urlencode({"since": cutoff, "per_page": 100,
                                     "page": page}))
    batch = fetch(url)
    if batch is None:
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
); then
  newc="? ?"
  FETCH_RC=1
fi
echo "$(ts) sglang-main dsv41-relevant-since-watch=${newc%% *} window-commits=${newc##* }" >> "$LOG"
hits=${newc%% *}
if [ "$hits" != "?" ] && [ "$hits" -gt 0 ] 2>/dev/null; then
  echo "$hits DSV41-relevant commit(s) (of ${newc##* } in the window, since watch began) — run the settle sweep" >> "$ALERT.tmp"
  need_alert=1
fi

if [ "$FETCH_RC" != "0" ]; then
  echo "TOOL FAILURE: at least one leg failed to fetch (see $LOG) — the clean-looking result is NOT a clean result; redo this poll by hand over unauthenticated REST" >> "$ALERT.tmp"
  need_alert=1
fi

if [ "$need_alert" = "1" ]; then
  mv "$ALERT.tmp" "$ALERT"
  echo "$(ts) ALERT raised (fetch_rc=$FETCH_RC)" >> "$LOG"
else
  rm -f "$ALERT.tmp" "$ALERT"
fi
exit "$FETCH_RC"
