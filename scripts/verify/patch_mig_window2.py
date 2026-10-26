#!/usr/bin/env python3
"""Patch /tmp/mig_window.sh: capture worker logs as the FIRST rollback action."""
s = open("/tmp/mig_window.sh").read()
# 1) drop the (too-late) capture inside the worker loop
late = """ssh -n "$h" 'docker logs --tail 300 dsv41-worker > /tmp/worker-crash.log 2>&1 || true' >/dev/null 2>&1; """
assert late in s
s = s.replace(late, "")
# 2) insert capture before 'start-tp4.sh stop' inside rollback()
anchor = """  MARK "ROLLBACK to production line"
"""
assert anchor in s
capture = """  MARK "ROLLBACK to production line"
  for h in spark-a89f spark-141e spark-2fe7; do ssh -n "$h" 'docker logs --tail 400 dsv41-worker > /tmp/worker-crash.log 2>&1' >/dev/null 2>&1 || true; done
"""
s = s.replace(anchor, capture, 1)
open("/tmp/mig_window.sh", "w").write(s)
print("worker-log capture moved to first rollback action")
