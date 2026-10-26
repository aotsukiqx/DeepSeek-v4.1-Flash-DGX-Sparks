#!/usr/bin/env python3
"""Patch /tmp/mig_window.sh: capture worker container logs before rollback removes them."""
s = open("/tmp/mig_window.sh").read()
old = """for h in spark-a89f spark-141e spark-2fe7; do ssh -n "$h" 'docker ps -aq --filter name=dsv41-worker | xargs -r docker rm -f' >/dev/null 2>&1; done"""
new = """for h in spark-a89f spark-141e spark-2fe7; do ssh -n "$h" 'docker logs --tail 300 dsv41-worker > /tmp/worker-crash.log 2>&1 || true' >/dev/null 2>&1; ssh -n "$h" 'docker ps -aq --filter name=dsv41-worker | xargs -r docker rm -f' >/dev/null 2>&1; done"""
n = s.count(old)
assert n >= 1, "pattern not found"
open("/tmp/mig_window.sh", "w").write(s.replace(old, new))
print(f"patched {n} occurrence(s)")
