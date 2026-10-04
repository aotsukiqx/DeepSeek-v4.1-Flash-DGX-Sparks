#!/usr/bin/env python3
"""Cluster boot hygiene for DSV41 TP4 experiment boots: rustup reaper + orphan check.

Two known stalls eat 10+ minutes each (see the dsv41-serve-boot-triage skill):
  1. the cargo/rustup shim starts a network download inside the scheduler and
     hangs every boot on an offline-ish node  -> kill rustup|cargo every 5 s;
  2. a failed boot leaves orphan scheduler processes holding ~75 GB of unified
     memory, starving the next boot's dequant -> check before serving.

Usage (run on the head, as a background job, BEFORE ./start-tp4.sh serve):
  nohup python3 boot_hygiene.py reap --minutes 12 &
  python3 boot_hygiene.py check
"""
import subprocess
import sys
import time

WORKERS = ["spark-a89f", "spark-141e", "spark-2fe7"]


def sh(cmd, host=None):
    """Run locally, or `bash -s` over ssh with the script fed on stdin (quoting-safe)."""
    if host:
        out = subprocess.run(["ssh", "-o", "BatchMode=yes", "-n", host, "bash", "-s"],
                             input=cmd[2] if len(cmd) > 2 else cmd[0],
                             capture_output=True, text=True)
        return out.stdout + out.stderr
    out = subprocess.run(cmd, capture_output=True, text=True)
    return out.stdout + out.stderr


def reap(minutes):
    deadline = time.time() + minutes * 60
    n = 0
    while time.time() < deadline:
        n += len(sh(["docker", "exec", "dsv41-head", "sh", "-c",
                     "pgrep -f 'cargo|rustup' | xargs -r kill -9"]).strip())
        for w in WORKERS:
            n += len(sh(["docker", "exec", "dsv41-worker", "sh", "-c",
                         "pgrep -f 'cargo|rustup' | xargs -r kill -9"], host=w).strip())
        time.sleep(5)
    print(f"reaper done after {minutes} min ({n} kills)")


def check():
    bad = False
    print("node        MemAvailable_GB   compute_MiB   containers")
    for host in [None] + WORKERS:
        name = "head" if host is None else host
        mem = sh(["bash", "-c", "awk '/MemAvailable/ {print $2/1048576}' /proc/meminfo"], host=host)
        apps = sh(["bash", "-c",
                   "nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits"
                   " | paste -sd+ - | bc"], host=host)
        ctrs = sh(["docker", "ps", "-a", "--format", "{{.Names}}:{{.Status}}"], host=host)
        up = "Up" in ctrs
        print(f"{name:11} {mem.strip():>16} {apps.strip():>13}   "
              f"{ctrs.strip().replace(chr(10), ' ')}")
        try:
            if float(apps.strip() or 0) > 150000 and not up:
                bad = True
        except ValueError:
            pass
    if bad:
        print("ORPHAN GPU MEMORY with no live container: purge before serving "
              "(docker rm -f + pkill -9 -f launch_server)")


if __name__ == "__main__":
    {"reap": reap, "check": check}[sys.argv[1]](*[int(x) for x in sys.argv[2:]])
