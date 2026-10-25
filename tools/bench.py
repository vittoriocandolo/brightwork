"""Performance check (plan §3, §8.1): render time and memory on this machine.

Each case runs in a fresh interpreter after numpy, scipy and brightwork_synth are
imported and a first render has paid the libraries' first-use costs, so
memory is measured above that baseline. Every timed render uses a take not
rendered before, so it computes its tables, as every new take does; a repeat
of the same take finds them cached, as audition's re-renders do. Both are
reported, and the budget applies to the first. Times include the report.

    python tools/bench.py
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys

HIT = r"""
import dataclasses, json, resource, time
from brightwork_synth import cli
from brightwork_synth.patch import loads, stretch, Noise
from brightwork_synth.render import decimator, render
p = loads(cli.reference_text(%(name)r).replace("sample_rate = 48000", "sample_rate = %(rate)d"))
if %(third)d:  # a third source: a dark wash under the crash
    wash = Noise(-12.0, (300.0, 6000.0), 0.0, 0.3, 1.0, 0.0, 220.0,
                 ((0.0, 0.0), (5.0, 1.0), (400.0, 0.3), (2000.0, 0.05), (4500.0, 0.0)))
    p = dataclasses.replace(p, noise=p.noise + (wash,))
p = stretch(p, "all", %(ms)f / p.length_ms, after_ms=0.0)
decimator(%(base)d)  # loading the shipped coefficients is not part of a render
render(dataclasses.replace(p, take=999))  # first-use costs of the libraries, not of this render
base = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
fresh, cached = 1e9, 1e9
for take in range(1, 1 + %(repeat)d):
    q = dataclasses.replace(p, take=take)
    t0 = time.perf_counter(); audio, report = render(q); fresh = min(fresh, time.perf_counter() - t0)
    t0 = time.perf_counter(); render(q); cached = min(cached, time.perf_counter() - t0)
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
subbands = sum(s.subbands for s in report.sources)
print(json.dumps({"fresh": fresh, "cached": cached, "mib": (peak - base) / 1024, "ms": p.length_ms,
                  "sources": len(p.sources), "subbands": subbands}))
"""

KIT = r"""
import json, resource, time
from brightwork_synth import cli
from brightwork_synth.patch import loads, softer
from brightwork_synth.pack import build
from brightwork_synth.render import decimator
decimator(48000)
groups = [(key, [(f"{name}{db}", softer(loads(cli.reference_text(name)), db)) for db in (0, 8, 16, 24)])
          for key, name in ((42, "closed"), (44, "pedal"), (46, "open"))]
base = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
t0 = time.perf_counter()
kit = build(groups, takes=8, chokes=[(46, (42, 44))])
seconds = time.perf_counter() - t0
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
files = sum(len(f) for k in kit.keys for f in k.audio)
print(json.dumps({"seconds": seconds, "mib": (peak - base) / 1024, "files": files}))
"""

# (label, reference, length in ms, a third source, the budget in s)
CASES = (
    ("closed hat 0.25 s, 2 sources", "closed", 250.0, 0, 0.25),
    ("open hat 1.5 s, 2 sources", "open", 1500.0, 0, 0.5),
    ("crash 5 s, 3 sources", "crash", 5000.0, 1, 1.5),
    ("ride 8 s, 3 sources", "ride", 8000.0, 0, 2.0),
)
KIT_S, MEMORY_MIB = 30.0, 500


def cpu() -> str:
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def run(code: str) -> dict:
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    return json.loads(out.strip().splitlines()[-1])


def main() -> int:
    import os

    import numpy
    import scipy

    print(
        f"machine: {cpu()}, {os.cpu_count()} cores, Python {platform.python_version()}, "
        f"NumPy {numpy.__version__}, SciPy {scipy.__version__}"
    )
    print(f"{'sound':<30s} {'rate':>6s} {'sub-bands':>9s} {'new take':>9s} {'cached':>8s} {'limit':>7s} {'memory':>8s}")
    ok = True
    for label, name, ms, third, limit in CASES:
        for rate in (44100, 48000, 88200, 96000):
            base = 44100 if rate in (44100, 88200) else 48000
            r = run(HIT % {"name": name, "rate": rate, "ms": ms, "third": third, "base": base, "repeat": 3})
            fine = r["fresh"] <= limit and r["mib"] <= MEMORY_MIB
            ok &= fine
            print(
                f"{label:<30s} {rate:6d} {r['subbands']:9d} {1000 * r['fresh']:7.0f}ms {1000 * r['cached']:6.0f}ms "
                f"{1000 * limit:5.0f}ms {r['mib']:5.0f}MiB {'' if fine else 'OVER BUDGET'}"
            )
    r = run(KIT)
    fine = r["seconds"] <= KIT_S and r["mib"] <= MEMORY_MIB
    ok &= fine
    print(
        f"{'kit: 3 keys × 4 layers × 8 takes':<30s} {48000:6d} {'':>9s} {r['seconds']:8.1f}s {'':>8s} {KIT_S:6.0f}s "
        f"{r['mib']:5.0f}MiB {'' if fine else 'OVER BUDGET'}  ({r['files']} files)"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
