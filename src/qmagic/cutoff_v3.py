"""Claim (d), corrected comparison: cultivation acceptance cutoff chosen under the QEM budget vs the cheapest cutoff that meets the
SAME N without QEM (not Gidney's error-minimising point). Measured rows only (scaled=False). Unified clock, u = pi/n_L given.
    python -m qmagic.cutoff_v3 [u]
"""
from qmagic.paths import find, RESULTS
import json, math, sys, copy
from qmagic import costmodel
u = float(sys.argv[1]) if len(sys.argv) > 1 else 0.1
prm = json.load(open(find("params_v3.json"))); prm["cultivate"] = [s for s in prm["cultivate"] if not s.get("scaled")]
prm["parallelism"] = u * prm["n_L"]
print(f"u=pi/n_L={u:g}: budget-set cutoff (arm C) vs cheapest no-QEM cutoff (arm B) meeting the same N, f_T=0.3, d_max=11")
for p in (1e-3, 5e-4):
    for N in (1e4, 1e5, 1e6):
        c = costmodel.arm(prm, "cultivate", True, p, 11, N, 0.3, 100.0); b = costmodel.arm(prm, "cultivate", False, p, 11, N, 0.3, 100.0)
        src = lambda r: next(s for s in prm["cultivate"] if s["name"] == r["src"] and s["p"] == p) if r else None   # names are shared across p: match (name, p)
        sc, sb = src(c), src(b)
        print(f"  p={p:g} N={N:.0e}: C(QEM) " + (f"d={c['d']} {sc['name']} att={sc['attempts']} v={sc['volume_per_T']:.0f} Γ={c['Gamma']:.1f}" if c else "infeasible")
              + " | B(no QEM) " + (f"d={b['d']} {sb['name']} att={sb['attempts']} v={sb['volume_per_T']:.0f}" if b else "infeasible")
              + (f" | volume ratio B/C = {sb['volume_per_T']/sc['volume_per_T']:.2f}" if (b and c) else ""))
