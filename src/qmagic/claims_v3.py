"""Re-derive the surviving claims (b) and (c) under the unified clock, separating measured from scaled cultivation rows.
    python -m qmagic.claims_v3  -> results/claims_v3.json + stdout tables
Claim (b): serial-ish architectures (u = pi/n_L small): cultivation adds little to Q, no N_max gain.
Claim (c): parallel (u ~ 1): hybrid saves most of the factory area vs QEM alone; N_max gain modest and conditional.
"""
from qmagic.paths import DATA, RESULTS, find
import json, math, copy
from qmagic import costmodel
prm0 = json.load(open(find("params_v3.json"))); G = 100.0
nmax = lambda prm, kind, p, d_max, f_T: costmodel.nmax(prm, kind, p, d_max, f_T, G)
VARIANTS = (("measured_only", False, {}), ("with_scaled", True, {}),
            ("measured_route", False, {"kappa_mode": "route"}),            # routing tiles charged in proportion to duty cycle u
            ("measured_growth1", False, {"growth_rounds": 1}),            # 1 idle round at d2 before growth (d2 < d rows)
            ("measured_growth3", False, {"growth_rounds": 3}),
            ("measured_growthd", False, {"growth_rounds": "d"}),          # whole consumption at d2: upper variant
            ("measured_route_growthd", False, {"kappa_mode": "route", "growth_rounds": "d"}))
out = {}
for label, keep_scaled, opts in VARIANTS:
    prm = copy.deepcopy(prm0); prm.update(opts)
    if not keep_scaled: prm["cultivate"] = [s for s in prm["cultivate"] if not s.get("scaled")]
    res = {}
    for u in (0.0333, 0.1, 0.3, 1.0):
        prm["parallelism"] = u * prm["n_L"]; prm["p"] = [1e-3, 5e-4]; prm["d_max"] = [11, 13]; prm["N"] = [1e4, 1e5, 1e6]; prm["f_T"] = [0.3]; prm["C_max"] = [G]
        rows = costmodel.sweep(prm)
        q = {}
        for r in rows:
            if r["arm"] in ("A", "C"): q[(r["arm"], r["p"], r["d_max"], r["N"])] = r
        sav, dq = [], []
        for p in (1e-3, 5e-4):
            for N in (1e4, 1e5, 1e6):
                a, c = q[("A", p, 11, N)], q[("C", p, 11, N)]
                if a["feasible"] and c["feasible"]: sav.append(c["saving_C_vs_A"]); dq.append(c["deltaQ_C_minus_A"])
        nm = {f"{p:g}_{dm}": (nmax(prm, "distill", p, dm, 0.3), nmax(prm, "cultivate", p, dm, 0.3)) for p in (1e-3, 5e-4) for dm in (11, 13)}
        res[u] = dict(saving_range=(min(sav), max(sav)) if sav else None, dQ_range=(min(dq), max(dq)) if dq else None, n_feasible=len(sav),
                      nmax={k: dict(A=v[0], C=v[1], ratio=v[1] / v[0]) for k, v in nm.items()})
        print(f"[{label}] u=pi/n_L={u:g} (rho={u*prm['n_L']*0.3:.1f} T/d-cycles): saving={res[u]['saving_range']} dQ={res[u]['dQ_range']} feasible={len(sav)}/6 | N_max C/A: " +
              ", ".join(f"{k}: {v['ratio']:.2f} (A {v['A']:.1e}, C {v['C']:.1e})" for k, v in res[u]["nmax"].items()))
    out[label] = res
json.dump(out, open(RESULTS / "claims_v3.json", "w"), indent=1, default=float)
