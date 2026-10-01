"""Panel-corrected parameter set (v3; v4 = full factory grid + bias constraint): Gidney-noise X+Z calibration, factory grid over ALL odd d_X (5..25) with exit-only p_out,
physical d_req for every cultivation row, symmetric consumption 0.75*p_L(d) on both arms, unified clock (rho = pi*f_T, kappa=1).
    python -m qmagic.build_params_v3  -> results/params_v3.json (+ prints factory floors)
"""
from qmagic.paths import DATA, RESULTS, EXTERNAL, find
import json, glob
from qmagic import factory as F
fit = json.load(open(find("calib_gidney.json")))["fit"]
pL_cycle = lambda p, d: fit["A"] * (p / fit["p_th"]) ** (fit["exponent_scale"] * (d + 1) / 2) / d      # per cycle (X+Z)
base = json.load(open(find("params_corr_d2.json")))
prm = dict(base); prm["pL_fit"] = dict(A=fit["A"], p_th=fit["p_th"], s=fit["exponent_scale"], source="calib_gidney.json (Gidney memory circuit, uniform_depolarizing incl. idle, X+Z, correlated matching)")
prm.update(unified_clock=True, kappa=1.0, consumption=True, parallelism=None,
           bias_r=1e-3, kappa_mode="fixed", growth_rounds=0)   # v4: common precision target N_e for all arms (QEM bias Sigma*bias_r <= N_e); kappa(u) and growth are sensitivities
prm["_note"] = "v3 (adversarial-review fixes): exit-only factory p_out; symmetric consumption; physical d_req; unified clock; Gidney-noise X+Z p_L. Rows with scaled=True are UNMEASURED extrapolations."
# distillation: exit-only p_out, all odd dX, Litinski's half-weights use pL(X+Z) -> pass pL_cycle/2 per type? Litinski's rule: X and Z each 0.5*pL where pL is the TOTAL per-cycle logical rate -> pass the total.
dist = []
for p in (1e-3, 5e-4):
    for dX in range(5, 27, 2):
        rs = [F.simulate(dX, dZ, dm, p, pL=pL_cycle, consume=False) for dZ in (3, 5, 7, 9) for dm in (3, 5, 7, 9) if dZ <= dX and dm <= dX]
        for r in rs:                                   # v4: keep the FULL (dZ, dm) grid; the sweep picks the Q-optimal factory under the error constraint
            dist.append(dict(name=f"15to1_{r['dX']},{r['dZ']},{r['dm']}", p=p, d_req=dX, qubits=r["qubits"], cycles_per_T=round(r["cycles"], 1), eps_T=max(r["p_out"], 1e-16),
                             source="factory.simulate(consume=False), Gidney-noise X+Z calibration"))
    print(f"p={p:g}: exit-only distillation floor per dX: " + ", ".join(f"{dX}: {min(s['eps_T'] for s in dist if s['p']==p and s['d_req']==dX):.1e}" for dX in range(5, 27, 2)))
prm["distill"] = dist
# cultivation: physical d_req = d2 for every row; keep scaled rows flagged
cult = []
for s in base["cultivate"]:
    s = dict(s); s["d_req"] = int(s.get("d2") or 15); cult.append(s)
# deep-tail d1=5, d2=11 reruns (5e9 / 2e9 shots): measured rows, physical d_req=11
def thin(curve, minerr=4):
    rs = sorted([x for x in curve if x["errors_raw"] >= minerr], key=lambda x: (x["err"], x["v"])); out, prev = [], None
    for x in rs:
        if prev is not None and x["v"] > prev * 0.9 and x["err"] != 0: continue
        prev = x["v"]; out.append(x)
    return out
for f in ("cultiv_d5deep_d2_11_p1e3_5e9.json", "cultiv_d5deep_d2_11_p5e4_2e9.json"):
    try: runs = json.load(open(find(f)))
    except FileNotFoundError: continue
    for r in runs:
        m = r["meta"]
        for x in thin(r["curve"]):
            cult.append(dict(name=f"cult_d1=5_d2=11_deep_att{x['attempts']:.1f}", p=m["p"], d_req=11, eps_T=x["err"], volume_per_T=x["v"], estimate=False, scaled=False,
                             attempts=round(x["attempts"], 2), errors=x["errors_raw"], d1=5, d2=11, q=r["q"], source=f"deep-tail rerun {f} (5e9/2e9 shots), errors x2 (S->T)"))
prm["cultivate"] = cult
json.dump(prm, open(RESULTS / "params_v3.json", "w"), indent=1)
print(len(dist), "distillation rows;", len(cult), "cultivation rows (", sum(1 for s in cult if s.get("scaled")), "scaled )")
