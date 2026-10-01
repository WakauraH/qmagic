"""After calib_pL_corr.py: compare fits, build params_corr.json (correlated p_L + factories recomputed with it), rerun Q1 sweep / Q5 / Q3.
    python rerun_corr.py
"""
from qmagic.paths import DATA, RESULTS, EXTERNAL, find
import json, math, csv, copy
from qmagic import factory as F, costmodel
u = json.load(open(find("calib_pL.json"))); c = json.load(open(find("calib_pL_corr.json")))
fu, fc = u["fit"], c["fit"]
lam = lambda f, p: (f["p_th"] / p) ** f["exponent_scale"]
print("fit           A        p_th      s     max_ratio  Λ(1e-3)  Λ(5e-4)   pL(11,1e-3)   pL(11,5e-4)")
for name, f in (("uncorrelated", fu), ("correlated", fc)):
    pl = lambda p, d: f["A"] * (p / f["p_th"]) ** (f["exponent_scale"] * (d + 1) / 2)
    print(f"{name:13} {f['A']:.4f}  {f['p_th']:.5f}  {f['exponent_scale']:.3f}  {f['max_ratio']:.2f}      {lam(f,1e-3):5.2f}    {lam(f,5e-4):5.2f}    {pl(1e-3,11):.2e}     {pl(5e-4,11):.2e}")
print("\nper-config ratio uncorr/corr pL (errors>=30 both):")
ru = {(r["p"], r["d"]): r for r in u["rows"]}; rc = {(r["p"], r["d"]): r for r in c["rows"]}
for k in sorted(ru):
    a, b = ru[k], rc.get(k)
    if b and a["errors"] >= 30 and b["errors"] >= 30: print(f"  p={k[0]:g} d={k[1]:2}: {a['pL']/b['pL']:.2f}  (corr errors {b['errors']}, shots {b['shots']:.0e})")
# params_corr.json
prm = json.load(open(find("params.json"))); prm = copy.deepcopy(prm)
prm["pL_fit"] = dict(A=fc["A"], p_th=fc["p_th"], s=fc["exponent_scale"], source="calib_pL_corr.json (pymatching 2.4 correlated matching)")
pL_own = lambda p, d: fc["A"] * (p / fc["p_th"]) ** (fc["exponent_scale"] * (d + 1) / 2) / d
dist = [s for s in prm["distill"] if s["source"].startswith("Litinski")]
for p in (1e-3, 5e-4):
    for dX in (5, 7, 9, 11, 15, 17, 21, 25):
        rs = [F.simulate(dX, dZ, dm, p, pL=pL_own) for dZ in (3, 5, 7) for dm in (3, 5, 7) if dZ <= dX and dm <= dX]
        for tag, key in (("minerr", lambda r: r["p_out"]), ("minq", lambda r: r["qubits"] * r["cycles"])):
            r = min(rs, key=key)
            dist.append(dict(name=f"15to1_{r['dX']},{r['dZ']},{r['dm']}_{tag}", p=p, d_req=dX, qubits=r["qubits"], cycles_per_T=round(r["cycles"], 1), eps_T=r["p_out"], source="factory15.py (Litinski model, CORRELATED pL calib)"))
prm["distill"] = dist
json.dump(prm, open(RESULTS / "params_corr.json", "w"), indent=1)
print("\ndistillation floor at dX<=11 (correlated pL):", {p: f"{min(s['eps_T'] for s in dist if s['p']==p and s['d_req']<=11):.1e}" for p in (1e-3, 5e-4)})
# Q1 sweep
rows = costmodel.sweep(prm)
with open(RESULTS / "results_corr.csv", "w", newline="") as f:
    keys = sorted({k for r in rows for k in r}); w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
print("\nQ1 sweep with correlated pL (d_max=11, fT=0.3, Gmax=100):")
for r in rows:
    if r["d_max"] == 11 and r["f_T"] == 0.3 and r["C_max"] == 100.0 and r["arm"] in ("A", "C"):
        print(f"  {r['arm']} p={r['p']:g} N={r['N']:.0e}: " + (f"d={r['d']} Q={r['Q']:.0f} (fac {r['Q_fac']:.0f}) Γ={r['Gamma']:.1f}" + (f"  save={r['saving_C_vs_A']:+.2f} dQ={r['deltaQ_C_minus_A']:+.3f}" if r["arm"] == "C" else "") if r["feasible"] else "INFEASIBLE"))
# Q5 N_max
G_max, N_e = 100.0, prm["N_e"]
def best_rate(kind, p, d_max, f_T):
    best = None
    for d in range(3, d_max + 1, 2):
        for s in prm[kind]:
            if s["p"] != p or s["d_req"] > d_max: continue
            r = costmodel.pL(prm, p, d) + f_T * s["eps_T"]
            if best is None or r < best[0]: best = (r, d, s["name"])
    return best
print("\nQ5 N_max with correlated pL (fT=0.3):")
print(f"{'p':>7} {'dmax':>4} | {'B0':>8} {'A':>8} {'B':>8} {'C':>8} | C/A")
for p in (1e-3, 5e-4):
    for d_max in (11, 13, 15):
        v = {}
        for a, kind, qem in costmodel.ARMS:
            b = best_rate(kind, p, d_max, 0.3); v[a] = ((math.log(G_max) / 4 if qem else N_e) / b[0]) if b else float("nan")
        print(f"{p:7g} {d_max:4d} | {v['B0']:8.1e} {v['A']:8.1e} {v['B']:8.1e} {v['C']:8.1e} | {v['C']/v['A']:.1f}")
# Q3 mapping
for L, name in ((1.56, "Lacroix"), (2.14, "Bluvstein"), (3.7, "regime-opening (was p=2e-3)")):
    print(f"Λ={L} ({name}): p_eff uncorr={fu['p_th']/L**(1/fu['exponent_scale']):.2e}  corr={fc['p_th']/L**(1/fc['exponent_scale']):.2e}")
