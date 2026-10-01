"""Fit p_L(d) per noise model from calib_noise.json, rebuild params per model under the v4 conventions (factories recomputed with
that p_L on the full grid; cultivation: measured biased points for bias models at d1=3; measured uniform points otherwise), and compare Q/N_max.
    python noise_costmodel.py -> noise_costmodel.json, fig9_noise_models.png
"""
from qmagic.paths import DATA, RESULTS, EXTERNAL, find
import json, math, copy, numpy as np
from qmagic import costmodel, factory as F
rows = json.load(open(find("calib_noise.json"))); G = 100.0
agg = {}
for r in rows: agg.setdefault((r["model"], r["p"], r["d"]), {})[r["basis"]] = r
def fit(model):
    pts = []
    for (m, p, d), v in agg.items():
        if m != model: continue
        ex, ez = v["X"]["errors"], v["Z"]["errors"]; pl = v["X"]["pL"] + v["Z"]["pL"]
        if ex + ez >= 20 and pl > 0: pts.append((p, d, pl))
    X = np.array([[1.0, (d + 1) / 2, (d + 1) / 2 * math.log(p)] for p, d, _ in pts]); y = np.array([math.log(pl) for *_, pl in pts])
    logA, b, c = np.linalg.lstsq(X, y, rcond=None)[0]; A, pth = math.exp(logA), math.exp(-b / c)
    pred = lambda p, d: A * (p / pth) ** (c * (d + 1) / 2)
    ratio = max(max(pred(p, d) / pl, pl / pred(p, d)) for p, d, pl in pts)
    return dict(A=A, p_th=pth, s=c, n=len(pts), max_ratio=ratio, Lambda_1e3=(pth / 1e-3) ** c)
fits = {m: fit(m) for m in ("uniform", "biased_10", "biased_100", "erasure_0.5", "erasure_0.9")}
print(f"{'model':12} {'A':>7} {'p_th':>8} {'s':>6} {'pts':>3} {'maxratio':>8} {'Λ(1e-3)':>8} {'pL(11,1e-3)':>12}")
for m, f in fits.items(): print(f"{m:12} {f['A']:7.3f} {f['p_th']:8.4f} {f['s']:6.3f} {f['n']:3d} {f['max_ratio']:8.2f} {f['Lambda_1e3']:8.1f} {f['A']*(1e-3/f['p_th'])**(f['s']*6):12.2e}")
# cultivation points under bias (d1=3, d2=7, p=1e-3) from our runs
def thin(curve, minerr=4):
    rs = sorted([x for x in curve if x["errors_raw"] >= minerr], key=lambda x: (x["err"], x["v"])); out, prev = [], None
    for x in rs:
        if prev is not None and x["v"] > prev * 0.9 and x["err"] != 0: continue
        prev = x["v"]; out.append(x)
    return out
bias_runs = json.load(open(find("cultiv_noise_bias.json"))) + json.load(open(find("cultiv_noise_bias2.json")))
def cult_points(model):
    if model.startswith("biased"):
        eta = model.split("_")[1]; out = []
        for r in bias_runs:
            m = r["meta"]
            if m["noise"] != f"bias{eta}" or m["d1"] != 3: continue
            for x in thin(r["curve"]): out.append(dict(name=f"cult_d1=3_d2=7_{model}_att{x['attempts']:.1f}", p=1e-3, d_req=7, eps_T=x["err"], volume_per_T=x["v"], d1=3, d2=7))
        return out
    return [s for s in base["cultivate"] if s["p"] == 1e-3]
# v4: same conventions as params_v3 (unified clock, kappa=1, symmetric exit-only consumption, physical d_req, measured rows,
# full factory grid, bias constraint). Cultivation under erasure is NOT re-simulated: uniform rows are used and the comparison is
# reported as undecided for the erasure models.
v3 = json.load(open(find("params_v3.json"))); U = 1.0
res = {}
for model, f in fits.items():
    prm = copy.deepcopy(v3); prm["pL_fit"] = dict(A=f["A"], p_th=f["p_th"], s=f["s"]); prm["parallelism"] = U * prm["n_L"]
    prm["p"] = [1e-3]; prm["d_max"] = [11]; prm["N"] = [1e4, 1e5, 1e6]; prm["f_T"] = [0.3]; prm["C_max"] = [G]
    pL_own = lambda p, d: f["A"] * (p / f["p_th"]) ** (f["s"] * (d + 1) / 2) / d
    dist = []
    for dX in (5, 7, 9, 11):
        for dZ in (3, 5, 7, 9):
            for dm in (3, 5, 7, 9):
                if dZ > dX or dm > dX: continue
                r = F.simulate(dX, dZ, dm, 1e-3, pL=pL_own, consume=False)
                dist.append(dict(name=f"15to1_{r['dX']},{r['dZ']},{r['dm']}", p=1e-3, d_req=dX, qubits=r["qubits"], cycles_per_T=round(r["cycles"], 1), eps_T=max(r["p_out"], 1e-16)))
    prm["distill"] = dist
    prm["cultivate"] = cult_points(model) if model.startswith("biased") else [c for c in v3["cultivate"] if c["p"] == 1e-3 and not c.get("scaled")]
    sw = costmodel.sweep(prm)
    r = dict(fit=f, u=U, dist_floor=min(x["eps_T"] for x in dist if x["d_req"] <= 11), cult_floor=min(x["eps_T"] for x in prm["cultivate"] if x["d_req"] <= 11),
             NmaxA=costmodel.nmax(prm, "distill", 1e-3, 11, 0.3, G), NmaxC=costmodel.nmax(prm, "cultivate", 1e-3, 11, 0.3, G),
             cultivation_resimulated=model.startswith("biased") or model == "uniform", rows={})
    for x in sw:
        if x["arm"] in ("A", "C"): r["rows"][(x["arm"], x["N"])] = (x["feasible"], x.get("d"), x.get("Q"), x.get("Q_fac"), x.get("Gamma"), x.get("saving_C_vs_A"), x.get("src"))
    res[model] = r
print(f"\n[u=pi/n_L={U:g}, d_max=11, p=1e-3, f_T=0.3, v4 conventions]")
print(f"{'model':12} | {'dist floor':>10} {'cult floor':>10} | {'N_max A':>8} {'N_max C':>8} {'C/A':>5} | saving C vs A at N=1e4/1e5/1e6 (d_A->d_C) | cultivation re-simulated?")
for model, r in res.items():
    sv = []
    for N in (1e4, 1e5, 1e6):
        a, c = r["rows"][("A", N)], r["rows"][("C", N)]
        sv.append(f"{c[5]:+.2f}({a[1]}->{c[1]})" if a[0] and c[0] else ("A✗" if not a[0] else "C✗"))
    print(f"{model:12} | {r['dist_floor']:10.1e} {r['cult_floor']:10.1e} | {r['NmaxA']:8.1e} {r['NmaxC']:8.1e} {r['NmaxC']/r['NmaxA']:5.2f} | " + "  ".join(sv) + f" | {r['cultivation_resimulated']}")
json.dump({m: dict(fit=r["fit"], u=r["u"], dist_floor=r["dist_floor"], cult_floor=r["cult_floor"], NmaxA=r["NmaxA"], NmaxC=r["NmaxC"], cultivation_resimulated=r["cultivation_resimulated"],
                   rows={f"{k[0]}_{k[1]:.0e}": v for k, v in r["rows"].items()}) for m, r in res.items()}, open(RESULTS / "noise_costmodel.json", "w"), indent=1)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6.5, 4.2))
for m in fits:
    pts = sorted([(d, v["X"]["pL"] + v["Z"]["pL"]) for (mm, p, d), v in agg.items() if mm == m and p == 1e-3 and v["X"]["errors"] + v["Z"]["errors"] >= 20])
    ax.semilogy([d for d, _ in pts], [pl for _, pl in pts], "o-", label=f"{m}  (Λ≈{fits[m]['Lambda_1e3']:.0f})")
ax.set_xlabel("d"); ax.set_ylabel("p_L = p_X + p_Z per d rounds (p=1e-3 total)"); ax.set_title("data block under noise models (p = 1e-3 total)"); ax.grid(alpha=.3, which="both"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(RESULTS / "fig9_noise_models.png", dpi=300); print("fig9 written")
