"""Q3: what does the model say at hardware-realistic Lambda?
Maps experimental Λ -> p_eff via the v3 fit, builds distillation (factory, full grid, exit-only) + cultivation (Gidney 2e-3 data)
sources at p_eff, and reports N_max per arm (costmodel.nmax, u=1) plus the data-block-only bound, and the 4-arm sweep at N in {1e3,1e4}.
    python q3_lambda.py -> q3_nmax.csv, stdout
"""
from qmagic.paths import DATA, RESULTS, EXTERNAL, find
import json, math, csv
from qmagic import factory as F, costmodel
base = json.load(open(find("params_v3.json"))); fit = base["pL_fit"]   # v3: Gidney-noise X+Z calibration
pL_own = lambda p, d: fit["A"] * (p / fit["p_th"]) ** (fit["s"] * (d + 1) / 2) / d      # per cycle, for factory15
# Experimental Λ are per-cycle (Lacroix Λ(3/5); Bluvstein: per-round in a 4-round circuit, d=3 vs 5). The model's Λ is the ratio of
# p_L per d rounds, so Λ_block(d->d+2) = Λ_cycle * d/(d+2)  (panel fix 6). For the 3->5 pair: x 3/5.
# v4: Google Willow (surface code, d=3/5/7, Lambda_cycle = 2.14 +- 0.02) added; its 5->7 block conversion is x5/7. The numerical
# coincidence with Bluvstein's 2.14(13) is accidental (different platform, code and circuit).
LAMBDAS_CYCLE = {"Lacroix2025_SC_colour_code": (1.56, 3), "Bluvstein2026_neutral_atoms": (2.14, 3), "GoogleWillow2025_surface_code": (2.14, 5)}
LAMBDAS = {k: v * d / (d + 2) for k, (v, d) in LAMBDAS_CYCLE.items()}          # block-basis values: 0.94, 1.28, 1.53
p_effs = {k: fit["p_th"] / lam ** (1 / fit["s"]) for k, lam in LAMBDAS.items()}
print("Λ (per-cycle -> per-d-rounds block, 3->5):", {k: round(v, 3) for k, v in LAMBDAS.items()})
p_list = [2e-3] + sorted(p_effs.values())
prm = dict(base); prm["p"] = p_list; prm["d_max"] = [11, 13, 15]; prm["N"] = [1e3, 1e4]; prm["f_T"] = [0.3]; prm["C_max"] = [100.0]
prm["parallelism"] = prm["n_L"]            # u = 1: the most optimistic (idle-free) clock; serial machines are worse
# distillation at each p: full (dZ, dm) grid per dX <= 15, exit-only p_out (v4 conventions). Factory rows whose density-matrix
# output is non-physical (p_out <= 0 or >= 0.5, cycles <= 0: the 15-to-1 model is outside its domain above threshold) are dropped.
dist = []; dropped = 0
for p in p_list:
    for dX in (5, 7, 9, 11, 13, 15):
        for dZ in (3, 5, 7, 9):
            for dm in (3, 5, 7, 9):
                if dZ > dX or dm > dX: continue
                r = F.simulate(dX, dZ, dm, p, pL=pL_own, consume=False)
                if not (0 < r["p_out"] < 0.5) or r["cycles"] <= 0 or r["qubits"] <= 0: dropped += 1; continue
                dist.append(dict(name=f"15to1_{r['dX']},{r['dZ']},{r['dm']}", p=p, d_req=dX, qubits=r["qubits"], cycles_per_T=round(r["cycles"], 1), eps_T=r["p_out"]))
prm["distill"] = dist; print(f"{len(dist)} physical factory rows, {dropped} non-physical rows dropped")
# cultivation: Gidney 2024 Fig.1 rows at p=2e-3 (errors >= 4)
pts = json.load(open(find("gidney2024_fig1_points.json")))
prm["cultivate"] = [dict(name=f"cult_d1={x['d1']}_att{x['attempts']:.0f}", p=x["p"], d_req=15, eps_T=x["err"], volume_per_T=x["v"], estimate=False)
                    for x in pts if "This Work" in x["c"] and x["state"] == "T" and x["p"] == 2e-3 and x["errors"] >= 4]
json.dump(prm, open(RESULTS / "params_q3.json", "w"), indent=1)
# N_max per arm (costmodel.nmax: same clock/consumption/bias rules as the sweep) and a data-block-only upper bound (T errors = 0):
# N_data = budget / min_d p_L(d) with budget = min(ln G/4, N_e/bias_r). No magic-state source can beat it.
G_max = 100.0
budget = min(math.log(G_max) / 4, prm["N_e"] / prm["bias_r"]) if prm.get("bias_r") else math.log(G_max) / 4
out = []
print(f"{'p_eff':>8} {'Λ':>5} {'dmax':>4} | {'N_data':>8} {'Nmax B0':>8} {'Nmax A':>8} {'Nmax B':>8} {'Nmax C':>8}")
for p in p_list:
    lam = (fit["p_th"] / p) ** fit["s"]
    for d_max in prm["d_max"]:
        rec = dict(p=p, Lambda=lam, d_max=d_max, N_data=budget / min(costmodel.pL(prm, p, d) for d in range(3, d_max + 1, 2)))
        for a, kind, qem in costmodel.ARMS:
            rec[f"Nmax_{a}"] = costmodel.nmax(prm, kind, p, d_max, 0.3, G_max, qem=qem)
        out.append(rec)
        print(f"{p:8.2e} {lam:5.2f} {d_max:4d} | {rec['N_data']:8.1e} {rec['Nmax_B0']:8.1e} {rec['Nmax_A']:8.1e} {rec['Nmax_B']:8.1e} {rec['Nmax_C']:8.1e}")
with open(RESULTS / "q3_nmax.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
print("\n4-arm sweep at N=1e3/1e4, fT=0.3, Gmax=100:")
rows = costmodel.sweep(prm)
for r in rows:
    if r["d_max"] != 11: continue
    print(f"{r['arm']:2} p={r['p']:.2e} N={r['N']:.0e}: " + (f"d={r['d']} Q={r['Q']:.0f} (fac {r['Q_fac']:.0f}) Γ={r['Gamma']:.1f} src={r['src']}" if r["feasible"] else "INFEASIBLE"))
