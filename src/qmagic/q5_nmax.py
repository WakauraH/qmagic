"""Q5: how far does cultivation extend the QEM-feasible circuit size?
Closed form from the cost model (costmodel.py):
   QEM arms : N_max = ln(G_max)/4 / min_{d<=d_max, src} [ p_L(d) + f_T eps_T(src) ]
   no-QEM   : N_max = N_e        / min_{d<=d_max, src} [ p_L(d) + f_T eps_T(src) ]
Also the pure Piveteau-style T-count ceiling t_max = ln(G_max)/(4 eps_T) using the best eps_T each source reaches at d<=d_max.
    python q5_nmax.py   -> q5_nmax.csv, fig4_nmax.png, stdout table
"""
from qmagic.paths import DATA, RESULTS, EXTERNAL, find
import json, math, csv, sys
from qmagic.costmodel import pL
PARAMS = sys.argv[1] if len(sys.argv) > 1 else "params.json"; SUF = sys.argv[2] if len(sys.argv) > 2 else ""
prm = json.load(open(find(PARAMS)))
G_max, N_e = 100.0, prm["N_e"]
ARMS = [("B0", "distill", False), ("A", "distill", True), ("B", "cultivate", False), ("C", "cultivate", True)]

def best_rate(kind, p, d_max, f_T):
    """min over d and sources of p_L(d) + f_T*eps_T, with the (d, src) achieving it."""
    best = None
    for d in range(3, d_max + 1, 2):
        for s in prm[kind]:
            if s["p"] != p or s["d_req"] > d_max: continue
            r = pL(prm, p, d) + f_T * s["eps_T"]
            if best is None or r < best[0]: best = (r, d, s["name"], s["eps_T"])
    return best

def eps_floor(kind, p, d_max):
    e = [s["eps_T"] for s in prm[kind] if s["p"] == p and s["d_req"] <= d_max]
    return min(e) if e else float("nan")

rows = []
for p in prm["p"]:
    for d_max in (7, 9, 11, 13, 15):
        for f_T in prm["f_T"]:
            rec = dict(p=p, d_max=d_max, f_T=f_T)
            for a, kind, qem in ARMS:
                b = best_rate(kind, p, d_max, f_T)
                rec[f"Nmax_{a}"] = ((math.log(G_max) / 4 if qem else N_e) / b[0]) if b else float("nan")
                rec[f"d_{a}"] = b[1] if b else None; rec[f"src_{a}"] = b[2] if b else None
            for kind, tag in (("distill", "dist"), ("cultivate", "cult")):
                ef = eps_floor(kind, p, d_max); rec[f"epsfloor_{tag}"] = ef; rec[f"tmax_{tag}"] = math.log(G_max) / (4 * ef)
            rec["ratio_C_over_A"] = rec["Nmax_C"] / rec["Nmax_A"]; rec["ratio_tmax"] = rec["tmax_cult"] / rec["tmax_dist"]
            rows.append(rec)
with open(RESULTS / f"q5_nmax{SUF}.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(f"{'p':>7} {'dmax':>4} {'fT':>4} | {'Nmax B0':>9} {'Nmax A':>9} {'Nmax B':>9} {'Nmax C':>9} | C/A  | eps floor dist/cult | t_max dist/cult (T-only) | ratio")
for r in rows:
    if r["f_T"] != 0.3 and r["d_max"] != 11: continue
    print(f"{r['p']:7g} {r['d_max']:4d} {r['f_T']:4g} | {r['Nmax_B0']:9.2e} {r['Nmax_A']:9.2e} {r['Nmax_B']:9.2e} {r['Nmax_C']:9.2e} | {r['ratio_C_over_A']:4.1f} | {r['epsfloor_dist']:.1e} / {r['epsfloor_cult']:.1e} | {r['tmax_dist']:.1e} / {r['tmax_cult']:.1e} | {r['ratio_tmax']:.0f}")
# figure: N_max vs d_max, fT=0.3
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for i, p in enumerate(prm["p"]):
    for a, _, _ in ARMS:
        rs = [r for r in rows if r["p"] == p and r["f_T"] == 0.3]
        ax[i].semilogy([r["d_max"] for r in rs], [r[f"Nmax_{a}"] for r in rs], "o-", label=a)
    ax[i].axhspan(1e4, 1e10, color="gray", alpha=.12); ax[i].text(7.2, 2e9, "Suzuki 2022\nN = 10^4 – 10^10", fontsize=8)
    ax[i].set_title(f"p={p:g}, f_T=0.3, Γ≤100 / N_e≤1e-3" + (" [correlated decoder]" if SUF else "")); ax[i].set_xlabel("d_max"); ax[i].grid(alpha=.3); ax[i].legend()
ax[0].set_ylabel("max feasible logical ops N_max")
fig.tight_layout(); fig.savefig(RESULTS / f"fig4_nmax{SUF}.png", dpi=150); print(f"fig4_nmax{SUF}.png written")
