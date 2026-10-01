from qmagic.paths import DATA, RESULTS, EXTERNAL, find
import json, math, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, numpy as np
fit = json.load(open(find("params.json")))["pL_fit"]; G, N_e = 100.0, 1e-3
pts = json.load(open(find("gidney2024_fig1_points.json")))
def cult_floor(p):
    e = [x["err"] for x in pts if "This Work" in x["c"] and x["state"] == "T" and x["p"] == p and x["errors"] >= 4]; return min(e) if e else None
from qmagic import factory as F
pL_own = lambda p, d: fit["A"] * (p / fit["p_th"]) ** (fit["s"] * (d + 1) / 2) / d
ps = np.array([5e-4, 7e-4, 1e-3, 1.4e-3, 2e-3, 2.8e-3, 3.43e-3, 4e-3, 4.72e-3])
lam = (fit["p_th"] / ps) ** fit["s"]; d_max = 11; f_T = 0.3
def pL(p, d): return fit["A"] * (p / fit["p_th"]) ** (fit["s"] * (d + 1) / 2)
NA, NC, NB0 = [], [], []
for p in ps:
    dist_floor = min(F.simulate(11, dZ, dm, float(p), pL=pL_own)["p_out"] for dZ in (5, 7) for dm in (5, 7))
    rate = lambda eps: min(pL(p, d) + f_T * eps for d in range(3, d_max + 1, 2))
    NA.append(math.log(G) / 4 / rate(dist_floor)); NB0.append(N_e / rate(dist_floor))
    cf = cult_floor(float(p)); NC.append(math.log(G) / 4 / rate(cf) if cf else float("nan"))
fig, ax = plt.subplots(figsize=(6.5, 4.2))
ax.semilogy(lam, NB0, "s--", label="B0: distillation, no QEM (N_e≤1e-3)")
ax.semilogy(lam, NA, "o-", label="A: distillation + QEM (Γ≤100)")
ax.semilogy(lam, NC, "^-", label="C: cultivation + QEM (Γ≤100)")
ax.axhspan(1e4, 1e10, color="gray", alpha=.12); ax.text(1.6, 3e9, "Suzuki 2022 N = 10^4–10^10", fontsize=8)
for L, name in ((1.56, "Lacroix 2025\nΛ=1.56"), (2.14, "Bluvstein 2026\nΛ=2.14"), (7.3, "p=1e-3\n(Q1/Q5)")):
    ax.axvline(L, color="k", ls=":", lw=.8); ax.text(L * 1.02, 2e-1, name, fontsize=7)
ax.set_xscale("log"); ax.set_xlabel("logical error suppression Λ per +2 in d   (p_eff = p_th / Λ^(1/s))"); ax.set_ylabel("max feasible logical ops N_max (d ≤ 11, f_T=0.3)")
ax.set_ylim(1e-1, 1e10); ax.grid(alpha=.3); ax.legend(fontsize=8, loc="lower right"); ax.set_title("Q3: feasible circuit size vs hardware Λ")
fig.tight_layout(); fig.savefig(RESULTS / "fig5_q3_nmax_vs_lambda.png", dpi=150); print("fig5 written")
