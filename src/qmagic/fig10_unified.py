"""Fig. 10: claims (b)/(c) under the unified clock — hybrid saving vs QEM-alone and N_max ratio C/A as functions of u = pi/n_L."""
from qmagic.paths import find, RESULTS
import json, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
c = json.load(open(find("claims_v3.json")))
us = sorted(float(u) for u in c["measured_only"])
fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
for lab, ls in (("measured_only", "-"), ("with_scaled", "--")):
    r = c[lab]
    lo = [r[str(u) if str(u) in r else f"{u:g}"]["saving_range"][0] * 100 for u in us]; hi = [r[str(u) if str(u) in r else f"{u:g}"]["saving_range"][1] * 100 for u in us]
    if lab == "measured_only":
        ax[0].fill_between(us, lo, hi, alpha=.25, label="$1-Q_{\\mathrm{C}}/Q_{\\mathrm{A}}$, range over $p$ and $N$"); ax[0].plot(us, hi, "o-"); ax[0].plot(us, lo, "o-")
        dq = [r[str(u) if str(u) in r else f"{u:g}"]["dQ_range"] for u in us]
        ax[0].fill_between(us, [d[0]*100 for d in dq], [d[1]*100 for d in dq], alpha=.25, color="gray", label="$\\Delta Q=(Q_{\\mathrm{A}}-Q_{\\mathrm{C}})/Q_{\\mathrm{B0}}$ [points]")
    for key, mk in (("0.001_11", "o"), ("0.001_13", "s"), ("0.0005_11", "^"), ("0.0005_13", "v")):
        ax[1].plot(us, [r[str(u) if str(u) in r else f"{u:g}"]["nmax"][key]["ratio"] for u in us], mk, ls=ls, label=f"{lab.replace('_',' ')}: $p={key.split('_')[0]}$, $d_{{\\max}}={key.split('_')[1]}$")
# v4: sensitivity band for the N_max ratio (routing-tile error kappa(u) and growth charge for d2 < d), measured rows, p=1e-3
for key, mk in (("0.001_11", "o"), ("0.001_13", "s")):
    lo = [min(c[v][str(u) if str(u) in c[v] else f"{u:g}"]["nmax"][key]["ratio"] for v in ("measured_only", "measured_route", "measured_growth1", "measured_growthd", "measured_route_growthd")) for u in us]
    hi = [max(c[v][str(u) if str(u) in c[v] else f"{u:g}"]["nmax"][key]["ratio"] for v in ("measured_only", "measured_route", "measured_growth1", "measured_growthd", "measured_route_growthd")) for u in us]
    ax[1].fill_between(us, lo, hi, alpha=.15, color="C0" if key.endswith("11") else "C1", label=f"$\\kappa(u)$/growth sensitivity, $p=10^{{-3}}$, $d_{{\\max}}={key.split('_')[1]}$")
ax[0].axhline(10, color="k", ls=":", lw=.8); ax[0].set_xscale("log"); ax[0].set_xlabel("$u=\\pi/n_L$  ($\\rho=n_L u f_T$ $T$ states per step; $f_T=0.3$)"); ax[0].set_ylabel("%  /  points"); ax[0].set_title("qubit saving of hybrid vs mitigation alone ($d_{\\max}=11$)"); ax[0].legend(fontsize=7); ax[0].grid(alpha=.3)
ax[1].axhline(1, color="k", ls=":", lw=.8); ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_xlabel("$u=\\pi/n_L$"); ax[1].set_ylabel("$N_{\\max}(\\mathrm{C})/N_{\\max}(\\mathrm{A})$"); ax[1].set_title("circuit-size ceiling ratio ($d_{\\max}=11$ and $13$)"); ax[1].legend(fontsize=6, loc="upper left"); ax[1].grid(alpha=.3, which="both")
ax[1].set_yticks([0.3, 0.5, 1, 2, 3]); ax[1].set_yticklabels(["0.3", "0.5", "1", "2", "3"]); ax[1].set_ylim(0.25, 4)
fig.tight_layout(); fig.savefig(RESULTS / "fig10_unified_clock.png", dpi=300); print("fig10 written")
