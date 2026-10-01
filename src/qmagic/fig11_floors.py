"""Fig. 11 (paper Fig. 2): exit-only 15-to-1 output error vs largest patch distance d_X (Litinski model, Gidney-noise X+Z calibration),
overlaid with measured cultivation floors at their physical escape distance d2 (d1=3 and d1=5) and Gidney's d2=15 points."""
from qmagic.paths import find, RESULTS
import json, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
prm = json.load(open(find("params_v3.json")))
fig, ax = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
for i, p in enumerate((1e-3, 5e-4)):
    dx = sorted({s["d_req"] for s in prm["distill"] if s["p"] == p}); fl = [min(s["eps_T"] for s in prm["distill"] if s["p"] == p and s["d_req"] == d) for d in dx]
    ax[i].semilogy(dx, fl, "k-o", label="15-to-1 distillation, best (d_X,d_Z,d_m), exit error (Litinski model)")
    for d1, mk, col in ((3, "s", "C0"), (5, "^", "C3")):
        pts = {}
        for s in prm["cultivate"]:
            if s["p"] == p and s.get("d1") == d1 and not s.get("scaled") and s.get("errors", 0) >= 4:
                pts[s["d2"]] = min(pts.get(s["d2"], 1), s["eps_T"])
        d2s = sorted(pts); ax[i].semilogy(d2s, [pts[d] for d in d2s], mk, color=col, ms=8, ls="--", label=f"cultivation d1={d1}, measured floor at escape distance d2")
    ax[i].axvline(11, color="gray", ls=":", lw=1); ax[i].text(11.2, 3e-3, "d_max = 11", fontsize=8, color="gray")
    ax[i].set_title(f"p = {p:g}"); ax[i].set_xlabel("largest patch distance (d_X for distillation, d2 for cultivation)"); ax[i].grid(alpha=.3, which="both")
ax[0].set_ylabel("error per output T state (exit, before consumption)"); ax[0].legend(fontsize=7, loc="upper right")
fig.tight_layout(); fig.savefig(RESULTS / "fig11_floors.png", dpi=300); print("fig11 written")
