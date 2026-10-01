from qmagic.paths import DATA, RESULTS, EXTERNAL, find
import csv, math, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
rows = list(csv.DictReader(open(find("results.csv"))))
g = lambda r, k: float(r[k]) if r.get(k, "") not in ("", None) else float("nan")
sel = lambda **kw: [r for r in rows if all(str(r[k]) == str(v) for k, v in kw.items())]
# Fig 1: saving 1-Q_C/Q_A vs d_max, per (p, N), fT=0.3, Gmax=100
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for i, p in enumerate(("0.001", "0.0005")):
    for N in ("10000.0", "100000.0", "1000000.0", "100000000.0"):
        rs = sorted(sel(arm="C", p=p, N=N, f_T="0.3", C_max="100.0"), key=lambda r: int(r["d_max"]))
        ax[i].plot([int(r["d_max"]) for r in rs], [100 * g(r, "saving_C_vs_A") for r in rs], "o-", label=f"N={float(N):.0e}")
    ax[i].set_title(f"p={p}: hybrid saving vs QEM-alone, f_T=0.3, Γ≤100"); ax[i].set_xlabel("d_max"); ax[i].set_ylabel("1 − Q_C/Q_A  [%]"); ax[i].legend(); ax[i].grid(alpha=.3)
fig.tight_layout(); fig.savefig(RESULTS / "fig1_saving_vs_dmax.png", dpi=150)
# Fig 2: stacked Q_data / Q_fac per arm at p=5e-4, N=1e5, fT=0.3, dmax=11  and p=1e-3 N=1e5
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for i, p in enumerate(("0.001", "0.0005")):
    arms = ["B0", "A", "B", "C"]; rs = {a: sel(arm=a, p=p, N="100000.0", f_T="0.3", C_max="100.0", d_max="11")[0] for a in arms}
    qd = [g(rs[a], "Q_data") if rs[a]["feasible"] == "True" else 0 for a in arms]; qf = [g(rs[a], "Q_fac") if rs[a]["feasible"] == "True" else 0 for a in arms]
    ax[i].bar(arms, qd, label="Q_data"); ax[i].bar(arms, qf, bottom=qd, label="Q_factory")
    for j, a in enumerate(arms):
        if rs[a]["feasible"] != "True": ax[i].text(j, 1000, "infeasible", ha="center", rotation=90)
        else: ax[i].text(j, qd[j] + qf[j], f"d={int(float(rs[a]['d']))}", ha="center", va="bottom")
    ax[i].set_title(f"p={p}, N=1e5, f_T=0.3, d_max=11"); ax[i].set_ylabel("physical qubits"); ax[i].legend()
fig.tight_layout(); fig.savefig(RESULTS / "fig2_breakdown.png", dpi=150)
# Fig 3: Q x T x Gamma product per arm vs N at dmax=11, fT=0.3, p=5e-4
fig, ax = plt.subplots(figsize=(6, 4))
for a in ("B0", "A", "B", "C"):
    rs = sorted([r for r in sel(arm=a, p="0.0005", f_T="0.3", C_max="100.0", d_max="11") if r["feasible"] == "True"], key=lambda r: float(r["N"]))
    ax.loglog([float(r["N"]) for r in rs], [g(r, "QTG") for r in rs], "o-", label=a)
ax.set_xlabel("N logical ops"); ax.set_ylabel("Q × (N_T·d·Γ cycles)  [qubit·cycles incl. samples]"); ax.set_title("p=5e-4, d_max=11, f_T=0.3, Γ≤100"); ax.legend(); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(RESULTS / "fig3_QTG.png", dpi=150)
print("figs written")
