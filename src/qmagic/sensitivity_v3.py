"""Sensitivity of the v3 (single-clock) results to n_L, f_T, r_route, Gamma_max, plus a worked example. Measured cultivation rows only.
    python -m qmagic.sensitivity_v3 -> results/sensitivity_v3.json + stdout"""
from qmagic.paths import find, RESULTS
import json, math, copy
from qmagic import costmodel
base = json.load(open(find("params_v3.json"))); base["cultivate"] = [s for s in base["cultivate"] if not s.get("scaled")]
def sweep(prm, u, N=(1e4, 1e5, 1e6), f_T=0.3, G=100.0):
    prm = copy.deepcopy(prm); prm["parallelism"] = u * prm["n_L"]; prm["p"] = [1e-3, 5e-4]; prm["d_max"] = [11]; prm["N"] = list(N); prm["f_T"] = [f_T]; prm["C_max"] = [G]
    rows = costmodel.sweep(prm); sav = [r["saving_C_vs_A"] for r in rows if r["arm"] == "C" and r["feasible"] and not math.isnan(r["saving_C_vs_A"])]
    dq = [r["deltaQ_C_minus_A"] for r in rows if r["arm"] == "C" and r["feasible"] and not math.isnan(r["deltaQ_C_minus_A"])]
    return (min(sav), max(sav), len(sav)) if sav else None, (min(dq), max(dq)) if dq else None
out = {}
print("saving 1-Q_C/Q_A (min,max,n feasible) | dQ (min,max) at d_max=11, over p in {1e-3,5e-4}, N in {1e4,1e5,1e6}")
for u in (1/30, 0.1, 1.0):
    for n_L in (20, 50, 100, 300):
        prm = copy.deepcopy(base); prm["n_L"] = n_L; s, d = sweep(prm, u); out[f"u={u:.3g},nL={n_L}"] = (s, d); print(f"  u={u:.3g} n_L={n_L:4d}: saving={s} dQ={d}")
    for f_T in (0.1, 1.0):
        s, d = sweep(base, u, f_T=f_T); out[f"u={u:.3g},fT={f_T}"] = (s, d); print(f"  u={u:.3g} f_T={f_T}: saving={s} dQ={d}")
    for r in (1.5, 3.0):
        prm = copy.deepcopy(base); prm["r_route"] = r; s, d = sweep(prm, u); out[f"u={u:.3g},r={r}"] = (s, d); print(f"  u={u:.3g} r_route={r}: saving={s} dQ={d}")
    for G in (10.0, 1e4):
        s, d = sweep(base, u, G=G); out[f"u={u:.3g},G={G:g}"] = (s, d); print(f"  u={u:.3g} Gamma_max={G:g}: saving={s} dQ={d}")
    # v4: precision target / calibration error (bias Sigma*bias_r <= N_e on QEM arms; Sigma <= N_e on the others)
    for tag, opts in (("r=1e-2", dict(bias_r=1e-2)), ("Ne=1e-2,r=1e-2", dict(N_e=1e-2, bias_r=1e-2)), ("nobias", dict(bias_r=None))):
        prm = copy.deepcopy(base); prm.update(opts); s, d = sweep(prm, u); out[f"u={u:.3g},{tag}"] = (s, d); print(f"  u={u:.3g} {tag}: saving={s} dQ={d}")
    # v4: routing-tile error kappa(u) and growth charge for d2 < d
    for tag, opts in (("kappa=route", dict(kappa_mode="route")), ("growth=1", dict(growth_rounds=1)), ("growth=d", dict(growth_rounds="d"))):
        prm = copy.deepcopy(base); prm.update(opts); s, d = sweep(prm, u); out[f"u={u:.3g},{tag}"] = (s, d); print(f"  u={u:.3g} {tag}: saving={s} dQ={d}")
# v4: spacetime objective (qubits x runtime x samples) over the same grid: ratio QTG(C)/QTG(A) where both arms exist
print("\nspacetime objective QTG = Q * T_cycles (samples included): C/A over p in {1e-3,5e-4}, N in {1e4,1e5,1e6}, d_max=11, f_T=0.3")
qtg = {}
for u in (1/30, 0.1, 0.3, 1.0):
    prm = copy.deepcopy(base); prm["parallelism"] = u * prm["n_L"]; rat = []
    for p in (1e-3, 5e-4):
        for N in (1e4, 1e5, 1e6):
            a = costmodel.arm(prm, "distill", True, p, 11, N, 0.3, 100.0, objective="QTG"); c = costmodel.arm(prm, "cultivate", True, p, 11, N, 0.3, 100.0, objective="QTG")
            if a and c: rat.append(c["QTG"] / a["QTG"])
    qtg[f"u={u:.3g}"] = dict(n=len(rat), min=min(rat), max=max(rat), n_C_better=sum(r < 1 for r in rat)); print(f"  u={u:.3g}: {len(rat)} points, QTG_C/QTG_A in [{min(rat):.2f}, {max(rat):.2f}], C better at {sum(r<1 for r in rat)}")
out["qtg"] = qtg
print("\nworked example p=1e-3, N=1e5, f_T=0.3, Gamma<=100, d_max=11:")
for u in (1/30, 0.1, 1.0):
    prm = copy.deepcopy(base); prm["parallelism"] = u * prm["n_L"]
    for a, kind, qem in costmodel.ARMS:
        r = costmodel.arm(prm, kind, qem, 1e-3, 11, 1e5, 0.3, 100.0)
        print(f"  u={u:.3g} {a:2}: " + (f"d={r['d']} Q={r['Q']:7.0f} = data {r['Q_data']:6.0f} + supply {r['Q_fac']:6.0f} (k={r['k']}) | Gamma={r['Gamma']:6.1f} | src={r['src']} | T_cycles={r['T_cycles']:.2e} QTG={r['QTG']:.2e}" if r else "infeasible"))
    out[f"example_u={u:.3g}"] = {a: (costmodel.arm(prm, k, q, 1e-3, 11, 1e5, 0.3, 100.0) or None) for a, k, q in costmodel.ARMS}
json.dump(out, open(RESULTS / "sensitivity_v3.json", "w"), indent=1, default=float)
