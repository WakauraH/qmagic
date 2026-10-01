"""Gap-informed QEM: (1) universality of eps(gap) across d2 and p; (2) cross-run prediction of the accepted-state error
eps(c) from a run's own gap histogram + a mapping calibrated on a DIFFERENT run; (3) figure.
    python gap_qem.py -> gap_qem.json, fig7_gap_calibration.png
"""
from qmagic.paths import DATA, RESULTS, EXTERNAL, find
import json, collections, math, numpy as np
runs = [r for r in json.load(open(find("cultiv_d2scan_d3_5e7.json"))) if r["meta"]["p"] == 1e-3] + json.load(open(find("cultiv_d2scan_d3_p5e4_3e8.json"))) \
     + json.load(open(find("cultiv_validate_d15_2e7.json"))) + json.load(open(find("cultiv_d2scan_d5_5e8.json")))
def hist(cc):
    C = collections.Counter(); E = collections.Counter()
    for k, n in cc.items(): (E if k[0] == "E" else C)[int(k[1:])] += n
    return C, E
def mapping(cc, bin=1):
    """eps(g) = 2*E/(C+E) per 1-dB gap (S->T doubling), with empirical-Bayes smoothing toward a log-linear fit."""
    C, E = hist(cc); gs = sorted(set(C) | set(E))
    pts = [(g, 2 * E[g] / (C[g] + E[g]), C[g] + E[g], E[g]) for g in gs if C[g] + E[g] > 0]
    # log-linear fit on bins with >= 20 errors
    fit = [(g, e) for g, e, n, ne in pts if ne >= 20 and 0 < e < 1]
    a, b = np.polyfit([g for g, _ in fit], [math.log10(e) for _, e in fit], 1) if len(fit) >= 3 else (float("nan"), float("nan"))
    m = {}
    for g, e, n, ne in pts:
        m[g] = e if ne >= 20 else 10 ** (a * g + b)                     # sparse bins -> fitted curve
    m["_fit"] = (a, b)
    return m, (a, b)
def predict(cc, m):
    """predicted and measured eps at each cutoff c: eps(c) = sum_{g>=c} n_g eps(g) / sum_{g>=c} n_g.
    Gaps absent from the calibrating run fall back to its fitted curve (v4 fix: they were silently counted as error-free)."""
    a, b = m["_fit"]; eps_of = lambda g: m[g] if g in m else 10 ** (a * g + b)
    C, E = hist(cc); gs = sorted(set(C) | set(E), reverse=True); out = []; n = 0; pe = 0.0; ne = 0
    for g in gs:
        n += C[g] + E[g]; ne += E[g]; pe += (C[g] + E[g]) * eps_of(g)
        out.append(dict(cutoff=g, kept=n, eps_pred=pe / n, eps_meas=2 * ne / n, errors=ne))
    return out
res = {}
label = lambda r: f"d1={r['meta']['d1']},d2={r['meta']['d2']},p={r['meta']['p']:g}"
maps = {label(r): mapping(r["raw"]["custom_counts"]) for r in runs}
print("log-linear fit log10 eps = a*gap + b (bins with >=20 errors):")
for k, (m, (a, b)) in maps.items(): print(f"  {k:26}: a={a:+.4f} /dB (10^(-gap/10) would be -0.1), b={b:+.2f}")
print("\ncross-run prediction of eps(c) (mapping from run A applied to run B's gap histogram), at cutoffs giving 1e-5..1e-7 measured error:")
pairs = [("d1=3,d2=15,p=0.001", "d1=3,d2=7,p=0.001"), ("d1=3,d2=15,p=0.001", "d1=3,d2=11,p=0.0005"), ("d1=3,d2=7,p=0.0005", "d1=3,d2=11,p=0.001"),
         ("d1=5,d2=15,p=0.001", "d1=5,d2=11,p=0.0005"), ("d1=5,d2=15,p=0.0005", "d1=5,d2=11,p=0.001")]
byname = {label(r): r for r in runs}
for A, B in pairs:
    m, _ = maps[A]; pr = predict(byname[B]["raw"]["custom_counts"], m)
    rows = [x for x in pr if x["errors"] >= 10 and x["eps_meas"] <= 3e-4]
    sel = rows[::max(1, len(rows) // 5)][:6]
    r4 = [x["eps_pred"] / x["eps_meas"] for x in pr if x["errors"] >= 4 and x["eps_meas"] <= 3e-4]
    res[f"{A}->{B}_ratio_range_err>=4"] = (min(r4), max(r4)) if r4 else None
    print(f"  A={A} -> B={B}:  pred/meas over all cutoffs with >=4 errors: [{min(r4):.2f}, {max(r4):.2f}]")
    for x in sel: print(f"     cutoff>={x['cutoff']:2d} dB: meas {x['eps_meas']:.2e} ({x['errors']} err)  pred {x['eps_pred']:.2e}  ratio {x['eps_pred']/x['eps_meas']:.2f}")
    res[f"{A}->{B}"] = sel
# v4: attainable variance reduction from per-state (gap-conditioned) re-weighting, relative to the mean-channel inverse.
# For a Z-type T-state error eps(g) the single-state fidelity factor is a(g) = 1 - 2 eps(g); an unbiased estimator that
# rescales by a(g)/E[a^2] has variance proportional to 1/E[a^2] instead of 1/E[a]^2, so over N_T independent states the
# best possible ratio is (E[a^2]/E[a]^2)^{N_T}. Evaluated on each run's accepted ensemble at the model's operating cutoffs.
print("\ngap-conditioned re-weighting: max variance ratio (mean-channel / gap-conditioned) for N_T = 3e4 T states, at cutoffs with mean eps <= 1e-5:")
gain = {}
for r in runs:
    C, E = hist(r["raw"]["custom_counts"]); m, _ = maps[label(r)]; a, b = m["_fit"]; eps_of = lambda g: m[g] if g in m else 10 ** (a * g + b)
    gs = sorted(set(C) | set(E), reverse=True); n = 0; s1 = 0.0; s2 = 0.0; best = None
    for g in gs:
        w = C[g] + E[g]; e = eps_of(g); n += w; s1 += w * (1 - 2 * e); s2 += w * (1 - 2 * e) ** 2
        mean_eps = sum((C[h] + E[h]) * eps_of(h) for h in gs if h >= g) / n
        if mean_eps <= 1e-5: best = ((s2 / n) / (s1 / n) ** 2) ** 3e4
    gain[label(r)] = best
    if best: print(f"  {label(r):26}: {best:.6f}")
json.dump(dict(reweighting_gain_NT3e4=gain, maps={k: dict(fit=v[1], eps_by_gap={g: e for g, e in v[0].items() if g != "_fit"}) for k, v in maps.items()}, cross=res), open(RESULTS / "gap_qem.json", "w"), indent=1)
# figure
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(7, 4.5))
for r in runs:
    C, E = hist(r["raw"]["custom_counts"]); gs = sorted(set(C) | set(E))
    xs = [g for g in gs if E[g] >= 4]; ys = [2 * E[g] / (C[g] + E[g]) for g in xs]
    ax.semilogy(xs, ys, "o-" if r["meta"]["d1"] == 3 else "s--", ms=3, lw=1, alpha=.8, label=f"$d_1={r['meta']['d1']}$, $d_2={r['meta']['d2']}$, $p={r['meta']['p']:g}$")
g = np.arange(0, 80); ax.semilogy(g, 10 ** (-g / 10), "k:", label="$10^{-g/10}$")
ax.set_xlabel("complementary gap $g$ (dB)"); ax.set_ylabel("$P(\\mathrm{err}\\,|\\,g)$  ($\\times2$, $S\\to T$)"); ax.set_title("cultivation output: error probability vs decoder gap"); ax.grid(alpha=.3, which="both"); ax.legend(fontsize=6, ncol=2)
fig.tight_layout(); fig.savefig(RESULTS / "fig7_gap_calibration.png", dpi=300); print("fig7 written")
