"""Per-p Poisson maximum-likelihood fits of p_L(d) = A x^{(d+1)/2} from calib_gidney.json (X+Z, all d with >=1 error), and the
resulting d_X <= 11 distillation exit floor. Sensitivity of claim (a) to the common-s fit; also reports the tension at (p=1e-3, d=11).
    python -m qmagic.calib_perp -> results/calib_perp.json"""
from qmagic.paths import find, RESULTS
import json, math, numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson
from qmagic import factory as F
cal = json.load(open(find("calib_gidney.json"))); glob = cal["fit"]
agg = {}
for r in cal["rows"]: a = agg.setdefault((r["p"], r["d"]), dict(shots=0, errors=0)); a["shots"] = r["shots"]; a["errors"] += r["errors"]
pts = {p: sorted((d, v["shots"], v["errors"]) for (pp, d), v in agg.items() if pp == p) for p in sorted({p for p, _ in agg})}
def fit_p(rows):
    def nll(th):
        lnA, lnx = th; return sum(-poisson.logpmf(e, n * math.exp(lnA + lnx * (d + 1) / 2)) for d, n, e in rows)
    th = minimize(nll, [math.log(0.3), math.log(0.1)], method="Nelder-Mead").x
    return math.exp(th[0]), math.exp(th[1])
out = {}
for p, rows in pts.items():
    A, x = fit_p(rows); pL = lambda pp, d: A * x ** ((d + 1) / 2) / d         # per cycle, for the factory
    gl = lambda pp, d: glob["A"] * (pp / glob["p_th"]) ** (glob["exponent_scale"] * (d + 1) / 2) / d
    rec = dict(A=A, x=x, Lambda=1 / x, pL11_perp=A * x ** 6, pL11_global=glob["A"] * (p / glob["p_th"]) ** (glob["exponent_scale"] * 6),
               points=[dict(d=d, shots=n, errors=e, expected_perp=n * A * x ** ((d + 1) / 2), expected_global=n * gl(p, d) * d) for d, n, e in rows])
    if p in (1e-3, 5e-4):
        floor = lambda f: min(F.simulate(11, dZ, dm, p, pL=f, consume=False)["p_out"] for dZ in (3, 5, 7, 9) for dm in (3, 5, 7, 9))
        rec["floor11_perp"], rec["floor11_global"] = floor(pL), floor(gl)
    out[f"{p:g}"] = rec
    print(f"p={p:g}: per-p fit A={A:.3f} x={x:.4f} (Lambda={1/x:.2f}); p_L(11) per-p {rec['pL11_perp']:.2e} vs global {rec['pL11_global']:.2e}"
          + (f"; d_X<=11 floor per-p {rec['floor11_perp']:.2e} vs global {rec['floor11_global']:.2e}" if "floor11_perp" in rec else ""))
    for q in rec["points"]:
        if q["d"] >= 9: print(f"    d={q['d']}: observed {q['errors']} errors, expected per-p {q['expected_perp']:.1f}, global {q['expected_global']:.1f}")
json.dump(out, open(RESULTS / "calib_perp.json", "w"), indent=1)
