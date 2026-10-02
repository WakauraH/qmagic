"""Recalibration of the data block on the SAME noise engine as cultivation: Gidney's surface-code memory circuit,
gen.NoiseModel.uniform_depolarizing(p) (idle depolarization included), X- and Z-basis memories, rounds=d,
PyMatching 2.4 correlated matching. p_L = p_X + p_Z per d rounds.

Sequential design (default): every point stops at TARGET logical errors per basis (100) or MAXSHOTS; the cheap points
(d <= 7, and d <= 11 at p >= 2e-3) run first and give a provisional fit, which predicts for the remaining points how many
errors MAXSHOTS would yield. Points that cannot reach MIN_ERR (30, the fit's inclusion threshold) are skipped: they cost the
most and never enter the fit. The shipped data/calib_gidney.json was produced with --target 300 --no-adaptive.

    python -m qmagic.calib_gidney [--target 100] [--maxshots 3e7] [--no-adaptive] [--procs 16] [--out calib_gidney_seq.json]
"""
from qmagic.paths import DATA, RESULTS, EXTERNAL, find
import json, time, math, argparse, numpy as np, stim, pymatching
from qmagic import noise as NM
from gen._chunk._noise import NoiseModel
PS, DS = (5e-4, 1e-3, 2e-3, 3e-3), (3, 5, 7, 9, 11)
BATCH, MIN_ERR = 100_000, 30
def fit(rows, min_err=MIN_ERR, pmax=3e-3):
    agg = {}
    for r in rows: agg.setdefault((r["p"], r["d"]), {})[r["basis"]] = r
    pts = [(p, d, v["X"]["pL"] + v["Z"]["pL"], v["X"]["errors"] + v["Z"]["errors"]) for (p, d), v in agg.items() if "X" in v and "Z" in v]
    used = [(p, d, pl) for p, d, pl, ne in pts if ne >= min_err and p <= pmax and pl > 0]
    X = np.array([[1.0, (d + 1) / 2, (d + 1) / 2 * math.log(p)] for p, d, _ in used]); y = np.array([math.log(pl) for *_, pl in used])
    logA, b, c = np.linalg.lstsq(X, y, rcond=None)[0]; A, pth = math.exp(logA), math.exp(-b / c)
    pred = lambda p, d: A * (p / pth) ** (c * (d + 1) / 2)
    return dict(A=A, p_th=pth, exponent_scale=c, n_points=len(used), max_ratio=max(max(pred(p, d) / pl, pl / pred(p, d)) for p, d, pl in used), used=[(p, d) for p, d, _ in used])
def build(key):
    p, d, basis = key; return NM.memory(d, basis, NoiseModel.uniform_depolarizing(p))
def to_rows(rows):
    return [dict(p=k[0], d=k[1], basis=k[2], shots=v["shots"], errors=v["errors"], pL=v["errors"] / max(v["shots"], 1), sec=v["sec"]) for k, v in rows.items()]
def main():
    from qmagic import seqcalib
    ap = argparse.ArgumentParser(); ap.add_argument("--target", type=int, default=100); ap.add_argument("--maxshots", type=float, default=3e7)
    ap.add_argument("--no-adaptive", action="store_true"); ap.add_argument("--procs", type=int, default=16); ap.add_argument("--out", default="calib_gidney_seq.json")
    a = ap.parse_args(); maxshots = int(a.maxshots)
    cheap = [(p, d, b) for p in PS for d in DS for b in ("X", "Z") if d <= 7 or p >= 2e-3]; rest = [(p, d, b) for p in PS for d in DS for b in ("X", "Z") if (p, d, b) not in cheap]
    if a.no_adaptive: cheap, rest = cheap + rest, []
    def predict(closed, key):
        rows = to_rows(closed); agg = {}
        for r in rows: agg.setdefault((r["p"], r["d"]), []).append(r)
        full = [r for v in agg.values() if len(v) == 2 for r in v]
        if sum(1 for v in agg.values() if len(v) == 2 and sum(x["errors"] for x in v) >= MIN_ERR) < 6: return None
        f = fit(full); p, d, _ = key
        return maxshots * f["A"] * (p / f["p_th"]) ** (f["exponent_scale"] * (d + 1) / 2) / 2        # per basis
    rows, skipped, wall = seqcalib.run_points(cheap, rest, build, predict, target=a.target, maxshots=maxshots, min_err=MIN_ERR // 2, procs=a.procs)
    rows = to_rows(rows); f = fit(rows)
    json.dump(dict(model="pL = A*(p/p_th)**(s*(d+1)/2) per d rounds, p_L = p_X + p_Z, Gidney memory circuit, uniform_depolarizing (idle incl.), correlated matching",
                   design=dict(target=a.target, maxshots=maxshots, adaptive=not a.no_adaptive, skipped=[list(k) for k, _ in skipped], core_seconds=sum(r["sec"] for r in rows), wall_seconds=round(wall)),
                   fit=f, rows=rows), open(RESULTS / a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in f.items() if k != "used"}, indent=1)); print(f"core-seconds {sum(r['sec'] for r in rows)}, wall {wall:.0f} s")
    for r in sorted(rows, key=lambda r: (r["p"], r["d"], r["basis"])): print(f"p={r['p']:g} d={r['d']:2} {r['basis']}: shots={r['shots']:9d} err={r['errors']:4d} pL={r['pL']:.2e} {r['sec']}s")
if __name__ == "__main__": main()
