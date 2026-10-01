"""Recalibration of the data block on the SAME noise engine as cultivation: Gidney's surface-code memory circuit,
gen.NoiseModel.uniform_depolarizing(p) (idle depolarization included), X- and Z-basis memories, rounds=d,
PyMatching 2.4 correlated matching. p_L = p_X + p_Z per d rounds. -> results/calib_gidney.json
"""
from qmagic.paths import DATA, RESULTS, EXTERNAL, find
import json, time, math, numpy as np, stim, pymatching
from multiprocessing import Pool
from qmagic import noise as NM
from gen._chunk._noise import NoiseModel
PS, DS = (5e-4, 1e-3, 2e-3, 3e-3), (3, 5, 7, 9, 11)
BATCH, TARGET, MAXSHOTS = 100_000, 300, 30_000_000
def run(cfg):
    p, d, basis = cfg; c = NM.memory(d, basis, NoiseModel.uniform_depolarizing(p))
    m = pymatching.Matching.from_detector_error_model(c.detector_error_model(decompose_errors=True), enable_correlations=True)
    s = c.compile_detector_sampler(seed=int(p * 1e6) * 1000 + d * 10 + (basis == "X")); shots = errs = 0; t0 = time.time()
    while errs < TARGET and shots < MAXSHOTS:
        det, obs = s.sample(BATCH, separate_observables=True)
        errs += int(np.sum(m.decode_batch(det, enable_correlations=True)[:, 0] != obs[:, 0])); shots += BATCH
    return dict(p=p, d=d, basis=basis, shots=shots, errors=errs, pL=errs / shots, sec=round(time.time() - t0))
def fit(rows, min_err=30, pmax=3e-3):
    agg = {}
    for r in rows: agg.setdefault((r["p"], r["d"]), {})[r["basis"]] = r
    pts = [(p, d, v["X"]["pL"] + v["Z"]["pL"], v["X"]["errors"] + v["Z"]["errors"]) for (p, d), v in agg.items() if "X" in v and "Z" in v]
    used = [(p, d, pl) for p, d, pl, ne in pts if ne >= min_err and p <= pmax and pl > 0]
    X = np.array([[1.0, (d + 1) / 2, (d + 1) / 2 * math.log(p)] for p, d, _ in used]); y = np.array([math.log(pl) for *_, pl in used])
    logA, b, c = np.linalg.lstsq(X, y, rcond=None)[0]; A, pth = math.exp(logA), math.exp(-b / c)
    pred = lambda p, d: A * (p / pth) ** (c * (d + 1) / 2)
    return dict(A=A, p_th=pth, exponent_scale=c, n_points=len(used), max_ratio=max(max(pred(p, d) / pl, pl / pred(p, d)) for p, d, pl in used), used=[(p, d) for p, d, _ in used])
if __name__ == "__main__":
    cfgs = [(p, d, b) for p in PS for d in DS for b in ("X", "Z")]
    with Pool(16) as pool: rows = pool.map(run, cfgs)
    f = fit(rows)
    json.dump(dict(model="pL = A*(p/p_th)**(s*(d+1)/2) per d rounds, p_L = p_X + p_Z, Gidney memory circuit, uniform_depolarizing (idle incl.), correlated matching", fit=f, rows=rows), open(RESULTS / "calib_gidney.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in f.items() if k != "used"}, indent=1))
    for r in sorted(rows, key=lambda r: (r["p"], r["d"], r["basis"])): print(f"p={r['p']:g} d={r['d']:2} {r['basis']}: shots={r['shots']:9d} err={r['errors']:4d} pL={r['pL']:.2e} {r['sec']}s")
