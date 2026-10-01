"""Phase 1: calibrate p_L(d) = A (p/p_th)^((d+1)/2) for rotated surface code memory (Z basis),
uniform circuit-level depolarizing noise, via Stim + PyMatching.

Run:  python calib_pL.py            -> writes calib_pL.json, calib_pL.png
Self-check: fit reproduces measured p_L within factor 1.5 on every used point.
"""
from qmagic.paths import DATA, RESULTS, EXTERNAL, find
import json, sys, time
from multiprocessing import Pool
import numpy as np, stim, pymatching

PS = [5e-4, 1e-3, 2e-3, 3e-3, 5e-3]
DS = [3, 5, 7, 9, 11, 13]
BATCH, TARGET_ERR, MAX_SHOTS = 100_000, 300, 10_000_000
CORR = "--corr" in sys.argv          # PyMatching 2.4 two-pass correlated matching

def run(cfg):
    p, d = cfg
    c = stim.Circuit.generated("surface_code:rotated_memory_z", distance=d, rounds=d,
        after_clifford_depolarization=p, before_round_data_depolarization=p,
        before_measure_flip_probability=p, after_reset_flip_probability=p)
    dem = c.detector_error_model(decompose_errors=True)
    m = pymatching.Matching.from_detector_error_model(dem, enable_correlations=CORR)
    sampler = c.compile_detector_sampler(seed=int(p*1e6)*100+d)
    shots = errs = 0; t0 = time.time()
    while errs < TARGET_ERR and shots < MAX_SHOTS:
        det, obs = sampler.sample(BATCH, separate_observables=True)
        pred = m.decode_batch(det, enable_correlations=CORR)
        errs += int(np.sum(pred[:, 0] != obs[:, 0])); shots += BATCH
    return dict(p=p, d=d, shots=shots, errors=errs, pL=errs/shots,
                pL_per_round=(errs/shots)/d, sec=round(time.time()-t0, 1))

def fit(rows):
    # log pL = log A + ((d+1)/2) * (log p - log pth); linear in [1, (d+1)/2, (d+1)/2*log p]
    used = [r for r in rows if r["errors"] >= 30 and r["p"] <= 3e-3]
    X = np.array([[1.0, (r["d"]+1)/2, (r["d"]+1)/2*np.log(r["p"])] for r in used])
    y = np.array([np.log(r["pL"]) for r in used])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    # coef = [logA, -log pth * 1, 1] ideally; keep general: logA, b, c  -> pth = exp(-b/c)
    logA, b, c = coef
    A, pth = float(np.exp(logA)), float(np.exp(-b/c))
    pred = lambda p, d: A*(p/pth)**(c*(d+1)/2)
    ratios = [pred(r["p"], r["d"])/r["pL"] for r in used]
    return dict(A=A, p_th=pth, exponent_scale=float(c), n_points=len(used),
                max_ratio=float(max(max(ratios), 1/min(ratios))))

if __name__ == "__main__":
    cfgs = [(p, d) for p in PS for d in DS]
    with Pool(min(16, len(cfgs))) as pool:
        rows = pool.map(run, cfgs)
    f = fit(rows)
    json.dump(dict(decoder="pymatching 2.4 correlated (two-pass)" if CORR else "pymatching uncorrelated", model="pL = A*(p/p_th)**(s*(d+1)/2), per d rounds (memory experiment)",
                   noise="stim surface_code:rotated_memory_z, uniform circuit depolarizing p",
                   fit=f, rows=rows), open(RESULTS / ("calib_pL_corr.json" if CORR else "calib_pL.json"), "w"), indent=1)
    print(json.dumps(f, indent=1))
    assert f["max_ratio"] < 1.5, f"fit off by {f['max_ratio']:.2f}x"
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 4))
        for p in PS:
            pts = [r for r in rows if r["p"] == p and r["errors"] > 0]
            ax.semilogy([r["d"] for r in pts], [r["pL"] for r in pts], "o", label=f"p={p:g}")
            ax.semilogy(DS, [f["A"]*(p/f["p_th"])**(f["exponent_scale"]*(d+1)/2) for d in DS], "-", alpha=.5)
        ax.set_xlabel("d"); ax.set_ylabel("p_L per d rounds"); ax.legend(); fig.tight_layout(); fig.savefig(RESULTS / ("calib_pL_corr.png" if CORR else "calib_pL.png"), dpi=150)
    except Exception as e: print("plot skipped:", e)
