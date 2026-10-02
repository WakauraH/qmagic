"""Importance sampling for the data-block calibration: sample the detector error model with every mechanism's probability
inflated by lambda, decode with the SAME correlated matcher, and reweight each shot by the exact likelihood ratio.
Reports, per (p, d, lambda): estimate of p_L, its variance per shot, and the shots / wall time needed for 10% relative
precision, against plain sampling (lambda = 1).  Needs external/ (Gidney memory circuit).
    python -m qmagic.calib_is [shots_per_setting=400000] -> results/calib_is.json

Outcome (2026-10-02, 4e5 shots per setting, results/calib_is.json): NO gain. With circuit-level noise the mean number of
error events per shot (sum_i p_i = 5.9 at d=9, 10.8 at d=11, p=1e-3) already exceeds the minimum failing weight (d+1)/2,
so logical failures are rare because errors must align, not because they are few; inflating every mechanism by lambda
makes the likelihood-ratio weights heavy-tailed (lambda=2: variance per shot equal to plain sampling; lambda>=3: the
estimate collapses by orders of magnitude because the dominant low-weight failures are no longer sampled). Rare-event
methods that help here must target error geometry (splitting / minimum-weight enumeration), not error count.
"""
from qmagic.paths import RESULTS
import json, math, sys, time, numpy as np, stim, pymatching
from qmagic import noise as NM
from gen._chunk._noise import NoiseModel
def scaled_dem(dem, lam):
    out = stim.DetectorErrorModel()
    for ins in dem.flattened():
        if ins.type == "error":
            q = min(0.5, lam * ins.args_copy()[0]); out.append("error", [q], ins.targets_copy())
        else: out.append(ins)
    return out
def experiment(p, d, lam, shots, batch=20000, seed=1):
    c = NM.memory(d, "Z", NoiseModel.uniform_depolarizing(p))
    dem = c.detector_error_model(decompose_errors=True).flattened()
    m = pymatching.Matching.from_detector_error_model(dem, enable_correlations=True)
    p_i = np.array([ins.args_copy()[0] for ins in dem if ins.type == "error"]); q_i = np.minimum(0.5, lam * p_i)
    a = np.log(p_i / q_i) - np.log((1 - p_i) / (1 - q_i)); b0 = np.sum(np.log((1 - p_i) / (1 - q_i)))     # log w = b0 + errs @ a
    sampler = scaled_dem(dem, lam).compile_sampler(seed=seed)
    s1 = s2 = 0.0; n = 0; nfail = 0; t0 = time.time()
    while n < shots:
        k = min(batch, shots - n)
        det, obs, err = sampler.sample(k, return_errors=True)
        pred = m.decode_batch(det, enable_correlations=True)[:, 0]
        fail = pred != obs[:, 0]
        w = np.exp(b0 + err[fail].astype(np.float64) @ a) if fail.any() else np.zeros(0)   # weights only needed on failures
        s1 += w.sum(); s2 += (w ** 2).sum(); nfail += int(fail.sum()); n += k
    sec = time.time() - t0; est = s1 / n; var = s2 / n - est ** 2
    shots_10pct = var / (0.01 * est ** 2) if est > 0 else float("inf")
    return dict(p=p, d=d, lam=lam, shots=n, failures=nfail, est=est, var_per_shot=var, sec=sec, sec_per_shot=sec / n,
                shots_for_10pct=shots_10pct, sec_for_10pct=shots_10pct * sec / n, mechanisms=int(len(p_i)))
if __name__ == "__main__":
    shots = int(float(sys.argv[1])) if len(sys.argv) > 1 else 400_000
    out = []
    for p, d in ((1e-3, 9), (1e-3, 11), (5e-4, 11)):
        for lam in (1.0, 2.0, 3.0, 4.0, 6.0):
            r = experiment(p, d, lam, shots); out.append(r)
            print(f"p={p:g} d={d:2} lam={lam:g}: est={r['est']:.2e} failures={r['failures']:6d} var/shot={r['var_per_shot']:.2e} "
                  f"{r['sec_per_shot']*1e6:.0f} us/shot | shots for 10%: {r['shots_for_10pct']:.2e} = {r['sec_for_10pct']/3600:.2f} core-h", flush=True)
    json.dump(out, open(RESULTS / "calib_is.json", "w"), indent=1)
