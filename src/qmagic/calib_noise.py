"""Data-block calibration under alternative noise models (Gidney memory circuit, rounds=d, correlated matching).
Models: uniform(p) | biased(p, eta=10,100) | erasure(p_pauli, pe) with pe/(p_pauli+pe) in {0.5, 0.9} at fixed total.
Reports p_L = p_X + p_Z (X- and Z-basis memories) per d rounds.   -> calib_noise.json
"""
from qmagic.paths import DATA, RESULTS, EXTERNAL, find
import json, time, numpy as np, stim, pymatching
from multiprocessing import Pool
from qmagic import noise as NM
from gen._chunk._noise import NoiseModel
BATCH, TARGET, MAXSHOTS = 100_000, 200, 10_000_000
def build(model, p, d, basis):
    if model == "uniform": return NM.memory(d, basis, NoiseModel.uniform_depolarizing(p))
    if model.startswith("biased"): return NM.memory(d, basis, NM.BiasedNoiseModel(p, float(model.split("_")[1])))
    if model.startswith("erasure"):
        frac = float(model.split("_")[1]); pp = p * (1 - frac); pe = p * frac
        return NM.erasure_circuit(NM.memory(d, basis, NoiseModel.uniform_depolarizing(pp)), pe)
def run(cfg):
    model, p, d, basis = cfg; c = build(model, p, d, basis)
    dem = c.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True)
    m = pymatching.Matching.from_detector_error_model(dem, enable_correlations=True)
    s = c.compile_detector_sampler(seed=hash(cfg) % (2**31)); shots = errs = 0; t0 = time.time()
    while errs < TARGET and shots < MAXSHOTS:
        det, obs = s.sample(BATCH, separate_observables=True)
        errs += int(np.sum(m.decode_batch(det, enable_correlations=True)[:, 0] != obs[:, 0])); shots += BATCH
    return dict(model=model, p=p, d=d, basis=basis, shots=shots, errors=errs, pL=errs / shots, sec=round(time.time() - t0))
if __name__ == "__main__":
    cfgs = [(m, p, d, b) for m in ("uniform", "biased_10", "biased_100", "erasure_0.5", "erasure_0.9") for p in (1e-3, 2e-3) for d in (3, 5, 7, 9) for b in ("X", "Z")]
    with Pool(16) as pool: rows = pool.map(run, cfgs)
    json.dump(rows, open(RESULTS / "calib_noise.json", "w"), indent=1)
    agg = {}
    for r in rows: agg.setdefault((r["model"], r["p"], r["d"]), {})[r["basis"]] = r
    print(f"{'model':12} {'p':>6} {'d':>2} | {'pL_X':>9} {'pL_Z':>9} {'pL_X+Z':>9} (errors X/Z)")
    for k in sorted(agg):
        x, z = agg[k]["X"], agg[k]["Z"]; print(f"{k[0]:12} {k[1]:6g} {k[2]:2d} | {x['pL']:9.2e} {z['pL']:9.2e} {x['pL']+z['pL']:9.2e} ({x['errors']}/{z['errors']})")
