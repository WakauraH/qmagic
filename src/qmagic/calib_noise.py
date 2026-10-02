"""Data-block calibration under alternative noise models (Gidney memory circuit, rounds=d, correlated matching).
Models: uniform(p) | biased(p, eta=10,100) | erasure(p_pauli, pe) with pe/(p_pauli+pe) in {0.5, 0.9} at fixed total.
Reports p_L = p_X + p_Z (X- and Z-basis memories) per d rounds.   -> calib_noise.json
Sequential design (default): TARGET errors per basis (100) or MAXSHOTS; per model the d <= 7 points run first and give a
provisional fit that decides whether the d = 9 points can reach MIN_ERR (20, the fit threshold in noise_costmodel) within
MAXSHOTS; otherwise they are skipped. Seeds are deterministic. The shipped data/calib_noise.json used --target 200 --no-adaptive.
    python -m qmagic.calib_noise [--target 100] [--maxshots 1e7] [--no-adaptive] [--procs 16] [--out calib_noise_seq.json]
"""
from qmagic.paths import DATA, RESULTS, EXTERNAL, find
import json, time, math, argparse, numpy as np, stim, pymatching
from qmagic import noise as NM
from gen._chunk._noise import NoiseModel
BATCH, MIN_ERR = 100_000, 20
MODELS, PS, DS = ("uniform", "biased_10", "biased_100", "erasure_0.5", "erasure_0.9"), (1e-3, 2e-3), (3, 5, 7, 9)
def build(model, p, d, basis):
    if model == "uniform": return NM.memory(d, basis, NoiseModel.uniform_depolarizing(p))
    if model.startswith("biased"): return NM.memory(d, basis, NM.BiasedNoiseModel(p, float(model.split("_")[1])))
    if model.startswith("erasure"):
        frac = float(model.split("_")[1]); pp = p * (1 - frac); pe = p * frac
        return NM.erasure_circuit(NM.memory(d, basis, NoiseModel.uniform_depolarizing(pp)), pe)
def build_key(key): return build(*key)          # module-level: lambdas cannot be sent to worker processes
def fit_model(rows, model):
    pts = {}
    for r in rows:
        if r["model"] == model: pts.setdefault((r["p"], r["d"]), []).append(r)
    used = [(p, d, sum(x["pL"] for x in v)) for (p, d), v in pts.items() if len(v) == 2 and sum(x["errors"] for x in v) >= MIN_ERR and sum(x["pL"] for x in v) > 0]
    X = np.array([[1.0, (d + 1) / 2, (d + 1) / 2 * math.log(p)] for p, d, _ in used]); y = np.array([math.log(pl) for *_, pl in used])
    logA, b, c = np.linalg.lstsq(X, y, rcond=None)[0]; return math.exp(logA), math.exp(-b / c), c
def main():
    from qmagic import seqcalib
    ap = argparse.ArgumentParser(); ap.add_argument("--target", type=int, default=100); ap.add_argument("--maxshots", type=float, default=1e7)
    ap.add_argument("--no-adaptive", action="store_true"); ap.add_argument("--procs", type=int, default=16); ap.add_argument("--out", default="calib_noise_seq.json")
    a = ap.parse_args(); maxshots = int(a.maxshots)
    cheap = [(m, p, d, b) for m in MODELS for p in PS for d in DS for b in ("X", "Z") if d <= 7]; rest = [(m, p, d, b) for m in MODELS for p in PS for d in DS for b in ("X", "Z") if d > 7]
    if a.no_adaptive: cheap, rest = cheap + rest, []
    to_rows = lambda rows: [dict(model=k[0], p=k[1], d=k[2], basis=k[3], shots=v["shots"], errors=v["errors"], pL=v["errors"] / max(v["shots"], 1), sec=v["sec"]) for k, v in rows.items()]
    def predict(closed, key):
        m, p, d, _ = key; rows = [r for r in to_rows(closed) if r["model"] == m]
        pts = {}
        for r in rows: pts.setdefault((r["p"], r["d"]), []).append(r)
        if sum(1 for v in pts.values() if len(v) == 2 and sum(x["errors"] for x in v) >= MIN_ERR) < 4: return None
        A, pth, c = fit_model(rows, m); return maxshots * A * (p / pth) ** (c * (d + 1) / 2) / 2      # per basis
    rows, skipped, wall = seqcalib.run_points(cheap, rest, build_key, predict, target=a.target, maxshots=maxshots, min_err=MIN_ERR // 2, procs=a.procs, seed_base=7_000_000)
    rows = to_rows(rows)
    json.dump(rows, open(RESULTS / a.out, "w"), indent=1)
    json.dump(dict(target=a.target, maxshots=maxshots, adaptive=not a.no_adaptive, skipped=[list(k) for k, _ in skipped], core_seconds=sum(r["sec"] for r in rows), wall_seconds=round(wall)),
              open(RESULTS / a.out.replace(".json", "_design.json"), "w"), indent=1)
    agg = {}
    for r in rows: agg.setdefault((r["model"], r["p"], r["d"]), {})[r["basis"]] = r
    print(f"{'model':12} {'p':>6} {'d':>2} | {'pL_X':>9} {'pL_Z':>9} {'pL_X+Z':>9} (errors X/Z)")
    for k in sorted(agg):
        x, z = agg[k].get("X"), agg[k].get("Z")
        if x and z: print(f"{k[0]:12} {k[1]:6g} {k[2]:2d} | {x['pL']:9.2e} {z['pL']:9.2e} {x['pL']+z['pL']:9.2e} ({x['errors']}/{z['errors']})")
    print(f"core-seconds {sum(r['sec'] for r in rows)}, wall {wall:.0f} s")
if __name__ == "__main__": main()
