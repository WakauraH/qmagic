"""Direct multiprocessing runner for Gidney's end2end cultivation circuits (desaturation sampler), bypassing sinter.collect.
Reproduces the paper's post-processing: gap-threshold curve (cultiv.stat_to_gap_stats), expected volume
v = compute_expected_injection_growth_volume(noisy r_end=1 circuit)/keep_rate, errors x2 (S->T assumption).
    python -m qmagic.cultivation <circuit_dir> <out_json> <shots_per_circuit> [procs]     (circuits: python -m qmagic.make_circuits)
"""
from qmagic.paths import DATA, RESULTS, EXTERNAL, find
import sys, json, pathlib, time, collections
from multiprocessing import Pool
import stim, sinter
REPO = EXTERNAL
sys.path.insert(0, str(REPO / "src"))
import cultiv, gen

def worker(args):
    path, shots, seed = args
    c = stim.Circuit.from_file(path)
    comp = cultiv.sinter_samplers()["desaturation"].compiled_sampler_for_task(
        sinter.Task(circuit=c, decoder="desaturation", json_metadata={}, detector_error_model=c.detector_error_model(approximate_disjoint_errors=True)))
    tot = sinter.AnonTaskStats(); done = 0
    while done < shots:
        n = min(20000, shots - done); tot += comp.sample(n); done += n
    return dict(shots=tot.shots, discards=tot.discards, errors=tot.errors, custom_counts=dict(tot.custom_counts))

def postprocess(meta, agg, circuit_path):
    d1, d2, p = meta["d1"], meta["d2"], meta["p"]
    stat = sinter.TaskStats(strong_id="x", decoder="desaturation", json_metadata=meta, shots=agg["shots"], errors=agg["errors"],
                            discards=agg["discards"], custom_counts=collections.Counter(agg["custom_counts"]))
    gs = cultiv.stat_to_gap_stats([stat], rounding=1, func=lambda arg: sinter.AnonTaskStats(
        shots=arg.source.shots, discards=arg.at_least.discards + arg.less.shots, errors=arg.at_least.errors))
    c = cultiv.make_end2end_cultivation_circuit(dcolor=d1, dsurface=d2, basis="Y", r_growing=d1, r_end=1, inject_style="unitary")
    noise = str(meta.get("noise", "uniform"))
    if noise.startswith("bias") and noise != "bias1":
        from qmagic import noise as NM; cn = NM.BiasedNoiseModel(p, float(noise[4:])).noisy_circuit_skipping_mpp_boundaries(c)
    else: cn = gen.NoiseModel.uniform_depolarizing(p).noisy_circuit_skipping_mpp_boundaries(c)
    baseline = cultiv.compute_expected_injection_growth_volume(cn, discard_rate=0)
    rows = []
    for g in gs:
        kept = g.shots - g.discards
        if kept <= 0: continue
        rows.append(dict(gap=g.json_metadata["gap"], shots=g.shots, kept=kept, attempts=g.shots / kept,
                         errors_raw=g.errors, err=2 * g.errors / kept, v=baseline / (kept / g.shots)))
    return dict(meta=meta, q=cultiv_q(circuit_path), baseline_volume=baseline, raw=agg, curve=rows)

def cultiv_q(path): return stim.Circuit.from_file(path).num_qubits

def run_dir(cdir, out, shots, procs=16):
    """Sample every *.stim circuit in cdir with Gidney's desaturation sampler and write RESULTS/out (list of per-circuit records)."""
    cdir = pathlib.Path(cdir); results = []
    for f in sorted(cdir.glob("*.stim")):
        meta = sinter.comma_separated_key_values(str(f)); t0 = time.time()
        per = -(-shots // procs)
        with Pool(procs) as pool: parts = pool.map(worker, [(str(f), per, i) for i in range(procs)])
        agg = dict(shots=sum(x["shots"] for x in parts), discards=sum(x["discards"] for x in parts), errors=sum(x["errors"] for x in parts),
                   custom_counts=dict(sum((collections.Counter(x["custom_counts"]) for x in parts), collections.Counter())))
        res = postprocess(meta, agg, str(f)); res["seconds"] = time.time() - t0; results.append(res)
        kept = agg["shots"] - agg["discards"]
        print(f"d1={meta['d1']} d2={meta['d2']} p={meta['p']}: shots={agg['shots']} keep={kept/agg['shots']:.3f} raw_err/kept={agg['errors']/max(kept,1):.2e} "
              f"baseline_vol={res['baseline_volume']:.0f} q={res['q']} {res['seconds']:.0f}s | gap curve: " +
              "; ".join(f"v={r['v']:.0f} att={r['attempts']:.1f} err={r['err']:.1e}({r['errors_raw']})" for r in res["curve"] if r["errors_raw"] >= 4)[:400], flush=True)
        json.dump(results, open(RESULTS / out, "w"), indent=1)
    return results

if __name__ == "__main__":
    run_dir(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 16)
