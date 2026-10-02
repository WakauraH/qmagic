"""Dynamic sequential sampler for logical-error calibrations.
Every point (a circuit + decoder) is sampled in CHUNK-shot tasks spread over all cores; a point closes when its error count
reaches `target` or its shots reach `maxshots`. Points are opened in waves: `cheap` points first, then, once the provisional
fit `fit_fn(rows)` can predict them, the `rest` points that are expected to reach `min_err` errors within `maxshots`;
the others are skipped (they would cost the most and never enter the fit). All cores stay busy across waves.
"""
import time, math, numpy as np
from multiprocessing import Pool
CHUNK = 500_000
def _task(args):
    key, builder, seed, shots = args
    import pymatching
    c = builder(key)
    m = pymatching.Matching.from_detector_error_model(c.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True), enable_correlations=True)
    s = c.compile_detector_sampler(seed=seed); errs = 0; done = 0; t0 = time.time()
    while done < shots:
        n = min(100_000, shots - done); det, obs = s.sample(n, separate_observables=True)
        errs += int(np.sum(m.decode_batch(det, enable_correlations=True)[:, 0] != obs[:, 0])); done += n
    return key, shots, errs, time.time() - t0
def run_points(cheap, rest, builder, predict, target=100, maxshots=30_000_000, min_err=30, procs=16, seed_base=0, log=print):
    """cheap/rest: lists of point keys (hashable, e.g. (p, d, basis)). builder(key) -> stim.Circuit. predict(closed_rows, key) ->
    expected errors at maxshots or None if no fit yet. Returns (rows, skipped, wall) with rows = {key: dict(shots, errors, sec)}."""
    state = {k: dict(shots=0, errors=0, sec=0.0, inflight=0, open=True) for k in cheap}
    pending_rest = list(rest); skipped = []; seeds = {}; t0 = time.time()
    def nseed(k): seeds[k] = seeds.get(k, 0) + 1; return seed_base + hash(repr(k)) % 100_000 * 1000 + seeds[k]
    def tasks_for(k):
        st = state[k]; left = maxshots - st["shots"] - st["inflight"] * CHUNK
        if left <= 0 or not st["open"]: return []
        rate = st["errors"] / st["shots"] if st["shots"] else None
        want = left if rate is None or rate == 0 else min(left, max(CHUNK, (target - st["errors"]) / rate * 1.2))
        n = max(1, min(procs, math.ceil(want / CHUNK)))
        return [(k, builder, nseed(k), min(CHUNK, left - i * CHUNK)) for i in range(n) if left - i * CHUNK > 0]
    with Pool(procs) as pool:
        results = []
        def submit(k):
            for t in tasks_for(k): state[k]["inflight"] += 1; results.append(pool.apply_async(_task, (t,)))
        for k in cheap: submit(k)
        while results or pending_rest:
            if not results: break
            done = [r for r in results if r.ready()]
            if not done: time.sleep(0.05); continue
            for r in done:
                results.remove(r); key, shots, errs, sec = r.get(); st = state[key]
                st["shots"] += shots; st["errors"] += errs; st["sec"] += sec; st["inflight"] -= 1
                if st["open"] and (st["errors"] >= target or st["shots"] >= maxshots): st["open"] = False
                if st["open"] and st["inflight"] == 0: submit(key)
            closed = {k: v for k, v in state.items() if not v["open"]}
            for k in list(pending_rest):                              # open rest points as soon as the fit can judge them
                e = predict(closed, k)
                if e is None: continue
                pending_rest.remove(k)
                if e >= min_err: state[k] = dict(shots=0, errors=0, sec=0.0, inflight=0, open=True); submit(k); log(f"open {k}: expected {e:.1f} errors at {maxshots:.0e}")
                else: skipped.append((k, e)); log(f"skip {k}: expected {e:.1f} errors at {maxshots:.0e} < {min_err}")
        for k in pending_rest: skipped.append((k, None)); log(f"skip {k}: no fit available")
    rows = {k: dict(shots=v["shots"], errors=v["errors"], sec=round(v["sec"])) for k, v in state.items()}
    return rows, skipped, time.time() - t0
