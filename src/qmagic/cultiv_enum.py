"""Exact low-weight enumeration of the d1=3 (or d1=5) inject+cultivate stage (Gidney's ErrorEnumerationReport, external repo):
all sets of up to `max_weight` error mechanisms of the detector error model that trigger no detector (full post-selection)
and flip the logical observable, with their exact probabilities. Gives the error floor of the cultivation stage with zero
statistical error, per weight, and the acceptance rate.  Needs external/ (python -m qmagic.make_circuits is not needed:
circuits are built directly).  Errors are doubled (S -> T convention of the sampled data).
    python -m qmagic.cultiv_enum [max_weight=5] [d1=3] -> results/cultiv_enum_d<d1>.json   (d1=5, weight 5: 2797 mechanisms, ~45 min)
"""
from qmagic.paths import RESULTS, find
import json, sys, time
from qmagic import noise as NM
cultiv = NM.cultiv; gen = NM.gen
from cultiv import ErrorEnumerationReport
def main(max_weight=5, d1=3):
    out = {}; cache = {}
    meas = {s["p"]: {} for s in json.load(open(find("params_v3.json")))["cultivate"] if s.get("d1") == d1}
    for s in json.load(open(find("params_v3.json")))["cultivate"]:
        if s.get("d1") == d1 and not s.get("scaled") and s.get("errors", 0) >= 4:
            meas[s["p"]][s["d2"]] = min(meas[s["p"]].get(s["d2"], 1.0), s["eps_T"])
    for p in (1e-3, 5e-4):
        c = cultiv.make_inject_and_cultivate_circuit(inject_style="unitary", dcolor=d1, basis="Y")
        t0 = time.time(); rep = ErrorEnumerationReport.from_circuit(c, max_weight=max_weight, noise=p, cache=cache); sec = time.time() - t0
        byw = {int(w): float(v) for w, v in rep.distance_to_heralded_error_rate.items()}
        rec = dict(p=p, max_weight=max_weight, keep_rate=rep.keep_rate, heralded_error_rate_proxy=rep.heralded_error_rate,
                   floor_T=2 * rep.heralded_error_rate, by_weight_proxy=byw, n_logical_sets={int(w): sum(1 for e in rep.logical_errs if len(e.src_errors) == w) for w in byw},
                   seconds=sec, measured_floor_T_by_d2=meas.get(p, {}))
        out[f"{p:g}"] = rec
        print(f"p={p:g}: keep={rep.keep_rate:.3f}  exact proxy error (w<={max_weight}) = {rep.heralded_error_rate:.3e}  x2 -> {rec['floor_T']:.3e}  "
              f"| by weight: " + ", ".join(f"w{w}: {v:.2e} ({rec['n_logical_sets'][w]} sets)" for w, v in sorted(byw.items()) if v > 0)
              + f" | measured floors (x2, >=4 errors) by d2: " + ", ".join(f"d2={d}: {e:.2e}" for d, e in sorted(meas.get(p, {}).items())) + f" | {sec:.1f} s")
    json.dump(out, open(RESULTS / f"cultiv_enum_d{d1}.json", "w"), indent=1)
if __name__ == "__main__": main(int(sys.argv[1]) if len(sys.argv) > 1 else 5, int(sys.argv[2]) if len(sys.argv) > 2 else 3)
