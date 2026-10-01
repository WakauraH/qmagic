"""Reproduce the shipped results from shipped data (fast path, ~2-3 min) or re-run the slow simulations.

    qmagic-reproduce            # fast (~6 min): every table and figure of the paper from data/ -> results/
    qmagic-reproduce --verify   # + compare the regenerated JSON/CSV with reference/ (numbers to 1e-6 relative)
    qmagic-reproduce --calib    # + p_L calibration with Stim/PyMatching (~15 min, correlated matching)
    qmagic-reproduce --cultiv   # + cultivation reruns (needs scripts/fetch_external.sh; hours)
"""
import argparse, runpy, subprocess, sys, time
from qmagic.paths import RESULTS, DATA, EXTERNAL

STEPS_FAST = [("factory reconstruction + two-level validation", "qmagic.factory", []),
              ("correlated-matching parameter set (params_corr.json)", "qmagic.rerun_corr", []),
              ("d2-scan parameter set (params_corr_d2.json)", "qmagic.build_params_d2scan", []),
              ("v3/v4 parameter set (full factory grid, physical d_req, bias constraint)", "qmagic.build_params_v3", []),
              ("per-p calibration fits (claim (a) sensitivity)", "qmagic.calib_perp", []),
              ("claims (b)/(c) + sensitivity variants (kappa(u), growth)", "qmagic.claims_v3", []),
              ("sensitivity (n_L, f_T, r_route, Gamma_max, precision, spacetime)", "qmagic.sensitivity_v3", []),
              ("acceptance cutoff under the mitigation budget", "qmagic.cutoff_v3", ["1.0"]),
              ("Q3 hardware-Lambda mapping", "qmagic.q3_lambda", []),
              ("PEC numerical demo", "qmagic.pec", []),
              ("gap analysis + fig7", "qmagic.gap", []),
              ("noise-model cost comparison + fig9", "qmagic.noise_costmodel", []),
              ("fig6", "qmagic.fig6_d2scan", []), ("fig10", "qmagic.fig10_unified", []), ("fig11", "qmagic.fig11_floors", []),
              # legacy (pre-v3) outputs kept for the lab notebook; not used by the paper
              ("legacy costmodel (params_corr_d2 -> results.csv)", "qmagic.costmodel", ["params_corr_d2.json"]),
              ("legacy Q5 N_max frontier", "qmagic.q5_nmax", ["params_corr_d2.json", "_corr_d2"]),
              ("legacy Q1 pivot", "qmagic.pivot", ["results.csv", "11"]),
              ("legacy fig1-3", "qmagic.plots", []), ("legacy fig5", "qmagic.fig5_q3", []), ("legacy fig8", "qmagic.fig8_bias", [])]

def run_module(mod, args):
    cmd = [sys.executable, "-m", mod, *args]; t0 = time.time()
    r = subprocess.run(cmd, cwd=RESULTS, capture_output=True, text=True)
    tail = "\n".join(l for l in r.stdout.splitlines() if "Warning" not in l)[-1500:]
    print(f"--- {mod} ({time.time()-t0:.0f}s, exit {r.returncode})\n{tail}")
    if r.returncode: print(r.stderr[-2000:])
    return r.returncode

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calib", action="store_true", help="re-run the Stim/PyMatching calibrations (needs external/)")
    ap.add_argument("--cultiv", action="store_true", help="re-run cultivation under biased noise (needs external/; hours)")
    ap.add_argument("--verify", action="store_true", help="after the fast path, compare results/ with reference/")
    a = ap.parse_args()
    print(f"data: {DATA}\nresults: {RESULTS}")
    fails = 0
    if a.calib:
        fails += run_module("qmagic.calib", ["--corr"]); fails += run_module("qmagic.calib_noise", [])
    if a.cultiv:
        if not EXTERNAL.exists(): sys.exit("external repo missing: run scripts/fetch_external.sh")
        fails += run_module("qmagic.cultiv_noise", [])
    for name, mod, args in STEPS_FAST:
        print(f"== {name}"); fails += run_module(mod, args)
    if a.verify:
        print("== verify against reference/"); fails += run_module("qmagic.verify", [])
    print("DONE" if not fails else f"FAILED steps: {fails}"); sys.exit(1 if fails else 0)

if __name__ == "__main__": main()
