"""Generate the end-to-end cultivation circuits used in this study with Gidney's own generator (external repository), i.e.
    tools/make_circuits --circuit_type end2end-inplace-distillation --gateset css --basis Y --r1 d1 --r2 5 ...
    python -m qmagic.make_circuits <out_subdir> --d1 3 --d2 7 9 11 --p 1e-3 5e-4      -> external/magic-state-cultivation/out/<out_subdir>/*.stim
Then sample with:  python -m qmagic.cultivation external/magic-state-cultivation/out/<out_subdir> <out.json> <shots> [procs]
"""
from qmagic.paths import EXTERNAL
import argparse, subprocess, sys, pathlib
def make(out_dir, d1, d2s, ps, r2=5):
    out_dir = pathlib.Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(EXTERNAL / "tools" / "make_circuits"), "--circuit_type", "end2end-inplace-distillation", "--gateset", "css",
           "--basis", "Y", "--d1", str(d1), "--r1", "d1", "--r2", str(r2), "--d2", *map(str, d2s), "--noise_strength", *map(str, ps),
           "--out_dir", str(out_dir)]
    subprocess.run(cmd, check=True, cwd=EXTERNAL)
    return sorted(out_dir.glob("*.stim"))
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_subdir"); ap.add_argument("--d1", type=int, default=3); ap.add_argument("--d2", type=int, nargs="+", default=[7, 9, 11])
    ap.add_argument("--p", type=float, nargs="+", default=[1e-3]); ap.add_argument("--r2", type=int, default=5)
    a = ap.parse_args()
    if not EXTERNAL.exists(): sys.exit("external repository missing: run scripts/fetch_external.sh")
    for f in make(EXTERNAL / "out" / a.out_subdir, a.d1, a.d2, a.p, a.r2): print("wrote", f.name)
if __name__ == "__main__": main()
