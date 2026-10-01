"""Compare regenerated results/ with the shipped reference/ outputs (JSON and CSV, numbers to a relative tolerance).
    python -m qmagic.verify [rtol]      exit 0 if every reference file is reproduced
"""
from qmagic.paths import ROOT, RESULTS
import json, csv, math, sys
REF = ROOT / "reference"
TOL = {"pec_demo.json": 1e-2}            # Monte-Carlo demo: seeded, but BLAS/RNG-version differences are tolerated
def num(x):
    try: return float(x)
    except (TypeError, ValueError): return None
def close(a, b, rtol):
    fa, fb = num(a), num(b)
    if fa is not None and fb is not None and not isinstance(a, bool) and not isinstance(b, bool):
        if math.isnan(fa) and math.isnan(fb): return True
        return math.isclose(fa, fb, rel_tol=rtol, abs_tol=1e-300)
    return a == b
def diff(a, b, rtol, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        out = [f"{path}: keys differ {sorted(set(a) ^ set(b))}"] if set(a) != set(b) else []
        for k in a:
            if k in b: out += diff(a[k], b[k], rtol, f"{path}/{k}")
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b): return [f"{path}: length {len(a)} vs {len(b)}"]
        return [m for i, (x, y) in enumerate(zip(a, b)) for m in diff(x, y, rtol, f"{path}[{i}]")]
    return [] if close(a, b, rtol) else [f"{path}: {a!r} vs {b!r}"]
def main():
    rtol = float(sys.argv[1]) if len(sys.argv) > 1 else 1e-6
    if not REF.exists(): sys.exit(f"no reference directory at {REF}")
    files = [f for f in sorted(REF.iterdir()) if f.suffix in (".json", ".csv")]
    if not files: sys.exit(f"no reference files in {REF}")
    bad = 0
    for ref in sorted(REF.iterdir()):
        if ref.suffix not in (".json", ".csv"): continue
        out = RESULTS / ref.name
        if not out.exists(): print(f"MISSING  {ref.name}"); bad += 1; continue
        tol = TOL.get(ref.name, rtol)
        if ref.suffix == ".json": msgs = diff(json.load(open(ref)), json.load(open(out)), tol)
        else: msgs = diff(list(csv.DictReader(open(ref))), list(csv.DictReader(open(out))), tol)
        if msgs: print(f"DIFFERS  {ref.name}: {len(msgs)} mismatches, e.g. {msgs[0]}"); bad += 1
        else: print(f"ok       {ref.name}")
    print("VERIFIED" if not bad else f"{bad} file(s) not reproduced"); sys.exit(1 if bad else 0)
if __name__ == "__main__": main()
