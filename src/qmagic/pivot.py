from qmagic.paths import DATA, RESULTS, EXTERNAL, find
import csv, sys
f = sys.argv[1]; dmax = sys.argv[2] if len(sys.argv) > 2 else "11"
rows = list(csv.DictReader(open(f)))
def g(r, k):
    try: return float(r.get(k, ""))
    except: return float("nan")
cell = lambda r: (f"d{int(float(r['d'])):2d} {g(r,'Q'):6.0f}({g(r,'Q_fac'):5.0f}){'*' if r.get('est')=='True' else ' '}" if r["feasible"] == "True" else f"{'--':>19}")
print(f"{f}  d_max={dmax}  Gmax=100 | Q(fac) per arm | save=1-Q_C/Q_A | dQ=(Q_A-Q_C)/Q_ref | G_C/G_A")
for p in ("0.001", "0.0005"):
    for N in ("10000.0", "100000.0", "1000000.0", "100000000.0", "10000000000.0"):
        for fT in ("0.1", "0.3", "1.0"):
            s = {r["arm"]: r for r in rows if r["p"] == p and r["d_max"] == dmax and r["N"] == N and r["f_T"] == fT and r["C_max"] == "100.0"}
            c = s["C"]
            print(f"{float(p):.4f} {float(N):5.0e} {fT:>3} | B0 {cell(s['B0'])} | A {cell(s['A'])} | B {cell(s['B'])} | C {cell(c)} | {g(c,'saving_C_vs_A'):+.2f} {g(c,'deltaQ_C_minus_A'):+.3f} {g(c,'Gamma_ratio_C_over_A'):5.2f}")
