"""Build params_corr_d2.json from params_corr.json:
 - d1=3: our d2-scan points (d_req=d2), 5e-4 rows from the 3e8-shot run.
 - d1=5: Gidney's d2=15 tails kept (d_req=11 as before) PLUS scaled copies at d2 in {11,13}: volume x (our baseline(d2)/baseline(15)),
         error unchanged (d2-independence validated for d1=3 fully, for d1=5 at the curve head), flagged scaled=True; our raw d1=5 points added too.
"""
from qmagic.paths import DATA, RESULTS, EXTERNAL, find
import json, glob
prm = json.load(open(find("params_corr.json")))
gid5 = [s for s in prm["cultivate"] if s.get("d1") == 5]
def thin(curve, minerr=4):
    rows = sorted([x for x in curve if x["errors_raw"] >= minerr], key=lambda x: (x["err"], x["v"])); out, prev = [], None
    for x in rows:
        if prev is not None and x["v"] > prev * 0.9 and x["err"] != 0: continue
        prev = x["v"]; out.append(x)
    return out
def load(f): return json.load(open(find(f)))
new, base5 = [], {}
runs = [r for r in load("cultiv_d2scan_d3_5e7.json") if r["meta"]["p"] == 1e-3] + load("cultiv_d2scan_d3_p5e4_3e8.json") + load("cultiv_d2scan_d5_5e8.json")
for r in runs:
    m = r["meta"]
    if m["d1"] == 5: base5[(m["p"], m["d2"])] = r["baseline_volume"]
    for x in thin(r["curve"]):
        new.append(dict(name=f"cult_d1={m['d1']}_d2={m['d2']}_att{x['attempts']:.1f}", p=m["p"], d_req=m["d2"], eps_T=x["err"], volume_per_T=x["v"],
                        estimate=False, attempts=round(x["attempts"], 2), errors=x["errors_raw"], d1=m["d1"], d2=m["d2"], q=r["q"],
                        source="our rerun of Gidney's construction (run_cultiv2.py), desaturation sampler, errors x2 (S->T), v = baseline/keep"))
# Gidney's d1=5 baseline at d2=15 = v/attempts of the first row
scaled = []
for p in (1e-3, 5e-4):
    rows15 = [s for s in gid5 if s["p"] == p]
    if not rows15: continue
    b15 = min(s["volume_per_T"] / s["attempts"] for s in rows15)
    for d2 in (11, 13):
        if (p, d2) not in base5: continue
        ratio = base5[(p, d2)] / b15
        for s in rows15:
            scaled.append(dict(s, name=s["name"].replace("cult_d1=5", f"cult_d1=5_d2={d2}scaled"), d_req=d2, d2=d2, volume_per_T=s["volume_per_T"] * ratio, scaled=True, estimate=True,
                               source=f"Gidney 2024 d2=15 tail, volume x{ratio:.2f} (= our baseline({d2})/Gidney baseline(15)); error assumed d2-independent"))
    print(f"p={p:g}: Gidney d1=5 baseline(15)={b15:.0f}; ours " + ", ".join(f"d2={d2}: {base5[(p,d2)]:.0f} ({base5[(p,d2)]/b15:.2f})" for d2 in (11, 13) if (p, d2) in base5))
prm["cultivate"] = gid5 + scaled + new
json.dump(prm, open(RESULTS / "params_corr_d2.json", "w"), indent=1)
print(len(gid5), "Gidney d1=5 rows;", len(scaled), "scaled d1=5 rows;", len(new), "our measured rows")
