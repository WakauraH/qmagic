"""Replace cultivation entries in params.json with Gidney 2024 Fig.1 sourced points (gidney2024_fig1_points.json)."""
from qmagic.paths import DATA, RESULTS, EXTERNAL, find
import json
pts = json.load(open(find("gidney2024_fig1_points.json"))); prm = json.load(open(find("params.json"))); cult = []
for x in pts:
    if "This Work" not in x["c"] or x["state"] != "T" or x["p"] not in (1e-3, 5e-4) or x["errors"] < 4: continue
    cult.append(dict(name=f"cult_d1={x['d1']}_att{x['attempts']:.0f}", p=x["p"], d_req=11, eps_T=x["err"], volume_per_T=x["v"],
                     estimate=False, attempts=round(x["attempts"], 2), errors=x["errors"], d1=x["d1"], d2=x["d2"], q=x["q"],
                     source="Gidney 2024 Fig.1 data: Strilanc/magic-state-cultivation assets/emulated-historical-stats.csv (v = expected qubit-rounds incl. retries; d2=15 grafted code ~ d=11 surface code per Sec 3.3)"))
prm["cultivate"] = cult; prm.pop("volume_scale", None)
prm["_note"] = "pL_fit from calib_pL.json (Phase 1). distill from factories_d11.json (Litinski model validated to 18%). cultivate: ALL points sourced from Gidney 2024 Fig.1 data (GitHub Strilanc/magic-state-cultivation, Apache-2.0); points with <4 errors dropped."
json.dump(prm, open(RESULTS / "params.json", "w"), indent=1)
print(len(cult), "sourced cultivation points")
