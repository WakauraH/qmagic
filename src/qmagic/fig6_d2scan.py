from qmagic.paths import DATA, RESULTS, EXTERNAL, find
import json, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
def thin(curve, minerr=4):
    rows = sorted([x for x in curve if x["errors_raw"] >= minerr], key=lambda x: (x["err"], x["v"])); out, prev = [], None
    for x in rows:
        if prev is not None and x["v"] > prev * 0.9 and x["err"] != 0: continue
        prev = x["v"]; out.append(x)
    return sorted(out, key=lambda x: x["v"])
g = json.load(open(find("gidney2024_fig1_points.json")))
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for i, p in enumerate((1e-3, 5e-4)):
    import glob
    src = [r for r in json.load(open(find("cultiv_d2scan_d3_5e7.json"))) if r["meta"]["p"] == 1e-3] + json.load(open(find("cultiv_d2scan_d3_p5e4_3e8.json"))) + json.load(open(find("cultiv_validate_d15_2e7.json"))) + json.load(open(find("cultiv_d2scan_d5_5e8.json")))
    for r in src:
        if r["meta"]["p"] != p: continue
        t = thin(r["curve"]); ax[i].loglog([x["v"] for x in t], [x["err"] for x in t], "o-", label=f"this work, $d_1={r['meta']['d1']}$, $d_2={r['meta']['d2']}$ ($q={r['q']}$)")
    for d1 in (3, 5):
        pts = sorted([x for x in g if "This Work" in x["c"] and x["state"] == "T" and x["p"] == p and x["d1"] == d1 and x["errors"] >= 4], key=lambda x: x["v"])
        ax[i].loglog([x["v"] for x in pts], [x["err"] for x in pts], "k--" if d1 == 3 else "k:", alpha=.7, label=f"Gidney 2024, $d_1={d1}$, $d_2=15$")
    ax[i].set_title(f"$p={p:g}$: cultivation error vs expected volume"); ax[i].set_xlabel("expected qubit$\\cdot$rounds per accepted $|T\\rangle$ (incl. retries)"); ax[i].set_ylabel("error per accepted $|T\\rangle$ state"); ax[i].grid(alpha=.3, which="both"); ax[i].legend(fontsize=7)
fig.tight_layout(); fig.savefig(RESULTS / "fig6_d2scan.png", dpi=300); print("fig6 written")
