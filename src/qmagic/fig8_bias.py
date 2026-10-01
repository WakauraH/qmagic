from qmagic.paths import DATA, RESULTS, EXTERNAL, find
import json, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
def thin(curve, minerr=4):
    rows = sorted([x for x in curve if x["errors_raw"] >= minerr], key=lambda x: (x["err"], x["v"])); out, prev = [], None
    for x in rows:
        if prev is not None and x["v"] > prev * 0.9 and x["err"] != 0: continue
        prev = x["v"]; out.append(x)
    return sorted(out, key=lambda x: x["v"])
runs = json.load(open(find("cultiv_noise_bias.json"))) + json.load(open(find("cultiv_noise_bias2.json")))
fig, ax = plt.subplots(figsize=(6.5, 4.2))
for r in runs:
    m = r["meta"]
    if m["d1"] != 3: continue
    t = thin(r["curve"]); eta = m["noise"].replace("bias", "η=")
    ax.loglog([x["v"] for x in t], [x["err"] for x in t], "o-", label=f"d1=3, d2=7, p=1e-3, {eta}" + (" (uniform)" if eta == "η=1" else ""))
ax.set_xlabel("expected qubit·rounds per kept T"); ax.set_ylabel("error per kept T state"); ax.set_title("cultivation under Z-biased noise (same total p)"); ax.grid(alpha=.3, which="both"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(RESULTS / "fig8_bias_cultivation.png", dpi=150); print("fig8 written")
