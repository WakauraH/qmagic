"""Phase 2: 4-arm cost model (B0 baseline / A QEM / B cultivation / C hybrid). See q1_plan.md §3 (Phase-0-corrected).

    python costmodel.py [params.json]   -> Suzuki self-check, sweep table on stdout, results.csv

Model (all provenance in phase0_extraction.md):
  layer 1  p_L(d)   = A (p/p_th)^{s(d+1)/2}      per logical op (d rounds), own Stim calibration
           Q_data   = n_L * 2 d^2 * r_route       r_route = 2.31 (Litinski)
  layer 2  T source: {qubits, cycles_per_T, eps_T, d_req}  -> k = ceil(cycles_per_T/d) copies (1 T per d cycles, Litinski)
                     or {volume_per_T (qubit*rounds)}       -> Q_fac = volume/d  (cultivation, Gidney 2024/2025 anchors)
  layer 3  no QEM : Sigma = N_C p_L + N_T eps_T <= N_e      (Suzuki's condition; N_e is the precision target delta)
           QEM    : samples Gamma = exp(4 Sigma) <= C_max   (Suzuki Eq.15, Piveteau Gamma^2=gamma^{2t})
                    AND residual bias Sigma * bias_r <= N_e  (same precision target as the no-QEM arms; bias_r = relative
                    calibration error of the logical rates; bias_r=None disables the constraint = pre-v4 behaviour)
  v4 options (panel fixes): kappa_mode="route" -> kappa(u) = 1 + (r_route-1) u (routing tiles accumulate lattice-surgery error
           in proportion to their duty cycle u); growth_rounds=k -> a cultivated state at d2 < d idles k rounds at d2 before
           growth, adding k p_L(d2)/d2 to eps_T (k="d" charges the whole consumption at d2: upper variant).
"""
from qmagic.paths import DATA, RESULTS, EXTERNAL, find
import json, math, sys, csv

def pL(prm, p, d):
    f = prm["pL_fit"]; return f["A"] * (p / f["p_th"]) ** (f["s"] * (d + 1) / 2)

def source_cost(s, d, t_rate=1.0):
    """t_rate = T gates consumed per d cycles (Litinski default 1). Factories scale with t_rate."""
    if "volume_per_T" in s: return t_rate * s["volume_per_T"] / d, 0
    k = math.ceil(t_rate * s["cycles_per_T"] / d); return k * s["qubits"], k

def arm(prm, kind, qem, p, d_max, N, f_T, C_max, cap_d=True, objective="Q"):
    n_L, r, N_e, vs, tr = prm["n_L"], prm["r_route"], prm["N_e"], prm.get("volume_scale", 1.0), prm.get("t_rate", 1.0)
    par = prm.get("parallelism", None)   # logical ops per time step; None = Suzuki per-op accounting (no idle error)
    kappa = prm.get("kappa", 1.0)          # patches accumulating idle error per step, in units of n_L (1 = data patches only; r_route counts routing tiles too)
    if prm.get("unified_clock"):           # panel fix 1: one clock. T supply rate follows parallelism: rho = pi * f_T
        tr = (par if par else n_L) * f_T
    cons = prm.get("consumption", None)    # panel fix 2: symmetric consumption channel 0.75*p_L(d) on BOTH arms (sources must then be 'no-cons' outputs)
    bias_r = prm.get("bias_r", None)       # v4: relative calibration error of logical rates -> residual bias Sigma*bias_r must stay <= N_e
    growth = prm.get("growth_rounds", 0)   # v4: idle rounds at d2 before growth to d (cultivation rows with d_req < d only)
    if par is not None and prm.get("kappa_mode") == "route": kappa = 1.0 + (r - 1.0) * par / n_L
    N_T, N_C = N * f_T, N * (1 - f_T)
    best = None
    for d in range(3, (d_max if cap_d else 61) + 1, 2):
        for s in prm[kind]:
            if s["p"] != p or (cap_d and s["d_req"] > d_max): continue
            if par is None: eps_data = N * pL(prm, p, d)                       # Suzuki: one patch-d-rounds of error per op
            else: eps_data = (N / par) * kappa * n_L * pL(prm, p, d)            # idle error: kappa*n_L patches for d rounds per time step
            eps_T = s["eps_T"] + (0.75 * pL(prm, p, d) if cons else 0.0)        # symmetric consumption at the data distance
            if growth and "volume_per_T" in s and s["d_req"] < d:                # v4: growth/teleportation charge for d2 < d
                d2 = s["d_req"]; k = d if growth == "d" else growth
                eps_T += k * pL(prm, p, d2) / d2
            eps_tot = eps_data + N_T * eps_T
            if qem:
                if 4 * eps_tot > math.log(C_max): continue
                if bias_r is not None and eps_tot * bias_r > N_e: continue        # v4: residual bias at the common precision target
                G = math.exp(4 * eps_tot)
            else:
                if eps_tot > N_e: continue
                G = 1.0
            Q_fac, k = source_cost(s, d, tr)
            if "volume_per_T" in s: Q_fac *= vs
            Q_data = n_L * 2 * d * d * r
            row = dict(d=d, src=s["name"], k=k, Q_data=Q_data, Q_fac=Q_fac, Q=Q_data + Q_fac, Gamma=G,
                       T_cycles=N_T * d / tr * G, eps_tot=eps_tot, est=s.get("estimate", False))
            row["QTG"] = row["Q"] * row["T_cycles"]
            if best is None or row[objective] < best[objective]: best = row
    return best

ARMS = [("B0", "distill", False), ("A", "distill", True), ("B", "cultivate", False), ("C", "cultivate", True)]

def nmax(prm, kind, p, d_max, f_T, G=100.0, qem=True):
    """Largest circuit size N for which the arm is feasible: budget / min_(d, source) [per-op data error + f_T * eps_T].
    QEM budget = min(ln G / 4, N_e / bias_r); no-QEM budget = N_e. Same idle/kappa/consumption/growth rules as arm()."""
    par, kap, n_L, r = prm.get("parallelism"), prm.get("kappa", 1.0), prm["n_L"], prm["r_route"]; best = None
    if par is not None and prm.get("kappa_mode") == "route": kap = 1.0 + (r - 1.0) * par / n_L
    growth = prm.get("growth_rounds", 0); cons = prm.get("consumption", None)
    for d in range(3, d_max + 1, 2):
        for s in prm[kind]:
            if s["p"] != p or s["d_req"] > d_max: continue
            eps_T = s["eps_T"] + (0.75 * pL(prm, p, d) if cons else 0.0)
            if growth and "volume_per_T" in s and s["d_req"] < d:
                d2 = s["d_req"]; eps_T += (d if growth == "d" else growth) * pL(prm, p, d2) / d2
            per_op = pL(prm, p, d) * (1 if par is None else kap * n_L / par)
            rt = per_op + f_T * eps_T; best = rt if best is None or rt < best else best
    if best is None: return float("nan")
    if not qem: return prm["N_e"] / best
    budget = math.log(G) / 4
    if prm.get("bias_r") is not None: budget = min(budget, prm["N_e"] / prm["bias_r"])
    return budget / best

def sweep(prm, objective="Q"):
    rows = []
    for p in prm["p"]:
        for d_max in prm["d_max"]:
            for N in prm["N"]:
                for f_T in prm["f_T"]:
                    for C_max in prm["C_max"]:
                        res = {a: arm(prm, k, q, p, d_max, N, f_T, C_max, objective=objective) for a, k, q in ARMS}
                        ref = arm(prm, "distill", False, p, d_max, N, f_T, C_max, cap_d=False, objective=objective)
                        Qref = ref["Q"] if ref else float("nan")
                        A, C = res["A"], res["C"]
                        for a, _, _ in ARMS:
                            r = res[a]
                            rows.append(dict(arm=a, p=p, d_max=d_max, N=N, f_T=f_T, C_max=C_max, feasible=r is not None,
                                             Q_ref_uncapped=Qref, d_ref=ref["d"] if ref else None,
                                             **(r or {}), reduction_vs_ref=(1 - r["Q"] / Qref) if r else float("nan"),
                                             deltaQ_C_minus_A=((A["Q"] - C["Q"]) / Qref) if (A and C) else float("nan"),
                                             saving_C_vs_A=(1 - C["Q"] / A["Q"]) if (A and C) else float("nan"),
                                             Gamma_ratio_C_over_A=(C["Gamma"] / A["Gamma"]) if (A and C) else float("nan")))
    return rows

def suzuki_check():
    """Reproduce Suzuki 2022 Sec. V with HIS constants: p_dec = C1 (C2 p/pth)^{(d+1)/2}, C1=.13, C2=.61, p/pth=.1, m=1.
    Expect d 9->4 (N=1e4) and 19->14 (N=1e10), N_e=1e-3 without QEM, Gamma<=1e2 with QEM."""
    C1, C2, x = 0.13, 0.61, 0.1
    dd = lambda pmax: 2 * math.log(pmax / C1) / math.log(C2 * x) - 1        # continuous d with p_dec = pmax
    out = {}
    for N, exp_n, exp_q in ((1e4, 9, 4), (1e10, 19, 14)):
        d_n, d_q = dd(1e-3 / N), dd(math.log(1e2) / (4 * N))
        out[N] = (round(d_n), round(d_q), (d_q / d_n) ** 2)
        assert (round(d_n), round(d_q)) == (exp_n, exp_q), (N, d_n, d_q)
    return out

if __name__ == "__main__":
    print("Suzuki self-check (his constants) {N: (d_noQEM, d_QEM, qubit ratio)}:", suzuki_check())
    prm = json.load(open(find(sys.argv[1] if len(sys.argv) > 1 else "params.json")))
    objective = sys.argv[2] if len(sys.argv) > 2 else "Q"          # Q (min qubits) or QTG (min qubits x runtime x samples)
    rows = sweep(prm, objective)
    keys = sorted({k for r in rows for k in r})
    out = "results.csv" if objective == "Q" else f"results_{objective}.csv"
    with open(RESULTS / out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    for r in rows:
        head = f'{r["arm"]:2} p={r["p"]:g} dmax={r["d_max"]:2} N={r["N"]:.0e} fT={r["f_T"]:<4} Gmax={r["C_max"]:<6g}'
        body = (f'd={r["d"]:2} {r["src"]:<22} Q={r["Q"]:8.0f} (data {r["Q_data"]:7.0f} + fac {r["Q_fac"]:7.0f}) G={r["Gamma"]:7.2f} red={r["reduction_vs_ref"]:+.2f}'
                + (" [est]" if r.get("est") else "")) if r["feasible"] else "INFEASIBLE"
        tail = f'  | dQ(C-A)={r["deltaQ_C_minus_A"]:+.3f} G_C/G_A={r["Gamma_ratio_C_over_A"]:.2f}' if r["arm"] == "C" else ""
        print(head, body, tail)
