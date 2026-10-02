"""Exact (sampling-free) version of the PEC demo: one logical qubit through alternating H and T with stochastic Pauli noise,
mitigated by the quasi-probability inverse of (possibly misestimated) channels. The signed estimator gamma*sign*z has
est^2 = gamma^2 for every trajectory, so Var*M = gamma^2 - mu^2 exactly; its mean is tr(Z rho_tilde) with rho_tilde the
state propagated through N_est^{-1} N_true U, a 2x2 linear-map calculation. Replaces 8 min of Monte Carlo by milliseconds.
    python -m qmagic.pec_exact -> results/pec_exact.json
"""
from qmagic.paths import RESULTS
import json, math, time, numpy as np
H = np.array([[1, 1], [1, -1]]) / np.sqrt(2); T = np.diag([1, np.exp(1j * np.pi / 4)])
PAULI = [np.eye(2), np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.diag([1, -1])]
W = np.array([[1, 1, 1, 1], [1, 1, -1, -1], [1, -1, 1, -1], [1, -1, -1, 1]])
def pauli_map(rho, coeffs): return sum(c * P @ rho @ P.conj().T for c, P in zip(coeffs, PAULI))
def run(N, eps, eps_T, r_mis=0.0):
    ops = [(H, eps * np.array([0, 1, 1, 1]) / 3) if i % 2 == 0 else (T, eps_T * np.array([0, 0, 0, 1])) for i in range(N)]
    rho = np.array([[1, 0], [0, 0]], complex); ideal = np.array([[1, 0], [0, 0]], complex); gamma = 1.0; sum_eps = 0.0
    for U, pv in ops:
        p_true = pv.copy(); p_true[0] = 1 - p_true[1:].sum(); sum_eps += pv[1:].sum()
        p_est = pv * (1 + r_mis); p_est[0] = 1 - p_est[1:].sum()
        c = W.T @ (1 / (W @ p_est)) / 4; gamma *= np.abs(c).sum()           # quasi-probability inverse of the estimated channel
        rho = pauli_map(pauli_map(U @ rho @ U.conj().T, p_true), c); ideal = U @ ideal @ U.conj().T
    mean = float(np.real(np.trace(PAULI[3] @ rho))); mu_ideal = float(np.real(np.trace(PAULI[3] @ ideal)))
    return dict(N=N, eps=eps, eps_T=eps_T, r_mis=r_mis, sum_eps=sum_eps, gamma_tot=gamma, gamma2=gamma ** 2, exp4sum=math.exp(4 * sum_eps),
                mean=mean, ideal=mu_ideal, bias=mean - mu_ideal, rule_sigma_r=r_mis * sum_eps * mu_ideal,
                var_x_M=gamma ** 2 - mean ** 2, shot_multiplier=(gamma ** 2 - mean ** 2) / (1 - mu_ideal ** 2) if abs(mu_ideal) < 1 else float("nan"))
if __name__ == "__main__":
    t0 = time.time(); rec = dict(a=[], b=[])
    print("(a) exact: gamma^2 vs exp(4 sum eps), Var*M = gamma^2 - mu^2, shot multiplier")
    for N, eps, eps_T in ((2000, 5e-5, 5e-5), (2000, 2.5e-4, 2.5e-4), (2000, 5e-4, 5e-4), (4000, 2.5e-4, 5e-4)):
        r = run(N, eps, eps_T); rec["a"].append(r)
        print(f"  N={N} eps={eps:g} eps_T={eps_T:g}: sum_eps={r['sum_eps']:.3f} gamma^2={r['gamma2']:.3f} exp(4S)={r['exp4sum']:.3f} Var*M={r['var_x_M']:.3f} shot_multiplier={r['shot_multiplier']:.2f} mean={r['mean']:+.4f} ideal={r['ideal']:+.4f}")
    print("(b) exact bias from misestimated rates, N=2000, sum eps=0.5")
    for r_mis in (0.0, 0.05, 0.1, 0.2, -0.2):
        r = run(2000, 2.5e-4, 2.5e-4, r_mis); rec["b"].append(r)
        print(f"  r={r_mis:+.2f}: mean={r['mean']:+.5f} ideal={r['ideal']:+.5f} bias={r['bias']:+.5f}  Sigma*r*ideal={r['rule_sigma_r']:+.5f}")
    rec["seconds"] = time.time() - t0; json.dump(rec, open(RESULTS / "pec_exact.json", "w"), indent=1); print(f"elapsed {rec['seconds']:.2f} s")
