"""C-1: numerical check of logical-level PEC sampling overhead and bias from misestimated rates.
Model: 1 logical qubit, N ops = alternating H and T (Clifford + non-Clifford), each followed by a stochastic Pauli channel
with rates (pX,pY,pZ) = eps*(1,1,1)/3 (Cliffords) or eps_T*(0,0,1) (T gates, Z-type after twirl).
PEC: inverse channel quasi-probability c_I=(1+..), c_P<0; gamma_op = sum|c|. Estimator of <Z> at the end.
Checks: (a) sample variance * M ~ gamma_tot^2  (=> Gamma = gamma_tot^2 = exp(4 sum eps) samples for fixed precision)
        (b) bias when PEC uses eps*(1+r) instead of eps.
    python pec_demo.py
"""
from qmagic.paths import DATA, RESULTS, EXTERNAL, find
import numpy as np, math
rng = np.random.default_rng(1)
H = np.array([[1, 1], [1, -1]]) / np.sqrt(2); T = np.diag([1, np.exp(1j * np.pi / 4)])
PAULI = [np.eye(2), np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.diag([1, -1])]

def inverse_qpd(pv):
    """Pauli channel with probs pv=(pI,pX,pY,pZ) -> quasi-prob coefficients of its inverse (exact, via Walsh-Hadamard)."""
    Wmat = np.array([[1, 1, 1, 1], [1, 1, -1, -1], [1, -1, 1, -1], [1, -1, -1, 1]])   # eigenvalues lambda = W p
    lam = Wmat @ pv; c = Wmat.T @ (1 / lam) / 4                                       # inverse channel coefficients
    return c

def run(N, eps, eps_T, M, r_mis=0.0, seed=0, single_shot=True):
  with np.errstate(all="ignore"):      # spurious Accelerate-BLAS warnings on complex matmul
    return _run(N, eps, eps_T, M, r_mis, seed, single_shot)

def _run(N, eps, eps_T, M, r_mis, seed, single_shot):
    rng = np.random.default_rng(seed)
    ops = [(H, eps * np.array([0, 1, 1, 1]) / 3) if i % 2 == 0 else (T, eps_T * np.array([0, 0, 0, 1])) for i in range(N)]
    # true channels and PEC (with possibly misestimated rates)
    gamma_tot = 1.0; qpd = []
    for U, pvec in ops:
        p_true = pvec.copy(); p_true[0] = 1 - p_true[1:].sum()
        p_est = pvec * (1 + r_mis); p_est[0] = 1 - p_est[1:].sum()
        c = inverse_qpd(p_est); g = np.abs(c).sum(); gamma_tot *= g
        qpd.append((p_true, np.abs(c) / g, np.sign(c), g))
    # vectorized simulation over M samples: state as (M,2) complex
    psi = np.zeros((M, 2), complex); psi[:, 0] = 1
    sign = np.ones(M)
    for (U, _), (p_true, q, s, g) in zip(ops, qpd):
        psi = psi @ U.T
        k_err = rng.choice(4, size=M, p=p_true)          # true noise
        k_pec = rng.choice(4, size=M, p=q)               # PEC insertion
        sign *= s[k_pec]
        for k in range(1, 4):
            m = (k_err == k); psi[m] = psi[m] @ PAULI[k].T
            m = (k_pec == k); psi[m] = psi[m] @ PAULI[k].T
    p0 = np.abs(psi[:, 0]) ** 2
    if single_shot:                                       # panel fix: one +-1 measurement outcome per trajectory (real hardware)
        z = np.where(rng.random(M) < p0, 1.0, -1.0)
    else:                                                 # exact expectation per trajectory (original demo; an artefact, kept for comparison)
        z = 2 * p0 - 1
    est = gamma_tot * sign * z
    # ideal value
    psi0 = np.array([1, 0], complex)
    for U, _ in ops: psi0 = U @ psi0
    ideal = abs(psi0[0]) ** 2 - abs(psi0[1]) ** 2
    ideal_var = 1 - ideal ** 2                            # single-shot variance of the ideal circuit's +-1 outcome
    return dict(mean=est.mean(), sem=est.std() / np.sqrt(M), var_x_M=est.var(), gamma_tot=gamma_tot, ideal=ideal,
                sum_eps=sum((pv[1:].sum()) for _, pv in ops), shot_multiplier=est.var() / ideal_var if ideal_var > 0 else float("nan"))

def feasible_with_bias(sum_eps, r_mis, delta, G_max):
    """Panel fix: feasibility must bound the mean-squared error, not just the sample count.
    Relative bias from misestimated rates ~ exp(2 r sum_eps) - 1; require bias <= delta AND samples Gamma=exp(4 sum_eps) <= G_max."""
    bias = math.exp(2 * r_mis * sum_eps) - 1
    return bias <= delta and math.exp(4 * sum_eps) <= G_max, bias

if __name__ == "__main__":
    import json; rec = dict(a=[], b=[])
    print("(a) single-shot (+-1) PEC: Var*M vs gamma^2 - mu^2, and shot multiplier vs ideal circuit  (M=2e5)")
    for N, eps, eps_T in ((2000, 5e-5, 5e-5), (2000, 2.5e-4, 2.5e-4), (2000, 5e-4, 5e-4), (4000, 2.5e-4, 5e-4)):
        r = run(N, eps, eps_T, M=200_000); rec["a"].append(dict(N=N, eps=eps, eps_T=eps_T, **{k: float(v) for k, v in r.items()}))
        print(f"  N={N} eps={eps:g} eps_T={eps_T:g}: sum_eps={r['sum_eps']:.3f} gamma^2={r['gamma_tot']**2:.3f} | Var*M={r['var_x_M']:.3f} (gamma^2-mu^2={r['gamma_tot']**2-r['ideal']**2:.3f}) shot_multiplier={r['shot_multiplier']:.2f}  est={r['mean']:+.4f}±{r['sem']:.4f} ideal={r['ideal']:+.4f}")
    r = run(2000, 2.5e-4, 2.5e-4, M=200_000, single_shot=False)
    print(f"  [exact-expectation estimator, original demo] sum_eps=0.5: Var*M={r['var_x_M']:.3f} = {r['var_x_M']/r['gamma_tot']**2:.2f} gamma^2  <- artefact")
    print("\n(b) bias from misestimated rates (single-shot; PEC uses eps*(1+r)); N=2000, sum eps=0.5, M=4e5")
    for r_mis in (0.0, 0.05, 0.1, 0.2, -0.2):
        r = run(2000, 2.5e-4, 2.5e-4, M=400_000, r_mis=r_mis, seed=3); rec["b"].append(dict(r_mis=r_mis, bias=float(r["mean"] - r["ideal"]), rule_sigma_r=float(r_mis * r["sum_eps"] * r["ideal"]), **{k: float(v) for k, v in r.items()}))
        print(f"  r={r_mis:+.2f}: est={r['mean']:+.4f}±{r['sem']:.4f} ideal={r['ideal']:+.4f} bias={r['mean']-r['ideal']:+.4f}  rule exp(2 r sum_eps)-1 = {math.exp(2*r_mis*r['sum_eps'])-1:+.4f} (x ideal = {(math.exp(2*r_mis*r['sum_eps'])-1)*r['ideal']:+.4f})")
    print("\n(c) feasibility with bias: sum_eps budget at G_max=100 is ln(100)/4=1.151; the manuscript's rule Sigma*r <= N_e gives Sigma <= 1 at r = N_e = 1e-3")
    json.dump(rec, open(RESULTS / "pec_demo.json", "w"), indent=1)
