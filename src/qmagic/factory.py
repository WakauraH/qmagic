"""Litinski 2019 (arXiv:1905.06903) (15-to-1)_{dX,dZ,dm} surface-code factory: qubits, cycles, p_out, p_fail.
5-qubit density-matrix simulation following the error rules of his Secs. 2-3 (pessimistic ballpark, as he says).
    python factory15.py            -> validates against Table 1 rows, writes factories_d11.json (dX<=11 grid at p=1e-3, 5e-4)
"""
from qmagic.paths import DATA, RESULTS, EXTERNAL, find
import json, itertools, numpy as np

# 15 rotations of Fig. 3: Z_S for nonempty S in {2..5}; qubit 1 joins iff |S| even. Verified: product = Z1(-pi/8).
def rotations():
    R = []
    for r in range(1, 5):
        for S in itertools.combinations([1, 2, 3, 4], r):
            R.append(tuple(sorted(((0,) if r % 2 == 0 else ()) + S)))
    return R
ROTS = rotations()
# step schedule (Sec. 3): step1: rot 1-3 (Z2,Z3,Z4) + rot5 (Z2Z3Z4); step2: init q1, two of Z1ZiZj (i,j in 2-4); step3: init q5, Z5 + third Z1ZiZj + Z1Z4Z5; steps 4-6: remaining 6, two per step.
def schedule():
    single = {(1,), (2,), (3,), (4,)}; s234 = (1, 2, 3); q1_no5 = [S for S in ROTS if 0 in S and 4 not in S and len(S) == 3]
    rest = [S for S in ROTS if S not in single and S != s234 and S not in q1_no5 and S != (0, 3, 4)]
    steps = [[(1,), (2,), (3,), s234], q1_no5[:2], [(4,), q1_no5[2], (0, 3, 4)], rest[0:2], rest[2:4], rest[4:6]]
    assert sorted(sum(steps, [])) == sorted(ROTS)
    return steps

def pL_litinski(p, d): return 0.1 * (100 * p) ** ((d + 1) / 2)

I2 = np.eye(2); X = np.array([[0, 1], [1, 0]]); Z = np.diag([1, -1])
def kron_op(op, q, n=5):
    m = np.array([[1.0]])
    for i in range(n): m = np.kron(m, op if i == q else I2)
    return m
XS = [kron_op(X, q) for q in range(5)]; ZS = [kron_op(Z, q) for q in range(5)]
def zprod(S):
    d = np.ones(32)
    for q in S: d = d * np.diag(ZS[q])
    return d                       # diagonal of Z_S
def rot_unitary(S, theta): return np.diag(np.exp(-1j * theta * zprod(S)))
def apply_pauli_channel(rho, P, prob): return (1 - prob) * rho + prob * P @ rho @ P
def apply_rotation(rho, S, err):   # err: dict theta_error -> prob ; ideal theta = pi/8
    out = (1 - sum(err.values())) * (U := rot_unitary(S, np.pi / 8)) @ rho @ U.conj().T
    for dtheta, pr in err.items():
        U = rot_unitary(S, np.pi / 8 + dtheta); out = out + pr * U @ rho @ U.conj().T
    return out

def simulate(dX, dZ, dm, p, pL=pL_litinski, consume=True):
  with np.errstate(all="ignore"):      # spurious Accelerate-BLAS warnings on complex matmul; results validated finite
    return _simulate(dX, dZ, dm, p, pL, consume)

def _simulate(dX, dZ, dm, p, pL, consume=True):
    l = dX + 4 * dZ
    plus = np.array([1, 1]) / np.sqrt(2); psi = np.array([1.0])
    for _ in range(5): psi = np.kron(psi, plus)
    rho = np.outer(psi, psi.conj())
    present = {1, 2, 3}                                   # qubits 2-4 (indices 1..3) initialised first
    for si, step in enumerate(schedule()):
        if si == 1: present.add(0)
        if si == 2: present.add(4)
        for S in step:
            err = {-np.pi / 4: p / 3, np.pi / 4: p / 3, np.pi / 2: p / 3}   # Eq. (5)
            if len(S) == 1:                                # dZ x dm patch + faulty T measurement
                err[-np.pi / 4] += 0.5 * dZ * pL(p, dm)
                rho = apply_pauli_channel(rho, ZS[S[0]], 0.5 * dm ** 2 / dZ * pL(p, dZ))
            else:                                          # fast faulty T measurement, ancilla region l x dX for dm cycles
                err[-np.pi / 4] += 0.5 * l * dX / dm * pL(p, dm)          # X strings in ancilla -> wrong outcome
                err[-np.pi / 4] += 0.5 * pL(p, dm) * dm                  # X storage on |+> dm x dm patch
                err[np.pi / 2] += 0.5 * pL(p, dm) * dm                   # Z storage on |+> patch
                pz = 0.5 * (l / dX) * pL(p, dX) * dm                     # Z strings in ancilla
                rho = apply_pauli_channel(rho, ZS[0] if 0 in S else ZS[S[0]], pz)
            rho = apply_rotation(rho, S, err)
        for q in present:                                  # dm cycles of storage after each step
            if q == 0: px = pz = 0.5 * pL(p, dX)
            else: px, pz = 0.5 * (dZ / dX) * pL(p, dX), 0.5 * (dX / dZ) * pL(p, dZ)
            rho = apply_pauli_channel(rho, XS[q], px * dm); rho = apply_pauli_channel(rho, ZS[q], pz * dm)
    if consume:                                            # output consumed via dX x dX ancilla: dX cycles (Litinski). consume=False -> factory-exit error only
        for P in (XS[0], ZS[0]):
            rho = apply_pauli_channel(rho, P, 0.5 * pL(p, dX) * dX)
    # post-select qubits 2-5 on X=+1
    proj = np.array([[1.0]]); Pp = np.outer(plus, plus)
    for q in range(5): proj = np.kron(proj, I2 if q == 0 else Pp)
    rho_ps = proj @ rho @ proj; ptrace = np.real(np.trace(rho_ps)); p_fail = 1 - ptrace
    r = rho_ps.reshape(2, 16, 2, 16); rho1 = np.einsum("aibi->ab", r) / ptrace
    ideal = np.array([1, np.exp(-1j * np.pi / 4)]) / np.sqrt(2)          # |m~> = Z1(-pi/8)|+>
    p_out = 1 - np.real(ideal.conj() @ rho1 @ ideal)
    qubits = 2 * (dX + 4 * dZ) * 3 * dX + 4 * dm; cycles = 6 * dm / (1 - p_fail)
    return dict(dX=dX, dZ=dZ, dm=dm, p=p, p_out=float(p_out), p_fail=float(p_fail), qubits=int(qubits), cycles=float(cycles))

TABLE = [((7, 3, 3), 1e-4, 4.4e-8, 810, 18.1), ((9, 3, 3), 1e-4, 9.3e-10, 1150, 18.1),
         ((11, 5, 5), 1e-4, 1.9e-11, 2070, 30.0), ((17, 7, 7), 1e-3, 4.5e-8, 4620, 42.6)]

def main_1level():
    worst = 1
    for (dX, dZ, dm), p, pout, q, cyc in TABLE:
        r = simulate(dX, dZ, dm, p)
        ratio = r["p_out"] / pout; worst = max(worst, ratio, 1 / ratio)
        print(f"(15-to-1)_{dX},{dZ},{dm} p={p:g}: p_out {r['p_out']:.2e} (Litinski {pout:.1e}, x{ratio:.2f})  "
              f"qubits {r['qubits']} ({q})  cycles {r['cycles']:.1f} ({cyc})  p_fail {r['p_fail']:.3f}")
    print("worst p_out ratio vs Table 1:", round(worst, 2))
    assert worst < 3, "reconstruction disagrees with Litinski Table 1 by >3x"
    # Phase 1b: d_X <= 11 grid
    try: fit = json.load(open(find("calib_pL.json")))["fit"]; pL_own = lambda p, d: fit["A"] * (p / fit["p_th"]) ** (fit["exponent_scale"] * (d + 1) / 2) / d  # per cycle
    except FileNotFoundError: fit = None
    grid = []
    for p in (1e-3, 5e-4):
        for dX in (5, 7, 9, 11):
            for dZ in (3, 5, 7):
                for dm in (3, 5, 7):
                    if dZ > dX or dm > dX: continue
                    r = simulate(dX, dZ, dm, p); r["pL_model"] = "litinski"; grid.append(r)
                    if fit: r2 = simulate(dX, dZ, dm, p, pL=pL_own); r2["pL_model"] = "own_calib"; grid.append(r2)
    json.dump(dict(validation_worst_ratio=worst, rows=grid), open(RESULTS / "factories_d11.json", "w"), indent=1)
    best = {}
    for r in grid:
        k = (r["p"], r["pL_model"]); best.setdefault(k, []).append(r)
    for k, rs in best.items():
        rs.sort(key=lambda r: r["p_out"]); print(k, "best 3 by p_out:", [(r["dX"], r["dZ"], r["dm"], f'{r["p_out"]:.1e}', r["qubits"], round(r["cycles"], 1)) for r in rs[:3]])


# ---------------- two-level factory (from factory15_2level.py) ----------------
def simulate2(l1, l2, p, pL=pL_litinski, consume=True):
    """l1=(dX1,dZ1,dm1), l2=(dX2,dZ2,dm2). Returns dict incl. level-1 block count k. consume=False: level-2 exit error only
    (level-1 outputs are always consumed by level 2)."""
    r1 = simulate(*l1, p, pL=pL)
    dX, dZ, dm = l2; l = dX + 4 * dZ
    plus = np.array([1, 1]) / np.sqrt(2); psi = np.array([1.0])
    for _ in range(5): psi = np.kron(psi, plus)
    rho = np.outer(psi, psi.conj()); present = {1, 2, 3}
    with np.errstate(all="ignore"):
        for si, step in enumerate(schedule()):
            if si == 1: present.add(0)
            if si == 2: present.add(4)
            for S in step:
                err = {-np.pi / 4: 0.0, np.pi / 4: 0.0, np.pi / 2: r1["p_out"]}     # input state error (Z-type after twirl)
                if len(S) == 1:
                    err[-np.pi / 4] += 0.5 * dZ * pL(p, dm)
                    rho = apply_pauli_channel(rho, ZS[S[0]], 0.5 * dm ** 2 / dZ * pL(p, dZ))
                else:
                    err[-np.pi / 4] += 0.5 * l * dX / dm * pL(p, dm) + 0.5 * pL(p, dm) * dm
                    err[np.pi / 2] += 0.5 * pL(p, dm) * dm
                    rho = apply_pauli_channel(rho, ZS[0] if 0 in S else ZS[S[0]], 0.5 * (l / dX) * pL(p, dX) * dm)
                rho = apply_rotation(rho, S, err)
            for q in present:
                px, pz = (0.5 * pL(p, dX),) * 2 if q == 0 else (0.5 * (dZ / dX) * pL(p, dX), 0.5 * (dX / dZ) * pL(p, dZ))
                rho = apply_pauli_channel(rho, XS[q], px * dm); rho = apply_pauli_channel(rho, ZS[q], pz * dm)
        if consume:
            for P in (XS[0], ZS[0]): rho = apply_pauli_channel(rho, P, 0.5 * pL(p, dX) * dX)
        proj = np.array([[1.0]]); Pp = np.outer(plus, plus)
        for q in range(5): proj = np.kron(proj, I2 if q == 0 else Pp)
        rho_ps = proj @ rho @ proj; ptrace = np.real(np.trace(rho_ps)); p_fail = 1 - ptrace
        rho1 = np.einsum("aibi->ab", rho_ps.reshape(2, 16, 2, 16)) / ptrace
        ideal = np.array([1, np.exp(-1j * np.pi / 4)]) / np.sqrt(2); p_out = 1 - np.real(ideal.conj() @ rho1 @ ideal)
    cycles2 = 6 * dm / (1 - p_fail)
    k = int(np.ceil(15 * r1["cycles"] / cycles2))                 # level-1 blocks to supply 15 inputs per level-2 run
    qubits = k * r1["qubits"] + 2 * (dX + 4 * dZ) * 3 * dX + 4 * dm
    return dict(l1=l1, l2=l2, k=k, p=p, p_out=float(p_out), p_fail=float(p_fail), qubits=int(qubits), cycles=float(cycles2), p_out1=r1["p_out"])

TABLE2 = [(((11, 5, 5), (25, 11, 11)), 1e-3, 2.7e-12, 30700, 82.5, 6), (((13, 5, 5), (29, 11, 13)), 1e-3, 3.3e-14, 39100, 97.5, 6),
         (((17, 7, 7), (41, 17, 17)), 1e-3, 4.5e-20, 73400, 128, 6)]
def main_2level():
    worst = 1
    for (l1, l2), p, pout, q, cyc, k in TABLE2:
        r = simulate2(l1, l2, p); ratio = r["p_out"] / pout; worst = max(worst, ratio, 1 / ratio)
        print(f"{l1}x{l2} p={p:g}: p_out {r['p_out']:.2e} (Litinski {pout:.1e}, x{ratio:.2f}) qubits {r['qubits']} ({q}, k={r['k']} vs {k}) cycles {r['cycles']:.1f} ({cyc})")
    print("worst ratio:", round(worst, 2))
    fit = json.load(open(find("calib_gidney.json")))["fit"]       # v4: same calibration as params_v3; exit-only outputs
    pL_own = lambda p, d: fit["A"] * (p / fit["p_th"]) ** (fit["exponent_scale"] * (d + 1) / 2) / d
    grid = []
    for p in (1e-3, 5e-4):
        best = None
        for l1 in ((7, 3, 3), (9, 3, 3), (9, 5, 5), (11, 5, 5)):
            for l2 in ((9, 5, 5), (11, 5, 5), (11, 7, 7), (11, 9, 9)):
                r = simulate2(l1, l2, p, pL=pL_own, consume=False); r["pL_model"] = "calib_gidney_exit_only"; grid.append(r)
                if best is None or r["p_out"] < best["p_out"]: best = r
        one = min((simulate(11, dZ, dm, p, pL=pL_own, consume=False) for dZ in (3, 5, 7, 9) for dm in (3, 5, 7, 9)), key=lambda r: r["p_out"])
        print(f"p={p:g}: best 2-level d<=11: {best['l1']}x{best['l2']} p_out={best['p_out']:.2e} (level-1 {best['p_out1']:.1e}) qubits={best['qubits']} cycles={best['cycles']:.0f} | best 1-level d<=11: p_out={one['p_out']:.2e} qubits={one['qubits']} | ratio 2-level/1-level p_out={best['p_out']/one['p_out']:.2f}, qubits x{best['qubits']/one['qubits']:.1f}")
    json.dump(grid, open(RESULTS / "factories_2level_d11.json", "w"), indent=1)


if __name__ == "__main__":
    main_1level(); main_2level()
