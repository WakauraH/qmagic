"""Noise models beyond uniform depolarizing, built on Gidney's gen.NoiseModel so the SAME engine noises memory and cultivation circuits.
  biased(p, eta): Z-biased Pauli noise; 1q/idle PAULI_CHANNEL_1(px,py,pz) with pz = p*eta/(eta+1), px=py=p/(2(eta+1));
                  2q PAULI_CHANNEL_2 with Z-type {IZ,ZI,ZZ} sharing p*eta/(eta+1), the other 12 sharing p/(eta+1). Measurement flip p, reset flip p.
  erasure(p, pe): uniform depolarizing p plus HERALDED_ERASE(pe) on both qubits after every 2q gate (atom loss at entangling gates).
  add_herald_detectors(circ): declare a DETECTOR on every herald record so decoders can see the erasure flags.
  memory(d, basis, model): Gidney's surface-code memory circuit (rounds=d) noised with `model`.
"""
from qmagic.paths import DATA, RESULTS, EXTERNAL, find
import sys, pathlib, stim
REPO = EXTERNAL; sys.path.insert(0, str(REPO / "src"))
try:
    import gen, cultiv
    from gen._chunk._noise import NoiseModel, NoiseRule
    HAVE_EXTERNAL = True
except ImportError:                       # erasure_circuit / strip_herald_detectors work without the external repo
    HAVE_EXTERNAL = False
    class NoiseModel:                     # placeholder so the module imports; BiasedNoiseModel needs the external repo
        def __init__(self, *a, **k): raise ImportError("BiasedNoiseModel needs external/magic-state-cultivation (scripts/fetch_external.sh)")
    NoiseRule = None

def _biased_tuple(p, eta):
    pz = p * eta / (eta + 1); px = py = p / (2 * (eta + 1)); return (px, py, pz)

class BiasedNoiseModel(NoiseModel):
    def __init__(self, p, eta):
        self.p, self.eta = p, eta; px, py, pz = _biased_tuple(p, eta)
        # two-qubit: 15 Paulis in stim order IX IY IZ XI XX XY XZ YI YX YY YZ ZI ZX ZY ZZ
        z_type = {"IZ", "ZI", "ZZ"}; names = ["IX","IY","IZ","XI","XX","XY","XZ","YI","YX","YY","YZ","ZI","ZX","ZY","ZZ"]
        pz2 = p * eta / (eta + 1) / 3; po2 = p / (eta + 1) / 12
        t2 = tuple(pz2 if n in z_type else po2 for n in names)
        super().__init__(idle_depolarization=0,
                         any_clifford_1q_rule=NoiseRule(after={"PAULI_CHANNEL_1": (px, py, pz)}),
                         any_clifford_2q_rule=NoiseRule(after={"PAULI_CHANNEL_2": t2}),
                         measure_rules={b: NoiseRule(after={"PAULI_CHANNEL_1": (px, py, pz)}, flip_result=p) for b in ["X","Y","Z","XX","XY","XZ","YX","YY","YZ","ZX","ZY","ZZ"]},
                         gate_rules={"RX": NoiseRule(after={"Z_ERROR": p}), "RY": NoiseRule(after={"X_ERROR": p}), "R": NoiseRule(after={"X_ERROR": p})})
        self._idle_tuple = (px, py, pz)
    def _append_idle_error(self, *, moment_split_ops, out, system_qubit_indices, immune_qubit_indices):
        # same idle-set logic as the base class, but a biased channel instead of DEPOLARIZE1
        tmp = stim.Circuit(); self.idle_depolarization = 1e-9
        NoiseModel._append_idle_error(self, moment_split_ops=moment_split_ops, out=tmp, system_qubit_indices=system_qubit_indices, immune_qubit_indices=immune_qubit_indices)
        self.idle_depolarization = 0
        for inst in tmp:
            if inst.name == "DEPOLARIZE1": out.append("PAULI_CHANNEL_1", [t.value for t in inst.targets_copy()], self._idle_tuple)
            else: out.append(inst)

def erasure_circuit(noisy: stim.Circuit, pe: float, gates=("CX", "CZ")) -> stim.Circuit:
    """Insert HERALDED_ERASE(pe) on the targets of every 2q gate of an already-noised circuit, declare a DETECTOR per herald
    (coord -9), and re-index every rec[-k] in later DETECTOR/OBSERVABLE_INCLUDE instructions (heralds add measurement records)."""
    src = noisy.flattened(); orig_abs = []; n = 0
    for inst in src:
        if stim.gate_data(inst.name).produces_measurements: n += len(inst.targets_copy())
    out = stim.Circuit(); n_orig = 0; n_new = 0; o2n = {}
    for inst in src:
        gd = stim.gate_data(inst.name)
        if inst.name in ("DETECTOR", "OBSERVABLE_INCLUDE"):
            tg = []
            for tr in inst.targets_copy():
                oa = n_orig + tr.value              # rec[-k] -> absolute original index (tr.value is -k)
                tg.append(stim.target_rec(o2n[oa] - n_new))
            out.append(inst.name, tg, inst.gate_args_copy()); continue
        out.append(inst)
        if gd.produces_measurements:
            for j in range(len(inst.targets_copy())): o2n[n_orig + j] = n_new + j
            k = len(inst.targets_copy()); n_orig += k; n_new += k
        if inst.name in gates:
            q = [x.value for x in inst.targets_copy()]
            out.append("HERALDED_ERASE", q, pe)
            for j in range(len(q)): out.append("DETECTOR", [stim.target_rec(-len(q) + j)], [-9, j])
            n_new += len(q)
    return out

def add_herald_detectors(circ): return circ   # heralds are declared inside erasure_circuit

def strip_herald_detectors(circ: stim.Circuit) -> stim.Circuit:
    out = stim.Circuit()
    for inst in circ.flattened():
        if inst.name == "DETECTOR" and inst.gate_args_copy()[:1] == [-9.0]: continue
        out.append(inst)
    return out

def memory(d, basis, model, rounds=None):
    c = cultiv.make_surface_code_memory_circuit(dsurface=d, basis=basis, rounds=rounds or d)
    return model.noisy_circuit_skipping_mpp_boundaries(c)

if __name__ == "__main__":
    import numpy as np, pymatching, time
    def pL(circ, shots, corr=True, seed=1):
        dem = circ.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True)
        m = pymatching.Matching.from_detector_error_model(dem, enable_correlations=corr)
        det, obs = circ.compile_detector_sampler(seed=seed).sample(shots, separate_observables=True)
        pred = m.decode_batch(det, enable_correlations=corr); err = int(np.sum(pred[:, 0] != obs[:, 0])); return err / shots, err
    d, p, shots = 5, 1e-3, 400_000
    print(f"d={d} p={p} shots={shots}: logical error per {d} rounds")
    u = memory(d, "Z", NoiseModel.uniform_depolarizing(p)); e, n = pL(u, shots); print(f"  uniform basis=Z: {e:.2e} ({n})")
    for pe_frac in (0.5, 0.9):
        pe = p * pe_frac / (1 - pe_frac)      # erasure rate such that pe/(p+pe) = pe_frac
        ch = erasure_circuit(memory(d, "Z", NoiseModel.uniform_depolarizing(p)), pe); cu = strip_herald_detectors(ch)
        e_un, n_un = pL(cu, shots, corr=True); e_h0, n_h0 = pL(ch, shots, corr=False); e_h1, n_h1 = pL(ch, shots, corr=True)
        print(f"  erasure pe={pe:.1e} (frac {pe_frac}) basis=Z: unheralded {e_un:.2e} ({n_un}) | heralded uncorr {e_h0:.2e} ({n_h0}) | heralded corr {e_h1:.2e} ({n_h1})")
