"""Cultivation under biased noise: build Gidney's end2end circuit noiseless, noise it with BiasedNoiseModel (eta = 1 is the
uniform control), then sample every circuit with the desaturation sampler (qmagic.cultivation).
    python -m qmagic.cultiv_noise [shots_per_circuit=5e7] [procs=16]  -> external/.../out/noise_bias/*.stim, results/cultiv_noise_bias.json
"""
from qmagic.paths import DATA, RESULTS, EXTERNAL, find
import sys, pathlib, json
from qmagic import noise as NM
import stim
cultiv = NM.cultiv
from gen._chunk._noise import NoiseModel
def main(shots=50_000_000, procs=16):
    out = (EXTERNAL / "out/noise_bias"); out.mkdir(parents=True, exist_ok=True)
    for (d1, d2) in ((3, 7), (5, 11)):
        for eta in (1, 10, 100):
            p = 1e-3
            c = cultiv.make_end2end_cultivation_circuit(dcolor=d1, dsurface=d2, basis="Y", r_growing=d1, r_end=5, inject_style="unitary")
            model = NoiseModel.uniform_depolarizing(p) if eta == 1 else NM.BiasedNoiseModel(p, eta)
            cn = model.noisy_circuit_skipping_mpp_boundaries(c)
            name = f"c=end2end-inplace-distillation,p={p},noise=bias{eta},g=css,q={cn.num_qubits},b=Y,r={d1*2+5},r1={d1},d1={d1},r2=5,d2={d2}.stim"
            cn.to_file(out / name); print("wrote", name)
    from qmagic import cultivation
    cultivation.run_dir(out, "cultiv_noise_bias.json", shots, procs)

if __name__ == "__main__": main(int(float(sys.argv[1])) if len(sys.argv) > 1 else 50_000_000, int(sys.argv[2]) if len(sys.argv) > 2 else 16)
