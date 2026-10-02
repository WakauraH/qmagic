# qmagic — reproduction package

Code, data and reproduction scripts for

> H. Wakaura and T. Tanimae, *Where magic-state cultivation pays in early fault-tolerant quantum computing: a single-clock
> cost model of logical error mitigation and magic-state supply* (2026).

The package evaluates four early-fault-tolerant architectures (15-to-1 distillation or magic-state cultivation, each with
and without logical probabilistic error cancellation) on one clock, one noise calibration and one precision target, and
reports physical qubits, sampling overhead and the largest feasible circuit size.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate      # Python >= 3.11
pip install -e ".[dev]"                                 # numpy, scipy, matplotlib, stim 1.16.0, pymatching 2.4.0, pytest
python -m pytest -m quick                               # 12 regression tests, < 1 min
qmagic-reproduce --verify                               # ~1 min: regenerates every table/figure of the paper into results/
                                                        # and checks the JSON/CSV outputs against reference/
```

`qmagic-reproduce` runs, in order: the Litinski factory reconstruction, the three parameter-set builders, the claims and
sensitivity sweeps, the acceptance-cutoff and hardware-Λ analyses, the exact PEC calculation, the gap analysis, the
noise-model comparison, and the figures. All of it is deterministic; `--verify` compares the regenerated files with
`reference/` to a relative tolerance of 1e-6 (1e-2 for the optional Monte-Carlo PEC demo, `--mc`).

## What is shipped and what is regenerated

| directory | content |
|---|---|
| `data/` | inputs that the fast path does not regenerate: surface-code calibrations (`calib_*.json`, Stim + PyMatching), cultivation reruns of Gidney's circuits (`cultiv_*.json`), Gidney 2024 published statistics (`gidney2024_*`), base parameter set `params.json` |
| `reference/` | the outputs of `qmagic-reproduce` as used in the paper (JSON/CSV), for `--verify` |
| `results/` | created by `qmagic-reproduce` (ignored by git) |
| `src/qmagic/` | the package; `tests/` pytest; `scripts/fetch_external.sh`, `patches/` for the slow paths |

Where the paper's numbers come from:

| paper | file(s) |
|---|---|
| Table II, III, VI; Fig. 3 | `claims_v3.json`, `sensitivity_v3.json` (from `params_v3.json`) |
| Fig. 1, Sec. III/IV floors | `params_v3.json` (distillation rows), `factories_2level_d11.json`, `calib_perp.json`, `data/cultiv_enum_d3.json` (exact d₁=3 cultivation-stage floor) |
| Fig. 2 | `data/cultiv_*.json`, `data/gidney2024_fig1_points.json` |
| Sec. V.D, Fig. 4 | `gap_qem.json`, `cutoff` step (stdout) |
| Sec. V.E, Table V, Fig. 5 | `q3_nmax.csv`, `noise_costmodel.json` |
| Appendix D | `pec_exact.json` (`pec_demo.json` with `--mc`) |

## Slow paths (optional)

The calibrations and cultivation reruns in `data/` were produced with Gidney's open-source cultivation repository
(Apache-2.0), which is not vendored. To re-run them:

```bash
pip install -e ".[dev,cultivation]"     # + sinter 1.16.0, chromobius 1.1.1
scripts/fetch_external.sh               # clones Strilanc/magic-state-cultivation @ 871e68f and applies patches/
qmagic-reproduce --calib                # surface-code memory calibrations with sequential stopping (~6 min on 16 cores);
                                        # writes results/calib_*_seq.json next to the shipped data/calib_*.json
qmagic-reproduce --cultiv               # cultivation under biased noise (hours)
python -m qmagic.cultiv_enum 5          # exact d1=3 cultivation-stage floor by error-set enumeration (5 min; weight 4: 0.1 s)
python -m qmagic.cultivation <circuit_dir> <out.json> <shots> [procs]   # cultivation sampling runner (d2 scans, deep tails)
python -m pytest -m external            # import check against the external repository
```

Statistical reruns produce new samples; they will not match `data/` bit for bit, only within Poisson error.

## Model in one paragraph

Logical error per patch and per $d$ rounds $p_L=A(p/p_{\rm th})^{s(d+1)/2}$; data block $2d^2 r_{\rm route} n_L$ qubits;
$\pi$ logical operations per step of $d$ cycles, $u=\pi/n_L$; idle error $\varepsilon_{\rm data}=(N/\pi)\,\kappa n_L p_L$;
$T$ demand $\rho=\pi f_T$ per step; consumed $T$ error $\varepsilon_T=\varepsilon_{\rm exit}+0.75p_L(d)$; sources limited to
$d_{\rm req}\le d_{\max}$. No mitigation: $\Sigma\le N_e=10^{-3}$. Mitigation: $\Gamma=e^{4\Sigma}\le100$ and bias
$\Sigma r\le N_e$ with $r=10^{-3}$. Options `kappa_mode="route"` and `growth_rounds` charge routing-tile error and growth
of $d_2<d$ cultivated states (sensitivity). See `src/qmagic/costmodel.py`.

## License

MIT (code and data of this package). The external cultivation repository is Apache-2.0 and is fetched, not redistributed.
