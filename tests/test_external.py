"""Smoke tests for the slow paths; need external/magic-state-cultivation (scripts/fetch_external.sh) and the 'cultivation' extra."""
import json, pytest
from qmagic.paths import DATA, EXTERNAL
pytestmark = [pytest.mark.external, pytest.mark.skipif(not EXTERNAL.exists(), reason="external repo not fetched")]

def test_generate_and_sample_small_cultivation_circuit(tmp_path):
    """Generate the d1=3, d2=7, p=1e-3 circuit with Gidney's generator and sample it with the paper's runner (20k shots);
    the acceptance rate must agree with the shipped 5e7-shot run to within 20%."""
    from qmagic import make_circuits, cultivation
    files = make_circuits.make(tmp_path / "circ", 3, [7], [1e-3])
    assert len(files) == 1 and "d1=3" in files[0].name and "d2=7" in files[0].name
    import sinter
    meta = sinter.comma_separated_key_values(str(files[0]))
    agg = cultivation.worker((str(files[0]), 20_000, 0))
    res = cultivation.postprocess(meta, agg, str(files[0]))
    assert res["baseline_volume"] > 0 and res["q"] == 102 and len(res["curve"]) > 0
    ref = next(r for r in json.load(open(DATA / "cultiv_d2scan_d3_5e7.json")) if r["meta"]["d2"] == 7 and r["meta"]["p"] == 1e-3)
    keep = 1 - agg["discards"] / agg["shots"]; keep_ref = 1 - ref["raw"]["discards"] / ref["raw"]["shots"]
    assert abs(keep / keep_ref - 1) < 0.2
    assert abs(res["baseline_volume"] / ref["baseline_volume"] - 1) < 1e-6      # deterministic volume of the same circuit

def test_noise_models_import_and_memory_circuit():
    from qmagic import noise as NM
    from gen._chunk._noise import NoiseModel
    c = NM.memory(3, "Z", NoiseModel.uniform_depolarizing(1e-3)); assert c.num_detectors > 0
    cb = NM.memory(3, "Z", NM.BiasedNoiseModel(1e-3, 10)); assert cb.num_detectors == c.num_detectors
