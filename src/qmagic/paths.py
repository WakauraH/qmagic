"""Locations of shipped data, generated results, and the optional external Gidney repository."""
import os, pathlib
ROOT = pathlib.Path(os.environ.get("QMAGIC_ROOT", pathlib.Path(__file__).resolve().parents[2]))
DATA = ROOT / "data"; RESULTS = ROOT / "results"; EXTERNAL = ROOT / "external" / "magic-state-cultivation"
RESULTS.mkdir(parents=True, exist_ok=True)
def find(name):
    """Resolve an input file: results/ (regenerated) first, then data/ (shipped), then cwd."""
    for base in (RESULTS, DATA, pathlib.Path.cwd()):
        p = base / name
        if p.exists(): return p
    raise FileNotFoundError(f"{name} not found in {RESULTS}, {DATA} or cwd")
