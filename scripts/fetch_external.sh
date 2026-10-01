#!/usr/bin/env bash
# Clone Gidney's magic-state-cultivation repository (Apache-2.0) at the commit used in this study and apply our two-line patch
# (PAULI_CHANNEL support in the volume utility). Needed only for re-running cultivation simulations (data/cultiv_*.json are shipped).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/external/magic-state-cultivation"
if [ ! -d "$DEST" ]; then
  git clone https://github.com/Strilanc/magic-state-cultivation.git "$DEST"
fi
cd "$DEST"
git checkout -q 871e68ff6df2f75190b1bfd6351459d1b5a037e3
git apply --check "$ROOT/patches/gidney_magic-state-cultivation_871e68f.patch" 2>/dev/null && git apply "$ROOT/patches/gidney_magic-state-cultivation_871e68f.patch" || echo "patch already applied"
pip install "sinter==1.16.0" "chromobius==1.1.1" pygltflib networkx
echo "external repository ready at $DEST"
