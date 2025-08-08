#!/usr/bin/env bash
set -euo pipefail

echo "[init] Ensuring submodule..."
git submodule update --init --recursive

echo "[patch] Trying variable-resolution patch..."
pushd FilteredGuidedDiffusion >/dev/null
if git apply --check ../fgd_variable_resolution.patch 2>/dev/null; then
  git apply ../fgd_variable_resolution.patch
  echo "✓ Patch applied."
else
  echo "ℹ️ Patch skipped (already applied or not applicable)."
fi
popd

# quick env bootstrap (create if missing; else reuse)
if ! conda env list | grep -qE '^\s*dtfgd\s'; then
  conda env create -f environment.yml
fi

conda run -n dtfgd --live-stream python run_experiment.py --config configs/fig1_portrait_of_a_dog.json
echo "✓ Representative figure generated at ./results/woman_blueheadband_fgd.png"