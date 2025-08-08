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
popd >/dev/null

# Ensure conda is available
if ! command -v conda >/dev/null 2>&1; then
  echo "❌ conda not found in PATH. Please install Miniconda/Anaconda and retry."
  exit 1
fi

# Create or update env
if conda env list | grep -qE '^\s*dtfgd\s'; then
  echo "[env] Updating existing 'dtfgd'..."
  conda env update --prune -f environment.yml
else
  echo "[env] Creating 'dtfgd'..."
  conda env create -f environment.yml
fi

echo "[run] Executing experiments..."
# conda run -n dtfgd --live-stream python run_experiment.py --config configs/fig1_portrait_of_a_dog.json
# conda run -n dtfgd --live-stream python run_experiment.py --config configs/fig3_monalisa_dog.json
conda run -n dtfgd --live-stream python run_experiment.py --config configs/fig7_cat_red_hat.json
# conda run -n dtfgd --live-stream python run_experiment.py --config configs/fig9_gauss_bird.json
# conda run -n dtfgd --live-stream python run_experiment.py --config configs/fig10_steak_bread.json
# conda run -n dtfgd --live-stream python run_experiment.py --config configs/fig10_steak_bread1192.json

echo "✓ All runs complete. See ./results/"
