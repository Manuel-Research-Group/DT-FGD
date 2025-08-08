#!/usr/bin/env bash
set -euo pipefail

git submodule update --init --recursive

# quick env bootstrap (create if missing; else reuse)
if ! conda env list | grep -qE '^\s*dtfgd\s'; then
  conda env create -f environment.yml
fi

conda run -n dtfgd --live-stream python run_experiment.py --config configs/fig1_portrait_of_a_dog.json
echo "✓ Representative figure generated at ./results/woman_blueheadband_fgd.png"