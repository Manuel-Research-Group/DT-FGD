#!/bin/bash
git submodule add -f https://github.com/jaclyngu/FilteredGuidedDiffusion.git
# Ensure conda commands are available in scripts

# Only run this manually once if the env doesn't exist
if ! conda env list | grep -q 'dtfgd'; then
    echo "Creating conda environment 'dtfgd'..."
    conda env create -f environment.yml       
else
    echo "Conda environment 'dtfgd' already exists. Skipping creation."
fi


# Run experiments
conda run -n dtfgd --live-stream python run_experiment.py --config configs/fig1_dog.json
conda run -n dtfgd --live-stream python run_experiment.py --config configs/fig3_dog.json
conda run -n dtfgd --live-stream python run_experiment.py --config configs/fig7_cat.json
conda run -n dtfgd --live-stream python run_experiment.py --config configs/fig9_bird.json
conda run -n dtfgd --live-stream python run_experiment.py --config configs/fig10_bread.json
