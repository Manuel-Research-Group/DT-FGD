#!/bin/bash
git submodule update --init --recursive
# Ensure conda commands are available in scripts

# Only run this manually once if the env doesn't exist
if ! conda env list | grep -q 'dtfgd'; then
    echo "Creating conda environment 'dtfgd'..."
    conda env create -f environment.yml       
else
    echo "Conda environment 'dtfgd' already exists. Skipping creation."
fi




conda run -n dtfgd --live-stream python run_experiment.py --config configs/fig1_portrait_of_a_dog.json
conda run -n dtfgd --live-stream python run_experiment.py --config configs/fig3_monalisa_dog.json
conda run -n dtfgd --live-stream python run_experiment.py --config configs/fig7_cat_red_hat.json
conda run -n dtfgd --live-stream python run_experiment.py --config configs/fig9_gauss_bird.json
conda run -n dtfgd --live-stream python run_experiment.py --config configs/fig10_steak_bread.json
conda run -n dtfgd --live-stream python run_experiment.py --config configs/fig10_steak_bread1192.json