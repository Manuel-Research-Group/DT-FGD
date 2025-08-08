#!/bin/bash
git submodule update --init --recursive

pushd FilteredGuidedDiffusion
if git apply --check ../fgd_variable_resolution.patch; then
    echo "Applying variable resolution patch now..."
    git apply ../fgd_variable_resolution.patch
else
    echo "Variable resolution patch cannot be applied or has already been applied. Skipping."
fi
popd

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