# DT-FGD – Memory-Efficient Filter-Guided Diffusion with Domain Transform Filtering :sparkles:
<a href="docs/representative_250.png">
  <img src="docs/representative_250.png" alt="FGD vs dtFGD representative thumbnail" width="250" height="250" align="right">
</a>

A lightweight variant that replaces bilateral filtering with the efficient domain transform filter and introduces a normalization strategy for the guidance image’s latent representation.

*Memory‑Efficient **Filter‑Guided Diffusion** with Domain‑Transform Filtering*
<br clear="right"/>
---

## Requirements
- **NVIDIA GPU** with at least **12 GB VRAM** for full-resolution experiments.  
  (Lower-VRAM GPUs are supported with the `--decode-cpu` flag.)
- **NVIDIA CUDA Toolkit** — [Download and installation guide](https://developer.nvidia.com/cuda-downloads).
- **Conda** (Miniconda or Anaconda) — not strictly required, but the provided setup and environment files are designed for it, making installation significantly easier.  
  [Installation guide](https://docs.conda.io/en/latest/miniconda.html).
- **Native Linux environment** is strongly recommended.  
  WSL is not supported due to known compatibility issues with the Taichi module that may prevent execution.


## 1  Quick setup

```bash
git clone https://github.com/Manuel-Research-Group/dtFGD.git
cd dtfgd
git submodule update --init --recursive   # pulls original FGD implementation

conda env create -f environment.yml       # Torch 2.4, Diffusers 0.30
conda activate dtfgd
```
Or, using our bash script for initiating/running experiments for all test cases:

```bash
./setup_run.sh
```

---

## 2  Reproduce a paper figure

```bash
python run_experiment.py --config configs/fig1_dog.json
```

Outputs: `results/fgd.png`, `results/dtfgd.png`, `results/comparison.png`.

### Script for running directly
To reproduce ONLY the representative figure from the paper (dog portrait), run:
```bash
./replicate_representative.sh
```

---

## 3  Run your own experiment

```bash
python run_experiment.py \
  --prompt "A Van Gogh style fox playing guitar" \
  --guide assets/woman_blueheadband.png
  --sigma_s 4 --sigma_r 0.25 --detail 1.4 --t_end 20 \
  --plot-intermediate
```

---

## 4  Folder structure

```

DT-FGD/
├── assets/ # Example images used in configs and docs
│ ├── bread_2.png
│ ├── gauss.png
│ ├── monalisa.png
│ ├── red_hat.jpg
│ └── woman_blueheadband.png
│
├── configs/ # JSON configs for reproducing paper figures
│ ├── fig1_portrait_of_a_dog.json
│ ├── fig3_monalisa_dog.json
│ ├── fig7_cat_red_hat.json
│ ├── fig9_gauss_bird.json
│ ├── fig10_steak_bread.json
│ └── fig10_steak_bread1192.json
│
├── FilteredGuidedDiffusion/ # Original FGD submodule (MIT license)
│ ├── cbilateral.py
│ ├── diffusionModel.py
│ ├── FGD.py
│ ├── figures/
│ ├── imgs/
│ └── README.md
│
├── results/ # Generated outputs from experiments
│ ├── *_fgd.png
│ ├── *_dtfgd.png
│ └── *_comparison.png
│
├── src/ # Main implementation of DT-FGD
│ ├── diffusionModel.py # Modified diffusion pipeline
│ ├── dtFGD.py # Domain Transform–based guided diffusion
│ ├── ncdt.py # Normalized convolution domain transform
│ └── util.py
│
├── environment.yml # Conda environment definition
├── fgd_variable_resolution.patch # Patch for FGD resolution handling
├── guided_filter_dt.png # Illustration of guided filtering
├── LICENSE.txt
├── README.md
├── run_experiment.py # Main CLI entry point
└── setup_run.sh # Helper script to run all test cases
```

---

## 5  Licence

* All code in this repository is released under **MIT License**.
* See LICENSE.txt for the full text.

### Original FGD code

This project links to the reference implementation by Gu et al. at
FilteredGuidedDiffusion/, which is licensed under MIT in their repository.
The submodule retains its original copyright and license.
Please run git submodule update --init after cloning.


---

Happy experimenting :rocket:
