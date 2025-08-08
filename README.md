# DT-FGD – Memory-Efficient Filter-Guided Diffusion with Domain Transform Filtering :sparkles:
A lightweight variant that replaces bilateral filtering with the efficient domain transform filter and introduces a normalization strategy for the guidance image’s latent representation.

*Memory‑Efficient **Filter‑Guided Diffusion** with Domain‑Transform Filtering*

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

| Scenario                  | Flag(s)                  | Peak VRAM | Speed impact |
|---------------------------|--------------------------|-----------|--------------|
| 8 GiB GPU (save memory)   | `--decode-cpu`           | ↓ ≈ 5 GB  | +15 %        |

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

---

## 3  Run your own experiment

```bash
python run_experiment.py \
  --prompt "A Van Gogh style fox playing guitar" \
  --sigma_s 4 --sigma_r 0.25 --detail 1.4 --t_end 20 \
  --plot-intermediate
```

**New flags introduced**

| Flag                 | What it does                                           |
|----------------------|--------------------------------------------------------|
| `--plot-intermediate`| Shows per‑step visual diagnostics (slow, for research) |
| `--decode-cpu`       | Keeps VAE on CPU – cuts GPU memory        |

---

## 4  Folder structure

```
|-- run_experiment.py      ← main entry‑point (updated)
|-- src/
|   |-- diffusionModel.py  ← modified to support the new flags
|   |-- dtFGD.py, ncdt.py  ← your algorithm
```

---

## 5  Tests / CI

The GitHub Action in `.github/workflows/smoke.yml` installs the environment
and runs a **CPU‑only** quick inference to ensure the repo is self‑contained.
Public repositories incur **zero** cost on GitHub‑hosted runners.

---

## 6  Licence

* All code in this repository is released under **MIT License**.
* See LICENSE.txt for the full text.

### Original FGD code

This project links to the reference implementation by Gu et al. at
FilteredGuidedDiffusion/, which is licensed under MIT in their repository.
The submodule retains its original copyright and license.
Please run git submodule update --init after cloning.


---

Happy experimenting :rocket:
