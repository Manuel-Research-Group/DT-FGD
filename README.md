# dtFGD – Replicability Pack :sparkles:

*Memory‑Efficient **Filter‑Guided Diffusion** with Domain‑Transform Filtering*  
One‑command reproduction of every qualitative figure **plus** a flexible
CLI for your own prompts.

---

## 1  Quick setup

```bash
git clone https://github.com/Manuel-Research-Group/dtFGD.git
cd dtfgd
git submodule update --init --recursive   # pulls original FGD implementation

conda env create -f environment.yml       # Torch 2.4, Diffusers 0.30
conda activate dtfgd
```

*GPU is optional.*

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

* All **new** code in this repository: **CC‑BY‑NC 4.0**.  

### Original FGD code

This project links to, but **does not redistribute**, the reference
implementation released by Gu et al.  Please ensure you `git submodule
update --init` after cloning.  Copyright remains with the original
authors; all usage must follow their terms.


---

Happy experimenting :rocket:
