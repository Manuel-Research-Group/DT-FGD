#!/usr/bin/env python
"""
Run a dtFGD ⇆ original‑FGD comparison or a standalone dtFGD generation.

Examples
--------
# reproduce Fig. 1 with GPU
python run_experiment.py --config configs/fig1_dog.json

# low‑VRAM laptop
python run_experiment.py --config configs/fig1_dog.json --decode-cpu

# ad‑hoc prompt
python run_experiment.py --prompt "A watercolor fox" --sigma_s 4 --plot-intermediate
"""
import argparse, json, pathlib, sys, time, importlib.util
from PIL import Image
import torch
import matplotlib.pyplot as plt

# ----- repo paths ------------------------------------------------------------
root = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(root / "src"))

from dtFGD import dtFGD
from diffusionModel import diffusionModel

# original FGD from git sub‑module
orig_path = root / "FilteredGuidedDiffusion"
sys.path.insert(0, str(orig_path))
spec = importlib.util.spec_from_file_location("FGD", orig_path / "FGD.py")
FGD_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(FGD_mod)


def load_json(fp: str | pathlib.Path) -> dict:
    with open(fp) as f:
        return json.load(f)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--config", help="JSON with experiment parameters")
    # core parameters
    p.add_argument("--prompt", default="A high‑res photo of a cute corgi")
    p.add_argument("--guide",  default="assets/red_hat.jpg")
    p.add_argument("--initial_seed", type=int, default=1, help="torch seed for generation")
    p.add_argument("--seed", type=int, default=10, help="seed for SD latent init")
    # filter parameters
    p.add_argument("--sigma_s", type=float, default=3)
    p.add_argument("--sigma_r", type=float, default=0.3)
    p.add_argument("--detail",  type=float, default=1.2)
    p.add_argument("--t_end",   type=int, default=15)
    # SD parameters
    p.add_argument("--sd_version", default="1.5")
    p.add_argument("--steps", type=int, default=50, help="diffusion steps")
    p.add_argument("--image_size", type=int, nargs=2, default=[512, 512], help="Image size (H, W)")
    # resource flags
    p.add_argument("--decode-cpu", action="store_true", help="move VAE to CPU")
    p.add_argument("--plot-intermediate", action="store_true", help="show per‑step visuals")
    p.add_argument("--no-input-release", action="store_true",
                   help="keep intermediate tensors on GPU (debugging)")
    # misc
    p.add_argument("--outdir", default="results", help="directory to write PNGs")
    return p.parse_args()

def main():
    args = parse_args()

    # ---- override with JSON config ------------------------------------------
    if args.config:
        cfg = load_json(args.config)
        for k, v in cfg.items():
            setattr(args, k.replace('-', '_'), v)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ---- diffusion model -----------------------------------------------------
    model = diffusionModel(
        height=512,
        width=512,
        num_steps=args.steps,
        scheduler='ddpm',
        use_ema=True,
        version=args.sd_version,
        plot_intermediate=args.plot_intermediate,
        decode_cpu=args.decode_cpu,
        device=device,
    )
    model.initialize_latents_random(args.initial_seed)

    model.set_prompt(args.prompt)

    guide_img = Image.open(args.guide)

    # original FGD
    fgd = FGD_mod.FGD(model, guide_img,
                      detail=args.detail,
                      sigmas=[args.sigma_s, args.sigma_s, args.sigma_r])

    # our approach dtFGD
    dt_filter = dtFGD(model, guide_img,
                      detail=args.detail,
                      sigmas=[args.sigma_s, args.sigma_s, args.sigma_r],
                      t_end=args.t_end,
                      num_iterations=3,
                      device=device)

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(exist_ok=True)

    # ----------------- run ----------------------------------------------------
    t0 = time.time()
    img_orig = model.generate_FGD(fgd, seed=args.seed)
    t_orig = time.time() - t0
    img_orig.save(outdir / "fgd.png")

    t0 = time.time()
    img_dt = model.generate_FGD(dt_filter, seed=args.seed)
    t_dt = time.time() - t0
    img_dt.save(outdir / "dtfgd.png")

    # side-by-side
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(img_orig)
    ax[0].set_title(f"FGD")
    ax[0].axis("off")
    ax[1].imshow(img_dt)
    ax[1].set_title(f"DT-FGD")
    ax[1].axis("off")
    side = outdir / "comparison.png"
    plt.tight_layout()
    fig.savefig(side, dpi=200)

    # Show the figure to the user
    plt.show()

    print(f"✓ Results saved in {outdir}/")

if __name__ == "__main__":
    main()
