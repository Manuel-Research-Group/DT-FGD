import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from tqdm.auto import tqdm
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils import load_image
from PIL import Image
import json
from diffusers.utils.torch_utils import randn_tensor
from pytorch_lightning import seed_everything
import numpy as np
import logging
import pathlib, sys, importlib.util
root = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root / "src"))

orig_path = root / "FilteredGuidedDiffusion"
sys.path.insert(0, str(orig_path))
spec = importlib.util.spec_from_file_location("FGD", orig_path / "FGD.py")
FGD_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(FGD_mod)
import FGD

import time
from huggingface_hub import login
import matplotlib.pyplot as plt
from util import *
import os

log = logging.getLogger("lightning_fabric")
log.propagate = False
log.setLevel(logging.ERROR)

# diffusion pipeline adapted from https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/stable_diffusion.ipynb
class diffusionModel():
    def __init__(self, height=512,width=512,batch_size=1,guidance_scale=7.5,num_steps=50,scheduler='ddpm',version='1.5', use_ema=True, device=None, plot_intermediate=False, input_release=True, decode_cpu=False):
        self.height = height
        self.width = width
        self.batch_size = batch_size
        self.guidance_scale = guidance_scale
        self.num_steps = num_steps
        self.plot_intermediate = plot_intermediate
        
        self.scheduler_type = scheduler
        self.version = version
        self.decode_cpu = decode_cpu
        self.input_release = input_release
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device



        try:
            HF_TOKEN = os.environ['HF_TOKEN']   
            print("Token: ", HF_TOKEN)
        except:
            ...

        if version == '1.4':
            model_version = "CompVis/stable-diffusion-v1-4"
        elif version == '1.5':
            model_version = "runwayml/stable-diffusion-v1-5"
        elif version == '2.0':
            model_version = "stabilityai/stable-diffusion-2"
        elif version == '2.1':
            model_version = "stabilityai/stable-diffusion-2-1"
        elif version == 'sd-turbo':
            if(not HF_TOKEN):
                Exception(f'{version} needs hugging_face_api_key.')
            
            login(token=HF_TOKEN)
            model_version = "stabilityai/sd-turbo"
        else:
            raise Exception(f'{version} not supported. Select from [1.4, 1.5, 2.0, 2.1]')

         # 1. Load the autoencoder model which will be used to decode the latents into image space.
        self.vae = AutoencoderKL.from_pretrained(model_version,subfolder="vae", use_safetensors=True)
        # 2. Load the tokenizer and text encoder to tokenize and encode the text.
        self.tokenizer = CLIPTokenizer.from_pretrained(model_version, subfolder="tokenizer", use_safetensors=True, clean_up_tokenization_spaces=True)
        self.text_encoder = CLIPTextModel.from_pretrained(model_version, subfolder="text_encoder", use_safetensors=True)

        # 3. The UNet model for generating the latents.
        if use_ema:
            self.unet = UNet2DConditionModel.from_pretrained(model_version, subfolder="unet", use_safetensors=True)
        else:
            if version == '2.0' or version == '2.1':
                raise Exception(f"SD version {version} non-ema weights not availible")
            self.unet = UNet2DConditionModel.from_pretrained(model_version, subfolder="unet", variant="non_ema", use_safetensors=True)
        self.use_ema = use_ema

        # 4. Scheduler
        if scheduler == 'ddpm':
            self.scheduler = DDPMScheduler.from_pretrained(model_version, subfolder="scheduler", use_safetensors=True)
        else:
            raise ValueError(f"current schedule type {scheduler}: select from ['ddpm']")
        self.scheduler.set_timesteps(num_steps)

        if(self.decode_cpu):
            self.vae = self.vae.to('cpu')
        else:
            self.vae = self.vae.to(self.device)
        self.text_encoder = self.text_encoder.to(self.device)
        self.unet = self.unet.to(self.device)

        self.inital_latents = None
        self.text_embeddings = None
        self.prompt = None
        self.latent_initialization = None
        
        self.fgd_incremented_time_ns = 0

    def set_prompt(self, prompt, negative_prompt=""):
        self.prompt = prompt
        text_input = self.tokenizer(prompt, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")

        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]
        text_embeddings = torch.cat([text_embeddings] * self.batch_size)

        max_length = text_input.input_ids.shape[-1]
        uncond_input = self.tokenizer([negative_prompt] * self.batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
        with torch.no_grad():
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        self.text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    def initialize_latents_random(self, seed=10):
        self.initial_latents= self.get_random_latents(seed)
        self.latent_initialization = seed

    def get_random_latents(self, seed = 10):
        generator = torch.manual_seed(seed)
        initial_latents = torch.randn(
            (self.batch_size, self.unet.config.in_channels, self.height // 8, self.width // 8),
            generator=generator,
        )
        initial_latents = initial_latents.to(self.device)
        initial_latents = initial_latents * self.scheduler.init_noise_sigma
        return initial_latents


    # vanilla stable diffusion
    def generate(self, seed=10):
        seed_everything(seed)
       
        latents = self.initial_latents
        timesteps = self.scheduler.timesteps
        guidance_scale = self.guidance_scale

        for param in self.unet.parameters():
            param.requires_grad = False

        for t in tqdm(timesteps, leave=False):
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=self.text_embeddings).sample

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = self.scheduler.step(noise_pred, t, latents).prev_sample.detach()
        return self.decode_latents(latents)

    # stable diffusion with FGD
    def generate_FGD(self, filter:FGD, seed=10, return_time=False):
        start = time.time_ns()
        filter.reset()
        seed_everything(seed)

        latents = self.initial_latents
        timesteps = self.scheduler.timesteps
        s = self.num_steps

        guidance_scale = self.guidance_scale

        for param in self.unet.parameters():
            param.requires_grad = False

        for t in tqdm(timesteps, leave=False):
            latents.requires_grad_(True)
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            with torch.no_grad():
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=self.text_embeddings).sample
            
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # ------------------
            # FGD stuff is below
            # ------------------
            output = self.scheduler.step(noise_pred, t, latents)
            compute_time = time.time_ns()
            latents_new = output.prev_sample
            st = output.pred_original_sample

            xt_filtered = self.filter_st(latents.detach(), st, noise_pred, filter, t, s)
            s -= 1 
            compute_time = time.time_ns()-compute_time
            self.fgd_incremented_time_ns += compute_time
            
            latents = xt_filtered.detach()
        end = time.time_ns()

        diffusion_time_s = (end-start)*1e-9

        start = time.time_ns()
        decoded_latent = self.decode_latents(latents)
        end = time.time_ns()

        decoding_time_s = (end-start)*1e-9
        if(return_time):
            return diffusion_time_s, decoding_time_s, decoded_latent
        return decoded_latent
        

        
    def filter_st(self, xt, st, noise_pred, filter, t, s):
        st = st.clone()
        params = [st, None, self.scheduler, t]
        
        if self.num_steps-s < filter.norm_steps:
            guidance = filter.get_guidance_normalized(*params)
        if ((self.num_steps-s) < filter.norm_steps):
            guidance = filter.get_guidance_normalized(*params)
        else:
            guidance = filter.get_guidance(*params)
        if len(guidance) == 3:
            weight, d, st_low = guidance
        else:
            weight, d = guidance
            st_low = None

        st_filtered = st.clone()
        if s > filter.t_end:
            st_filtered += d*weight
        else:
            st_filtered = st

        if self.plot_intermediate:            
            # Decode images from both filters
            decoded_latents_1 = normalize_latents(st.clone().detach())  # Replace with second filter output
            decoded_latents_2 = normalize_latents(st_filtered.clone().detach())
            difference_vector = normalize_latents((d*weight).clone().detach())
            if st_low is not None:
                
                st_low_latent = normalize_latents(st_low.clone().detach())

            decoded_latents_1 = np.array(decoded_latents_1)
            decoded_latents_2 = np.array(decoded_latents_2)
            difference_vector = np.array(difference_vector)
            if st_low is not None:
                st_low_latent = np.array(st_low_latent)
            

            # Create a figure with two subplots
            fig, axes = plt.subplots(1, 4 if st_low is not None else 3, figsize=(15, 5))

            # Plot first filtered image
            axes[0].imshow(decoded_latents_1)
            axes[0].set_title("x0 estimation")
            axes[0].axis("off")

            # Plot second filtered image
            axes[1].imshow(decoded_latents_2)
            axes[1].set_title("x0 estimation + guidance")
            axes[1].axis("off")

            # Plot second filtered image
            axes[2].imshow(difference_vector)
            axes[2].set_title("delta x (f(g)-f(x))")
            axes[2].axis("off")

            if st_low is not None:
                # Plot second filtered image
                axes[3].imshow(st_low_latent)
                axes[3].set_title("filtered st")
                axes[3].axis("off")

            plt.show()

        assert self.scheduler_type == 'ddpm', "released FGD implementation only supports DDPM"

        prev_t = self.scheduler.previous_timestep(t)
        alpha_prod_t = self.scheduler.alphas_cumprod[t]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_t] if prev_t >= 0 else self.scheduler.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t
        current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t

        pred_prev_sample = pred_original_sample_coeff * st_filtered + current_sample_coeff * xt
        variance_noise = randn_tensor(
            noise_pred.shape, device=self.device, dtype=noise_pred.dtype
        )

        variance = (self.scheduler._get_variance(t) ** 0.5) * variance_noise
        xt_filtered = pred_prev_sample + variance
            
        return xt_filtered 

    def decode_latents(self, latents):
        if(self.decode_cpu):
            latents = latents.detach().cpu()

        with torch.no_grad():
            image = self.vae.decode(latents * (1 / self.vae.config.scaling_factor)).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]
        if latents.shape[0] == 1:
            return pil_images[0]
        else:
            return pil_images
        
    def encode_image(self, image):
        if isinstance(image, str):
            guide_image = load_image(image)
        else:
            guide_image = image
        processor = VaeImageProcessor(self.vae.config)
        
        if(self.decode_cpu):
            self.vae.to('cpu')
        else:
            self.vae.to(self.device)

        guide_processed = processor.preprocess(guide_image,self.height,self.width)
        if(self.decode_cpu):
            guide_processed = guide_processed.to('cpu')
        else:
            guide_processed = guide_processed.to(self.device)

        with torch.no_grad():
            guide_latent = self.vae.encode(guide_processed).latent_dist.sample()*self.vae.config.scaling_factor
        
        if(self.input_release):
            guide_processed = guide_processed.detach()
            del guide_processed

        if(self.decode_cpu):
            guide_latent = guide_latent.to(self.device)

        return guide_latent
    
    def get_params(self):
        params = {
            'prompt':self.prompt,
            'version':self.version,
            'use_ema':self.use_ema,
            'scheduler':self.scheduler_type,
            'initialization':self.latent_initialization,
        }
        return params
    def __str__(self):
        return (json.dumps(self.get_params(), indent=2))