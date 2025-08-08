import torch
from tqdm.auto import tqdm
import json
import numpy as np
import time
import matplotlib.pyplot as plt
from ncdt import NCDomainTransform
from util import *

class dtFGD():
    def __init__(self, diffusionModel, guide_image, detail=1.2, sigmas=[3,3,0.3], num_iterations=3, t_end=15, norm_steps=0, device="cuda", dt_filter_cls=NCDomainTransform):
        self.guide_image = guide_image
        self.detail = detail
        self.norm_steps = norm_steps
        self.model = diffusionModel
        self.bilateral_matrix_4d = None
        self.num_iterations = num_iterations
        
        self.guide_latent = None
        self.guide_latent_repr = None
        self.guide_structure = None
        self.guide_structure_normalized = None
        

        self.init_guide_latent = None
        self.init_guide_structure = None
        self.init_guide_stucture_normalized=None
        
        self.init_bilateral_matrix_4d=None

        self.filter_apply_ns = []
        
        self.t_end = t_end

        self.sigmas = sigmas

        self.device = device

        self.filter = None
        self.dt_filter_cls = dt_filter_cls
        self.filter = dt_filter_cls(self.sigmas, self.num_iterations, device=device)

        
        self.set_guide_image(guide_image)

    def set_ST(self, detail=1.6, recompute_matrix=True, sigmas=[3,3,0.3]):
        if recompute_matrix:
            self.set_bilateral_matrix(sigmas)
        self.detail = detail
        self.t_end = 15
        self.norm_steps = 50


    def reset(self):
        self.init_guide_latent = self.guide_latent
        self.guide_structure = self.init_guide_structure
        self.guide_structure_normalized = self.init_guide_structure_normalized
        self.bilateral_matrix_4d = self.init_bilateral_matrix_4d

    def set_guide_image(self, guide_image):
        self.guide_latent = self.model.encode_image(guide_image)
        
        self.latent_shape = self.guide_latent.detach().cpu().permute(0, 2, 3, 1).numpy()[0].shape
        self.guide_image = guide_image
        if self.sigmas != None:
            self.set_bilateral_matrix(self.sigmas)

    def set_bilateral_matrix(self,sigmas):
        assert len(sigmas)==2 or len(sigmas)==3, "sigmas has invalid number of entries (either 2 or 3)"
        sigmas = np.array(sigmas).astype(np.double)
        if len(sigmas) == 2:
            sigmas = np.insert(sigmas, 1, sigmas[0])
        self.sigmas = sigmas

        self.filter_preprocess(self.guide_latent)
        

        guide_structure_latent = self.filter(self.guide_latent)

        guide_mean = torch.mean(guide_structure_latent, (2,3), keepdim=True)
        guide_std = torch.std(guide_structure_latent, (2,3), keepdim=True)

        self.guide_structure_normalized = (guide_structure_latent - guide_mean) / guide_std
        self.guide_structure = guide_structure_latent

        self.init_guide_structure = self.guide_structure
        self.init_guide_structure_normalized=self.guide_structure_normalized
        self.init_bilateral_matrix_4d = self.bilateral_matrix_4d

        guide_filtered = normalize_latents(self.guide_structure)
        guide_filtered.save('./guided_filter_dt.png')

        self.sigmas = sigmas.tolist()

    def get_guide_structure(self):
        return self.guide_structure.cpu().detach().numpy()

    def filter_preprocess(self, tensor):
        self.filter.preprocess(tensor)

    def get_residual_structure(self, latents):
        start = time.time_ns()*1e-9
        current_structure = self.filter(latents)
        end = time.time_ns()*1e-9
        self.filter_apply_ns.append(end-start)
        
        d_structure = self.guide_structure - current_structure

        return d_structure
    
    def get_structure(self, latents):
        start = time.time_ns()*1e-9
        structure = self.filter(latents)
        end = time.time_ns()*1e-9
        self.filter_apply_ns.append(end-start)
        return structure

    def get_guidance(self, latents, input_latents, scheduler, t):
        guide_low = self.guide_structure
        
        st_low = self.get_structure(latents)
        st_high = latents - st_low

        weight = self.detail

        d = guide_low - st_low
        
        return weight, d, st_low

    def get_guidance_normalized(self, latents, input_latents, scheduler, t):
        current_structure = self.get_structure(latents)
        guide_structure = self.guide_structure
            
        current_mean = torch.mean(current_structure, (2,3), keepdim=True)
        current_std = torch.std(current_structure, (2,3), keepdim=True)

        guide_structure_renormalized = self.guide_structure_normalized * current_std + current_mean
        d_structure_renormalized = guide_structure_renormalized - current_structure

        residual_score = torch.mean(torch.abs(d_structure_renormalized)) 

        weight = self.detail

        return weight, d_structure_renormalized, guide_structure_renormalized
       
    def get_params(self):
        params = {
            'guide image':self.guide_image,
            'detail':self.detail,
            'sigmas':self.sigmas,
            't_end':self.t_end,
            'norm steps':self.norm_steps,
        }
        return params
    def __str__(self):
        return (json.dumps(self.get_params(), indent=2))