import torch
import numpy as np
from PIL import Image
def normalize_latents(latents):
    """
    Normalizes a latent tensor to be displayed as an image.
    
    Args:
        latents (torch.Tensor or np.ndarray): Latent tensor of shape (C, H, W), (H, W, C), or (B, C, H, W).
    
    Returns:
        PIL.Image: Normalized image.
    """

    # Convert to numpy if it's a torch tensor
    if isinstance(latents, torch.Tensor):
        latents = latents.detach().cpu().numpy()

    # Handle batch dimension (B, C, H, W)
    if latents.ndim == 4:
        latents = latents[0]  # Take the first batch element

    # Ensure shape is (H, W, C)
    if latents.shape[0] in [1, 4]:  # Channels-first (C, H, W)
        latents = np.transpose(latents, (1, 2, 0))  # Convert (C, H, W) → (H, W, C)
    
    elif latents.shape[-1] not in [1, 4]:  # Invalid shape
        raise ValueError(f"Unexpected latent shape: {latents.shape}")

    # Normalize to [0, 1]
    latents = (latents - latents.min()) / (latents.max() - latents.min() + 1e-8)  

    # Scale to [0, 255] and convert to uint8
    latents = (latents * 255).astype(np.uint8)

    # Convert to PIL image
    return Image.fromarray(latents)