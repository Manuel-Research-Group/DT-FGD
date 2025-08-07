import torch
import torch.nn.functional as F
import numpy as np

class NCDomainTransform:
    def __init__(self, sigmas, num_iterations=3, device='cuda'):
        """
        Enhanced NC filter implementation with anisotropic filtering and optional normalization.
        
        Args:
            sigma_s (tuple/list): (sigma_sx, sigma_sy) spatial std deviations for x and y.
                                  If a single float is given, applies isotropically.
            sigma_r (float): Range standard deviation.
            num_iterations (int): Number of iterations.
            device (str or torch.device): Device (default: input image device).
            normalize_joint (bool): If True, normalizes the joint image to [0, 1] in preprocess.
        """
        if len(sigmas) == 2:
            sigmas = np.insert(sigmas, 1, sigmas[0])
        elif len(sigmas) != 3:
            raise AssertionError(f"Len of sigmas list({len(sigmas)}) is not 2 nor 3, not supported.")

        self.sigma_sx, self.sigma_sy = sigmas[0], sigmas[1]
        self.sigma_r = sigmas[2]
        self.num_iterations = num_iterations
        self.device = device

        self.ct_H = None
        self.ct_V = None

    def __call__(self, tensor):
        return self.apply_filter(tensor)

    def preprocess(self, joint_image):
        I = joint_image.float().contiguous()
        B, C, H, W = I.shape

        print(f"Preprocessing joint image of shape: {I.shape}")
        I_min = I.view(I.size(0), -1).min(dim=1)[0].item()
        I_max = I.view(I.size(0), -1).max(dim=1)[0].item()

        I = (I - I_min) / (I_max - I_min + 1e-8)

        dIdx = torch.sum(torch.abs(I[:, :, :, 1:] - I[:, :, :, :-1]), dim=1)
        dIdx = F.pad(dIdx, (1, 0), mode='constant', value=0)

        dIdy = torch.sum(torch.abs(I[:, :, 1:, :] - I[:, :, :-1, :]), dim=1)
        dIdy = F.pad(dIdy, (0, 0, 1, 0), mode='constant', value=0)

        dHdx = 1 + (self.sigma_sx / self.sigma_r) * dIdx
        dVdy = 1 + (self.sigma_sy / self.sigma_r) * dIdy

        self.ct_H = torch.cumsum(dHdx, dim=2)
        self.ct_V = torch.cumsum(dVdy, dim=1).transpose(1, 2).contiguous()


    def _transformed_domain_box_filter_horizontal(self, I, xform_domain_position, box_radius):
        B, C, H, W = I.shape

        l_pos = xform_domain_position - box_radius
        u_pos = xform_domain_position + box_radius

        inf_col = torch.full((B, H, 1), float('inf'), device=I.device, dtype=I.dtype)
        xform_padded = torch.cat([xform_domain_position, inf_col], dim=2)

        l_idx = torch.searchsorted(xform_padded, l_pos, right=True).clamp(max=W)
        u_idx = torch.searchsorted(xform_padded, u_pos, right=True).clamp(max=W)

        SAT = torch.cumsum(I, dim=3)
        SAT = F.pad(SAT, (1, 0), mode='constant', value=0)

        idx_shape = (B, C, H, W)
        l_idx_exp = l_idx.unsqueeze(1).expand(idx_shape)
        u_idx_exp = u_idx.unsqueeze(1).expand(idx_shape)

        batch_idx = torch.arange(B, device=I.device)[:, None, None, None].expand(idx_shape)
        channel_idx = torch.arange(C, device=I.device)[None, :, None, None].expand(idx_shape)
        h_idx = torch.arange(H, device=I.device)[None, None, :, None].expand(idx_shape)

        F_u = SAT[batch_idx, channel_idx, h_idx, u_idx_exp]
        F_l = SAT[batch_idx, channel_idx, h_idx, l_idx_exp]

        denom = (u_idx_exp - l_idx_exp).clamp(min=1).float()
        F_out = (F_u - F_l) / denom

        return F_out

    def apply_filter(self, img):
        if self.ct_H is None or self.ct_V is None:
            raise ValueError("Run preprocess() first.")

        F_img = img.float()
        sqrt3 = np.sqrt(3)
        denom = np.sqrt(4 ** self.num_iterations - 1)

        for i in range(self.num_iterations):
            sigma_H_i_x = self.sigma_sx * sqrt3 * (2 ** (self.num_iterations - (i + 1))) / denom
            sigma_H_i_y = self.sigma_sy * sqrt3 * (2 ** (self.num_iterations - (i + 1))) / denom

            box_radius_x = sqrt3 * sigma_H_i_x
            box_radius_y = sqrt3 * sigma_H_i_y

            F_img = self._transformed_domain_box_filter_horizontal(F_img, self.ct_H, box_radius_x)
            F_img = F_img.transpose(2, 3).contiguous()

            F_img = self._transformed_domain_box_filter_horizontal(F_img, self.ct_V, box_radius_y)
            F_img = F_img.transpose(2, 3).contiguous()

        return F_img.type_as(img)

def NC(img, sigma_s, sigma_r, num_iterations=3, joint_image=None, device=None, normalize_joint=False):
    device = device or img.device
    img = img.to(device)
    joint_image = img if joint_image is None else joint_image.to(device)

    filter_instance = NCDomainTransform(
        sigmas=[*sigma_s, sigma_r],
        num_iterations=num_iterations,
        device=device,
    )
    filter_instance.preprocess(joint_image)
    return filter_instance.apply_filter(img)
