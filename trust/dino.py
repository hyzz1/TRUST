"""Frozen DINO features for TRUST visual affinities."""
from pathlib import Path
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
MODEL_DEFAULTS = {'dinov2': 'dinov2_vitl14', 'dinov3': 'dinov3_vitl16'}
MODEL_NAMES = {'dinov2': {'dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14', 'dinov2_vits14_reg', 'dinov2_vitb14_reg', 'dinov2_vitl14_reg', 'dinov2_vitg14_reg'}, 'dinov3': {'dinov3_vits16', 'dinov3_vits16plus', 'dinov3_vitb16', 'dinov3_vitl16', 'dinov3_vitl16plus', 'dinov3_vith16plus', 'dinov3_vit7b16'}}

class FrozenDINO(nn.Module):
    """Load an official local DINOv2 or DINOv3 backbone and checkpoint.

    No downloading, distillation, feature-channel projection or learned adapter.
    `forward` expects RGB pixels in [0,1] and the exact CLIP patch-covered extent.
    """

    def __init__(self, repo, checkpoint, family='dinov2', model_name='', device='cuda', precision='fp32', checkpoint_key=''):
        super().__init__()
        if family not in MODEL_DEFAULTS:
            raise ValueError(f'Unsupported DINO family: {family}')
        model_name = model_name or MODEL_DEFAULTS[family]
        if model_name not in MODEL_NAMES[family]:
            raise ValueError(f'Unsupported {family} backbone name: {model_name}')
        repo = Path(repo).expanduser().resolve()
        checkpoint = Path(checkpoint).expanduser().resolve()
        if not (repo / 'hubconf.py').is_file():
            raise FileNotFoundError(f'{family} local repository missing hubconf.py: {repo}')
        if not checkpoint.is_file():
            raise FileNotFoundError(f'{family} checkpoint not found: {checkpoint}')
        if precision not in ('fp32', 'fp16', 'bf16'):
            raise ValueError('DINO precision must be fp32, fp16 or bf16')
        if family == 'dinov3':
            sys.path.insert(0, str(repo))
            try:
                from dinov3.hub import backbones
                factory = getattr(backbones, model_name)
                self.encoder = factory(pretrained=False)
            finally:
                if sys.path and sys.path[0] == str(repo):
                    sys.path.pop(0)
        else:
            self.encoder = torch.hub.load(str(repo), model_name, source='local', pretrained=False)
        state = torch.load(str(checkpoint), map_location='cpu', weights_only=True)
        if checkpoint_key:
            for component in checkpoint_key.split('.'):
                state = state[component]
        elif isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
        if not isinstance(state, dict) or not state or (not all((isinstance(v, torch.Tensor) for v in state.values()))):
            raise ValueError('Expected a backbone state dict; set dino_checkpoint_key for a nested checkpoint')
        for prefix in ('module.', 'backbone.'):
            if all((k.startswith(prefix) for k in state)):
                state = {k[len(prefix):]: v for k, v in state.items()}
        try:
            self.encoder.load_state_dict(state, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(f'Checkpoint does not match {family} model {model_name}. Make sure the official repo and checkpoint belong to the same model.') from exc
        self.encoder.requires_grad_(False).eval().to(device)
        patch_size = self.encoder.patch_embed.patch_size
        self.patch_hw = (patch_size, patch_size) if isinstance(patch_size, int) else tuple(patch_size)
        self.family = family
        self.model_name = model_name
        self.precision = precision
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1))
        self.info = dict(family=family, repo=str(repo), checkpoint=str(checkpoint), model_name=model_name, precision=precision, checkpoint_key=checkpoint_key, patch_hw=list(self.patch_hw), checkpoint_bytes=checkpoint.stat().st_size)
        self.eval()

    def train(self, mode=True):
        return super().train(False)

    def _prepare(self, rgb01):
        """Apply the same resize/normalization protocol used by ``forward``."""
        if rgb01.ndim != 4 or rgb01.shape[1] != 3:
            raise ValueError('DINO input must be RGB [B,3,H,W] in [0,1]')
        h, w = rgb01.shape[-2:]
        ph, pw = self.patch_hw
        dh, dw = ((h + ph - 1) // ph * ph, (w + pw - 1) // pw * pw)
        image = rgb01.float()
        if (dh, dw) != (h, w):
            image = F.interpolate(image, size=(dh, dw), mode='bicubic', align_corners=False, antialias=True)
        image = (image - self.mean) / self.std
        if image.device.type != 'cuda' and self.precision != 'fp32':
            raise ValueError('Use fp32 for non-CUDA DINO inference')
        dtype = torch.float16 if self.precision == 'fp16' else torch.bfloat16
        return (image, (h, w), (dh, dw), dtype)

    @torch.no_grad()
    def forward(self, rgb01, target_hw):
        image, (h, w), (dh, dw), dtype = self._prepare(rgb01)
        ph, pw = self.patch_hw
        with torch.autocast(device_type=image.device.type, dtype=dtype, enabled=self.precision != 'fp32'):
            output = self.encoder.forward_features(image)
        if not isinstance(output, dict) or 'x_norm_patchtokens' not in output:
            raise RuntimeError(f'Expected official {self.family} forward_features dictionary with x_norm_patchtokens')
        tokens = output['x_norm_patchtokens'].float()
        gh, gw = (dh // ph, dw // pw)
        if tokens.shape[:2] != (image.shape[0], gh * gw):
            raise RuntimeError('DINO patch count does not match the recorded spatial grid')
        feature = tokens.transpose(1, 2).reshape(image.shape[0], -1, gh, gw)
        if (gh, gw) != tuple(target_hw):
            feature = F.interpolate(feature, size=target_hw, mode='bilinear', align_corners=False)
        feature = F.normalize(feature.flatten(2).transpose(1, 2), dim=-1, eps=1e-08)
        geometry = dict(dino_family=self.family, dino_source_hw=[h, w], dino_input_hw=[dh, dw], dino_grid_hw=[gh, gw], aligned_grid_hw=list(target_hw), alignment='same_extent_resize_then_feature_interpolation')
        return (feature, geometry)
