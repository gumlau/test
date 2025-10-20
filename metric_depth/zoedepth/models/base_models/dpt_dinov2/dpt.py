import os
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import FeatureFusionBlock, _make_scratch


def _make_fusion_block(features, use_bn, size = None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


class DPTHead(nn.Module):
    def __init__(self, in_channels, features=256, use_bn=False, out_channels=[256, 512, 1024, 1024], use_clstoken=False):
        super(DPTHead, self).__init__()
        
        self.use_clstoken = use_clstoken
        
        # out_channels = [in_channels // 8, in_channels // 4, in_channels // 2, in_channels]
        # out_channels = [in_channels // 4, in_channels // 2, in_channels, in_channels]
        # out_channels = [in_channels, in_channels, in_channels, in_channels]
        
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])
        
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])
        
        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(
                        nn.Linear(2 * in_channels, in_channels),
                        nn.GELU()))
        
        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )

        self.scratch.stem_transpose = None
        
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        head_features_1 = features
        head_features_2 = 32
        
        self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)
        
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            nn.Identity(),
        )
    
    def forward(self, out_features, patch_h, patch_w):
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]
            
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            
            out.append(x)
        
        layer_1, layer_2, layer_3, layer_4 = out
        
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        
        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        
        out = self.scratch.output_conv1(path_1)
        out = F.interpolate(out, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True)
        out = self.scratch.output_conv2(out)
            
        return out


def _resolve_dinov2_repo() -> Path:
    """Discover the local DINOv2 hub repo location if present."""
    override = os.environ.get("ZOE_DEPTH_DINOV2_REPO")
    if override:
        repo_path = Path(override).expanduser().resolve()
        if repo_path.exists():
            return repo_path
        warnings.warn(f"ZOE_DEPTH_DINOV2_REPO={repo_path} not found; falling back to defaults.")

    root = Path(__file__).resolve().parents[5]
    candidate = root / "torchhub" / "facebookresearch_dinov2_main"
    if candidate.exists():
        return candidate
    return candidate  # Return default location even if missing for error message


def _load_dinov2_encoder(encoder: str):
    repo = _resolve_dinov2_repo()
    model_name = f"dinov2_{encoder}14"
    if repo.exists():
        return torch.hub.load(str(repo), model_name, source="local", pretrained=False)

    warnings.warn(
        f"Local DINOv2 repo not found at {repo}. Attempting to download '{model_name}' from torch.hub. "
        "Set ZOE_DEPTH_DINOV2_REPO or clone facebookresearch/dinov2 into ../torchhub to avoid this step."
    )
    try:
        return torch.hub.load("facebookresearch/dinov2", model_name, pretrained=True)
    except Exception as exc:  # pragma: no cover - network dependent
        raise FileNotFoundError(
            f"Unable to load {model_name}. Clone facebookresearch/dinov2 into {repo} or set ZOE_DEPTH_DINOV2_REPO." 
        ) from exc


class DPT_DINOv2(nn.Module):
    def __init__(self, encoder='vitl', features=256, use_bn=False, out_channels=[256, 512, 1024, 1024], use_clstoken=False):

        super(DPT_DINOv2, self).__init__()

        torch.manual_seed(1)
        
        self.pretrained = _load_dinov2_encoder(encoder)
        
        dim = self.pretrained.blocks[0].attn.qkv.in_features
        
        self.depth_head = DPTHead(dim, features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken)

    def forward(self, x):
        h, w = x.shape[-2:]

        features = self.pretrained.get_intermediate_layers(x, 4, return_class_token=True)
        
        patch_h, patch_w = h // 14, w // 14

        depth = self.depth_head(features, patch_h, patch_w)
        depth = F.interpolate(depth, size=(h, w), mode="bilinear", align_corners=True)
        depth = F.relu(depth)

        return depth.squeeze(1)
