# MIT License

# Copyright (c) 2022 Intelligent Systems Lab Org

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# File author: Shariq Farooq Bhat

import os
from pathlib import Path
from typing import Iterable

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROJECT_PARENT = PROJECT_ROOT.parent


def _deduplicate(seq: Iterable[Path]) -> Iterable[Path]:
    seen = set()
    for item in seq:
        if item in seen:
            continue
        seen.add(item)
        yield item

def load_state_dict(model, state_dict):
    """Load state_dict into model, handling DataParallel and DistributedDataParallel. Also checks for "model" key in state_dict.

    DataParallel prefixes state_dict keys with 'module.' when saving.
    If the model is not a DataParallel model but the state_dict is, then prefixes are removed.
    If the model is a DataParallel model but the state_dict is not, then prefixes are added.
    """
    state_dict = state_dict.get('model', state_dict)
    # if model is a DataParallel model, then state_dict keys are prefixed with 'module.'

    do_prefix = isinstance(
        model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel))
    state = {}
    for k, v in state_dict.items():
        if k.startswith('module.') and not do_prefix:
            k = k[7:]

        if not k.startswith('module.') and do_prefix:
            k = 'module.' + k

        state[k] = v

    model.load_state_dict(state)
    print("Loaded successfully")
    return model


def load_wts(model, checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    return load_state_dict(model, ckpt)


def load_state_dict_from_url(model, url, **kwargs):
    state_dict = torch.hub.load_state_dict_from_url(url, map_location='cpu', **kwargs)
    return load_state_dict(model, state_dict)


def load_state_from_resource(model, resource: str):
    """Loads weights to the model from a given resource. A resource can be of following types:
        1. URL. Prefixed with "url::"
                e.g. url::http(s)://url.resource.com/ckpt.pt

        2. Local path. Prefixed with "local::"
                e.g. local::/path/to/ckpt.pt


    Args:
        model (torch.nn.Module): Model
        resource (str): resource string

    Returns:
        torch.nn.Module: Model with loaded weights
    """
    print(f"Using pretrained resource {resource}")

    if resource.startswith('url::'):
        url = resource.split('url::')[1]
        return load_state_dict_from_url(model, url, progress=True)

    elif resource.startswith('local::'):
        path = resource.split('local::', 1)[1].strip()
        path_obj = Path(path) if path else None

        primary_candidates = []
        if path_obj is not None and path:
            primary_candidates.append(path_obj)
            if not path_obj.is_absolute():
                primary_candidates.extend([
                    Path.cwd() / path_obj,
                    PROJECT_ROOT / path_obj,
                    PROJECT_ROOT / path_obj.name,
                    PROJECT_PARENT / path_obj,
                    PROJECT_PARENT / path_obj.name,
                ])

        fallback_names = []
        if path_obj is not None:
            fallback_names.append(path_obj.name)
        fallback_names.extend([
            "ZoeDepthNKv1_21-Oct_15-36-0d48cd37dedf_step4000.pt",
            "depth_anything_vitl14.pth",
        ])

        fallback_dirs = [Path.cwd(), PROJECT_ROOT, PROJECT_PARENT]
        for name in fallback_names:
            if not name:
                continue
            for directory in fallback_dirs:
                fallback_candidate = directory / name
                primary_candidates.append(fallback_candidate)

        errors = []
        for candidate in _deduplicate(primary_candidates):
            if not candidate.exists():
                continue
            try:
                return load_wts(model, str(candidate))
            except RuntimeError as exc:
                errors.append((candidate, exc))
                continue

        if errors:
            formatted = "\n".join(
                f" - {cand}: {err}" for cand, err in errors
            )
            raise RuntimeError(
                "All discovered checkpoints failed to load due to state dict mismatches:\n"
                f"{formatted}"
            ) from errors[-1][1]

        raise FileNotFoundError(f"Pretrained resource not found: {path or '[auto-detect]'}")
        
    else:
        raise ValueError("Invalid resource type, only url:: and local:: are supported")
    
