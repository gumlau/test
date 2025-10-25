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

import argparse
import os
from pprint import pprint

import torch
from torch.nn import functional as F
from zoedepth.utils.easydict import EasyDict as edict
from tqdm import tqdm

from PIL import Image
import numpy as np

from zoedepth.data.data_mono import DepthDataLoader, remove_leading_slash
from zoedepth.models.builder import build_model
from zoedepth.utils.arg_utils import parse_unknown
from zoedepth.utils.config import change_dataset, get_config, ALL_EVAL_DATASETS, ALL_INDOOR, ALL_OUTDOOR
from zoedepth.utils.misc import (RunningAverageDict, colors, compute_metrics,
                        count_parameters)


def get_instrument_mask(sample, config, device, target_shape):
    root = getattr(config, "instrument_mask_path_eval",
                   getattr(config, "instrument_mask_path", None))
    subdir = getattr(config, "instrument_mask_subdir", "instrument_mask")
    rel_path = sample.get('image_path', None)
    if root is None or rel_path is None:
        return None

    if isinstance(rel_path, (list, tuple)):
        if len(rel_path) == 0:
            return None
        rel_path = rel_path[0]

    rel_path = remove_leading_slash(rel_path)
    parts = [p for p in rel_path.split('/') if p]
    if 'imgs' not in parts:
        return None

    idx = parts.index('imgs')
    parts[idx] = subdir
    mask_rel = os.path.join(*parts)
    mask_path = os.path.join(root, mask_rel)
    if not os.path.exists(mask_path):
        return None

    mask_img = Image.open(mask_path).convert('L')
    mask_arr = np.asarray(mask_img, dtype=np.uint8) > 127
    if not np.any(mask_arr):
        return None

    mask_tensor = torch.from_numpy(mask_arr.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    if mask_tensor.shape[-2:] != target_shape:
        mask_tensor = F.interpolate(mask_tensor, size=target_shape, mode='nearest')
    return mask_tensor.to(torch.bool)


def prepare_eval_config(config, dataset):
    """Apply sensible defaults for zero-shot evaluation on custom datasets."""
    config.mode = "eval"
    config.distributed = False
    config.aug = False
    config.random_crop = False
    config.random_translate = False
    config.batch_size = 1
    config.bs = 1
    config.workers = 1
    config.num_workers = 1
    config.shuffle_test = False
    config.dataset = dataset
    if 'use_pretrained_midas' not in config:
        config.use_pretrained_midas = True
    else:
        config.use_pretrained_midas = bool(config.use_pretrained_midas)
    return config


@torch.no_grad()
def infer(model, images, **kwargs):
    """Inference with flip augmentation"""
    # images.shape = N, C, H, W
    def get_depth_from_prediction(pred):
        if isinstance(pred, torch.Tensor):
            pred = pred  # pass
        elif isinstance(pred, (list, tuple)):
            pred = pred[-1]
        elif isinstance(pred, dict):
            pred = pred['metric_depth'] if 'metric_depth' in pred else pred['out']
        else:
            raise NotImplementedError(f"Unknown output type {type(pred)}")
        return pred

    pred1 = model(images, **kwargs)
    pred1 = get_depth_from_prediction(pred1)

    pred2 = model(torch.flip(images, [3]), **kwargs)
    pred2 = get_depth_from_prediction(pred2)
    pred2 = torch.flip(pred2, [3])

    mean_pred = 0.5 * (pred1 + pred2)

    return mean_pred


@torch.no_grad()
def evaluate(model, test_loader, config, round_vals=True, round_precision=3, device=None):
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    metrics = RunningAverageDict()
    instrument_metrics = RunningAverageDict()
    for i, sample in tqdm(enumerate(test_loader), total=len(test_loader)):
        if 'has_valid_depth' in sample:
            if not sample['has_valid_depth']:
                continue
        image = sample['image'].to(device)
        depth = sample['depth']
        if isinstance(depth, torch.Tensor):
            depth = depth.to(device)
        else:
            depth = torch.as_tensor(depth, device=device)
        depth = depth.squeeze().unsqueeze(0).unsqueeze(0)
        focal_val = sample.get('focal', None)
        if isinstance(focal_val, torch.Tensor):
            focal = focal_val.to(device)
        elif focal_val is None:
            focal = torch.tensor([715.0873], device=device)
        else:
            focal = torch.as_tensor([focal_val], device=device, dtype=torch.float32)

        dataset_label = sample['dataset']
        if isinstance(dataset_label, (list, tuple)):
            dataset_label = dataset_label[0]

        pred = infer(model, image, dataset=dataset_label, focal=focal)

        # Save image, depth, pred for visualization
        if "save_images" in config and config.save_images:
            import os
            # print("Saving images ...")
            from PIL import Image
            import torchvision.transforms as transforms
            from zoedepth.utils.misc import colorize

            os.makedirs(config.save_images, exist_ok=True)
            # def save_image(img, path):
            vmin = getattr(config, "depth_viz_min", None)
            vmax = getattr(config, "depth_viz_max", None)
            if vmin is None:
                vmin = getattr(config, "min_depth_eval", None)
            if vmax is None:
                vmax = getattr(config, "max_depth_eval", None)

            d = colorize(depth.squeeze().cpu().numpy(), vmin, vmax, cmap='gray')
            p = colorize(pred.squeeze().cpu().numpy(), vmin, vmax, cmap='gray')
            im = transforms.ToPILImage()(image.squeeze().cpu())
            # ensure depth/pred visualizations match the input image resolution
            target_size = im.size  # (width, height)
            depth_vis = Image.fromarray(d).resize(target_size, resample=Image.BILINEAR)
            pred_vis = Image.fromarray(p).resize(target_size, resample=Image.BILINEAR)

            im.save(os.path.join(config.save_images, f"{i}_img.png"))
            depth_vis.save(os.path.join(config.save_images, f"{i}_depth.png"))
            pred_vis.save(os.path.join(config.save_images, f"{i}_pred.png"))



        # print(depth.shape, pred.shape)
        metrics.update(compute_metrics(depth, pred, mask=sample.get('mask', None), config=config))

        inst_mask = get_instrument_mask(sample, config, depth.device, depth.shape[-2:])
        if inst_mask is not None:
            inst_pixels = inst_mask.sum().item()
            depth_valid = torch.logical_and(
                depth > getattr(config, "min_depth_eval", 0.0),
                depth < getattr(config, "max_depth_eval", float('inf')),
            )
            valid_inst = torch.logical_and(depth_valid, inst_mask)
            valid_inst_pixels = valid_inst.sum().item()

            if valid_inst_pixels == 0:
                img_label = sample.get('image_path', '')
                if isinstance(img_label, (list, tuple)) and img_label:
                    img_label = img_label[0]
                sample_mask = sample.get('mask', None)
                mask_pixels = sample_mask.sum().item() if isinstance(sample_mask, torch.Tensor) else None
                print(
                    f"[Instrument metrics warning] sample={i} path={img_label} "
                    f"instrument_pixels={inst_pixels} valid_instrument_pixels=0 "
                    f"mask_pixels={mask_pixels}"
                )
                continue

            instrument_metrics.update(compute_metrics(depth, pred, mask=inst_mask, config=config))

    if round_vals:
        def r(m): return round(m, round_precision)
    else:
        def r(m): return m
    metrics = {k: r(v) for k, v in metrics.get_value().items()}
    inst_values = instrument_metrics.get_value()
    if inst_values:
        inst_values = {k: r(v) for k, v in inst_values.items()}
    else:
        inst_values = None
    return metrics, inst_values

def main(config):
    model = build_model(config)
    test_loader = DepthDataLoader(config, 'online_eval').data
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda') if use_cuda else torch.device('cpu')
    if not use_cuda:
        print("CUDA not available; running evaluation on CPU.")
    model = model.to(device)
    overall_metrics, instrument_metrics = evaluate(model, test_loader, config, device=device)
    print(f"{colors.fg.green}")
    if instrument_metrics:
        print({"overall": overall_metrics, "instrument": instrument_metrics})
    else:
        print(overall_metrics)
    print(f"{colors.reset}")

    combined = dict(overall_metrics)
    if instrument_metrics:
        combined.update({f"instrument/{k}": v for k, v in instrument_metrics.items()})
    combined['#params'] = f"{round(count_parameters(model, include_all=True)/1e6, 2)}M"
    return combined


def eval_model(model_name, pretrained_resource, dataset='nyu', **kwargs):

    # Load default pretrained resource defined in config if not set
    overwrite = {**kwargs, "pretrained_resource": pretrained_resource} if pretrained_resource else kwargs
    config = get_config(model_name, "eval", dataset, **overwrite)
    config = prepare_eval_config(config, dataset)
    if pretrained_resource:
        config.pretrained_resource = pretrained_resource
    # config = change_dataset(config, dataset)  # change the dataset
    pprint(config)
    print(f"Evaluating {model_name} on {dataset}...")
    metrics = main(config)
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str,
                        required=True, help="Name of the model to evaluate")
    parser.add_argument("-p", "--pretrained_resource", type=str,
                        required=False, default="", help="Pretrained resource to use for fetching weights. If not set, default resource from model config is used,  Refer models.model_io.load_state_from_resource for more details.")
    parser.add_argument("-d", "--dataset", type=str, required=False,
                        default='nyu', help="Dataset to evaluate on")

    args, unknown_args = parser.parse_known_args()
    overwrite_kwargs = parse_unknown(unknown_args)

    if "ALL_INDOOR" in args.dataset:
        datasets = ALL_INDOOR
    elif "ALL_OUTDOOR" in args.dataset:
        datasets = ALL_OUTDOOR
    elif "ALL" in args.dataset:
        datasets = ALL_EVAL_DATASETS
    elif "," in args.dataset:
        datasets = args.dataset.split(",")
    else:
        datasets = [args.dataset]
    
    for dataset in datasets:
        eval_model(args.model, pretrained_resource=args.pretrained_resource,
                    dataset=dataset, **overwrite_kwargs)
