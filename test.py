import os
from pathlib import Path
import cv2
import torch
import numpy as np
from tqdm import tqdm
from torchvision.transforms import Compose
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
import sys
import importlib

# Ensure local packages can be imported and provide a top-level 'zoedepth' alias
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
try:
    import zoedepth  # noqa: F401
except ModuleNotFoundError:
    try:
        zmod = importlib.import_module("metric_depth.zoedepth")
        # make the imported package available as top-level 'zoedepth'
        sys.modules["zoedepth"] = zmod
    except Exception:
        # fall back to original import error if this fails
        pass

from metric_depth.zoedepth.utils.config import get_config
from metric_depth.zoedepth.models.builder import build_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ckpt_path = Path("depth_anything_finetune_eye/ZoeDepthNKv1_21-Oct_15-36-0d48cd37dedf_step4000.pt")
input_dir = Path("test")
output_dir = Path("test_depth")
output_dir.mkdir(exist_ok=True)

config = get_config("zoedepth_nk", "infer", dataset="nyu",
                    pretrained_resource=f"local::{ckpt_path}")
model = build_model(config).to(device).eval()

transform = Compose([
    Resize(width=518, height=518, resize_target=False,
            keep_aspect_ratio=True, ensure_multiple_of=14,
            resize_method='lower_bound', image_interpolation_method=cv2.INTER_CUBIC),
    NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    PrepareForNet(),
])

exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}

# Replace the simple directory iteration with a recursive search and mirror output subfolders
for img_path in tqdm(sorted(input_dir.rglob("*"))):
    if not img_path.is_file() or img_path.suffix.lower() not in exts:
        continue
    bgr = cv2.imread(str(img_path))
    if bgr is None:
        print(f"跳过无法读取的文件: {img_path}")
        continue
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) / 255.0
    sample = transform({"image": rgb})
    image = torch.from_numpy(sample["image"]).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(image)["metric_depth"]
    pred = torch.nn.functional.interpolate(
        pred, size=bgr.shape[:2][::-1], mode="bilinear", align_corners=False)
    depth = pred.squeeze().cpu().numpy()

    # produce filename base and a simple visualization for saving as PNG
    base = img_path.stem

    # Mirror input subfolder structure under output_dir
    rel_dir = img_path.parent.relative_to(input_dir)
    out_subdir = output_dir / rel_dir
    out_subdir.mkdir(parents=True, exist_ok=True)

    # produce grayscale (black-white) visualization like eval.py
    d = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    mn, mx = float(np.min(d)), float(np.max(d))
    if mx > mn:
        norm = (d - mn) / (mx - mn)
    else:
        norm = np.zeros_like(d)
    # convert to 8-bit grayscale (0=black, 255=white)
    depth_vis = (255 * norm).astype(np.uint8)

    # save single-channel PNG (grayscale) and the raw numpy depth
    cv2.imwrite(str(out_subdir / f"{base}_depth.png"), depth_vis)
    np.save(str(out_subdir / f"{base}_depth.npy"), depth)
