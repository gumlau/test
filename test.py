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
from PIL import Image

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
    # ensure size is (height, width) to match the input image
    pred = torch.nn.functional.interpolate(
        pred, size=(bgr.shape[0], bgr.shape[1]), mode="bilinear", align_corners=False)
    depth = pred.squeeze().cpu().numpy()

    # produce filename base and a simple visualization for saving as PNG
    base = img_path.stem

    # Mirror input subfolder structure under output_dir
    rel_dir = img_path.parent.relative_to(input_dir)
    out_subdir = output_dir / rel_dir
    out_subdir.mkdir(parents=True, exist_ok=True)

    # simple min-max normalization (x - min) / (max - min)
    d = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    mn, mx = float(np.min(d)), float(np.max(d))
    if mx > mn:
        pred_image_normalized = (d - mn) / (mx - mn)
    else:
        pred_image_normalized = np.zeros_like(d)

    # get a compatible colormap getter across matplotlib versions; fall back to None
    try:
        # matplotlib 3.8+ has `colormaps`
        from matplotlib import colormaps
        def _get_cmap(name):
            return colormaps.get_cmap(name)
    except Exception:
        try:
            import matplotlib.cm as _cm
            def _get_cmap(name):
                return _cm.get_cmap(name)
        except Exception:
            _get_cmap = None

    if _get_cmap is not None:
        jet_cmap = _get_cmap('jet')
        pred_image_rgb = jet_cmap(pred_image_normalized)[:, :, :3]  # take RGB, drop alpha
        pred_image_rgb = (pred_image_rgb * 255).astype(np.uint8)
    else:
        # fallback: use OpenCV's colormap (returns BGR), convert to RGB
        tmp = (pred_image_normalized * 255).astype(np.uint8)
        tmp_col = cv2.applyColorMap(tmp, cv2.COLORMAP_JET)  # BGR
        pred_image_rgb = cv2.cvtColor(tmp_col, cv2.COLOR_BGR2RGB)

    pil_image = Image.fromarray(pred_image_rgb)
    pil_image.save(str(out_subdir / f"{base}_depth.png"))
    # keep raw numpy depth
    np.save(str(out_subdir / f"{base}_depth.npy"), depth)
    # closing kernel scales with image size (small fraction)
    k = max(3, int(min(h, w) * 0.03))
    if k % 2 == 0:
        k += 1
    try:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        tmp_u8 = cv2.morphologyEx(tmp_u8, cv2.MORPH_CLOSE, kernel)
    except Exception:
        pass
    if min(h, w) >= 5:
        tmp_u8 = cv2.medianBlur(tmp_u8, 5)
    pred_image_normalized = (tmp_u8.astype(np.float32) / 255.0)

    # get a compatible colormap getter across matplotlib versions; fall back to None
    try:
        # matplotlib 3.8+ has `colormaps`
        from matplotlib import colormaps
        def _get_cmap(name):
            return colormaps.get_cmap(name)
    except Exception:
        try:
            import matplotlib.cm as _cm
            def _get_cmap(name):
                return _cm.get_cmap(name)
        except Exception:
            _get_cmap = None

    if _get_cmap is not None:
        jet_cmap = _get_cmap('jet')
        pred_image_rgb = jet_cmap(pred_image_normalized)[:, :, :3]  # take RGB, drop alpha
        pred_image_rgb = (pred_image_rgb * 255).astype(np.uint8)
    else:
        # fallback: use OpenCV's colormap (returns BGR), convert to RGB
        tmp = (pred_image_normalized * 255).astype(np.uint8)
        tmp_col = cv2.applyColorMap(tmp, cv2.COLORMAP_JET)  # BGR
        pred_image_rgb = cv2.cvtColor(tmp_col, cv2.COLOR_BGR2RGB)

    pil_image = Image.fromarray(pred_image_rgb)
    pil_image.save(str(out_subdir / f"{base}_depth.png"))
    # keep raw numpy depth
    np.save(str(out_subdir / f"{base}_depth.npy"), depth)
