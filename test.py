#!/usr/bin/env python3
# test.py — DepthAnything + ZoeDepth 推理脚本（包含 zoedepth 顶级包兼容）
# 用法：把脚本放在项目根（包含 metric_depth 和 checkpoint 的目录），直接运行 `python test.py`

import sys
import importlib
from pathlib import Path
import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from torchvision.transforms import Compose
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
from PIL import Image

# -------------------------
# 1) 确保当前项目根加入 sys.path（使 metric_depth 可被导入）
# -------------------------
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 2) 如果没有顶级安装的 zoedepth 包，把本地 metric_depth.zoedepth 注册为 zoedepth
try:
    import zoedepth  # noqa: F401
except ModuleNotFoundError:
    try:
        zmod = importlib.import_module("metric_depth.zoedepth")
        sys.modules["zoedepth"] = zmod
    except Exception:
        # 不要吞掉异常——但继续，后续导入会报更明确的错误
        pass

# -------------------------
# 3) 导入模型构建工具（此处依赖上面步骤成功）
# -------------------------
from metric_depth.zoedepth.utils.config import get_config
from metric_depth.zoedepth.models.builder import build_model

# -------------------------
# 4) colormap 兼容器（matplotlib 版本差异）
# -------------------------
try:
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

# -------------------------
# 5) 参数配置（按需修改）
# -------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ckpt_path = project_root / "depth_anything_finetune_eye" / "ZoeDepthNKv1_21-Oct_15-36-0d48cd37dedf_step4000.pt"
input_dir = project_root / "test"          # 输入图片目录（会递归查找）
output_dir = project_root / "test_depth"   # 输出目录
output_dir.mkdir(parents=True, exist_ok=True)

# -------------------------
# 6) 构建并加载模型（通过 config 指定 pretrained_resource 使用本地 checkpoint）
# -------------------------
config = get_config("zoedepth_nk", "infer", dataset="nyu",
                    pretrained_resource=f"local::{ckpt_path}")
model = build_model(config).to(device).eval()

# -------------------------
# 7) 预处理变换（复用 Depth Anything 的 transform）
# -------------------------
transform = Compose([
    Resize(width=518, height=518, resize_target=False,
           keep_aspect_ratio=True, ensure_multiple_of=14,
           resize_method='lower_bound', image_interpolation_method=cv2.INTER_CUBIC),
    NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    PrepareForNet(),
])

exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}

# -------------------------
# 8) 遍历输入目录，逐文件推理并保存可视化 & 原始 .npy
# -------------------------
print(f"Device: {device}, checkpoint: {ckpt_path}")
for img_path in tqdm(sorted(input_dir.rglob("*"))):
    if not img_path.is_file() or img_path.suffix.lower() not in exts:
        continue

    bgr = cv2.imread(str(img_path))
    if bgr is None:
        print(f"跳过无法读取的文件: {img_path}")
        continue

    # 原尺寸
    h, w = bgr.shape[:2]

    # 转 RGB 且归一化到 [0,1]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) / 255.0

    # 预处理并转 batch tensor
    sample = transform({"image": rgb})
    image = torch.from_numpy(sample["image"]).unsqueeze(0).to(device)

    # 推理
    with torch.no_grad():
        out = model(image)
    # model 返回 dict 时取 "metric_depth"，否则尝试直接用输出
    pred = out["metric_depth"] if isinstance(out, dict) and "metric_depth" in out else out

    # resize 回原图大小并转 numpy（保证 shape = H x W）
    pred = torch.nn.functional.interpolate(pred, size=(h, w), mode="bilinear", align_corners=False)
    depth = pred.squeeze().cpu().numpy()

    # 输出路径（保留输入子目录结构）
    rel_dir = img_path.parent.relative_to(input_dir)
    out_subdir = output_dir / rel_dir
    out_subdir.mkdir(parents=True, exist_ok=True)
    base = img_path.stem

    # 打印原始数值范围，方便调试
    d_safe = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    # 忽略边缘区域再计算 min/max
    # --- 去除边缘和极值 ---
    # 忽略边缘区域再计算 min/max
    h, w = d_safe.shape
    crop_ratio = 0.1  # 忽略上下左右5%的边缘
    y0, y1 = int(h * crop_ratio), int(h * (1 - crop_ratio))
    x0, x1 = int(w * crop_ratio), int(w * (1 - crop_ratio))
    
    valid_center = d_safe[y0:y1, x0:x1]
    mn, mx = np.nanmin(valid_center), np.nanmax(valid_center)



    print(f"{img_path.name}: depth min={mn:.6f}, max={mx:.6f}")

    # 归一化到 [0,1]
    if mx > mn + 1e-12:
        depth_norm = (d_safe - mn) / (mx - mn)
    else:
        depth_norm = np.zeros_like(d_safe)

    # 根据视觉需要，默认不反转；如果你看到“近处是蓝色，远处是红色”可以改为 depth_norm = 1 - depth_norm
    # 若想自动判断是否反转（可选的启发式），可以开启下面的自动反转代码（注释保持默认关闭）
    # # 自动反转示例（简易启发式）：如果中间区域均较小而边缘大，可能需要反转 —— 这里保留人为控制更稳妥
    # if np.mean(depth_norm) > 0.6:
    #     depth_norm = 1 - depth_norm
    depth_norm = 1 - depth_norm
    # 伪彩色映射（兼容 matplotlib 或 OpenCV）
    if _get_cmap is not None:
        jet = _get_cmap('jet')
        vis_rgb = (jet(depth_norm)[:, :, :3] * 255).astype(np.uint8)
    else:
        tmp_u8 = (depth_norm * 255).astype(np.uint8)
        tmp_col = cv2.applyColorMap(tmp_u8, cv2.COLORMAP_JET)  # BGR
        vis_rgb = cv2.cvtColor(tmp_col, cv2.COLOR_BGR2RGB)

    # 保存彩色可视化
    Image.fromarray(vis_rgb).save(str(out_subdir / f"{base}_depth.png"))
    # 保存原始深度数组
    np.save(str(out_subdir / f"{base}_depth.npy"), depth)

print("处理完成，输出目录：", output_dir)
