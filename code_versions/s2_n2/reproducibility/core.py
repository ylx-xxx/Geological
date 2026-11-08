"""Shared preprocessing, channel definitions, metrics, and provenance."""
from pathlib import Path
import hashlib
import json
import numpy as np

# IARAI Landslide4Sense-2022: stored bands B1..B12 (B8A omitted), slope, DEM.
CHANNELS = {'full14': list(range(14)), 'ms12': list(range(12)),
            'first3_legacy': [0, 1, 2], 'first3_topo_legacy': [0, 1, 2, 12, 13],
            'rgb': [3, 2, 1], 'rgb_topo': [3, 2, 1, 12, 13]}
NORMALIZATION = 'finite_zero_p1_p99_patch_band_zscore_v1'
# N2 只替换 slope/DEM 的 patch 标准化；光谱与裁剪/有限值处理保持 N0 完全一致。
NORMALIZATION_N2 = 'finite_zero_p1_p99_patch_band_zscore_v1_terrain_train_fixed_v1'
TERRAIN_CHANNELS = (12, 13)

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def read_h5(path, key):
    import h5py
    with h5py.File(path, 'r') as f:
        if key in f:
            return f[key][()]
        if len(f) == 1:
            return f[next(iter(f))][()]
        raise ValueError(f'Ambiguous HDF5 keys in {path}: {list(f)}')

def normalize(image):
    """Numerically matches the original final-model dataset implementation."""
    x = np.asarray(image, dtype=np.float32)
    if x.ndim != 3:
        raise ValueError(f'Expected 3D input, got {x.shape}')
    if x.shape[-1] == 14:
        x = x.transpose(2, 0, 1)
    elif x.shape[0] != 14:
        raise ValueError(f'Expected 14 native bands, got {x.shape}')
    x = np.nan_to_num(x, nan=0., posinf=0., neginf=0.)
    out = np.zeros_like(x, dtype=np.float32)
    for c, band in enumerate(x):
        band = np.clip(band, np.percentile(band, 1), np.percentile(band, 99))
        std = band.std()
        out[c] = (band - band.mean()) / (1. if std < 1e-6 else std)
    return out

def normalize_n2(image, terrain_mu, terrain_sd):
    """N2：光谱 0-11 与 N0 完全相同；坡度/DEM 用固定 Train 统计。

    与阶段 0 `stage0_q1_norm.normalize_n2` 的处理顺序逐项一致：
    float32 -> nan_to_num(0) -> 逐 patch 1%/99% 裁剪；
    地形在 float64 中用固定 (x-mu)/sd 计算后转回 float32。
    """
    x = np.asarray(image, dtype=np.float32)
    if x.ndim != 3:
        raise ValueError(f'Expected 3D input, got {x.shape}')
    if x.shape[-1] == 14:
        x = x.transpose(2, 0, 1)
    elif x.shape[0] != 14:
        raise ValueError(f'Expected 14 native bands, got {x.shape}')
    x = np.nan_to_num(x, nan=0., posinf=0., neginf=0.)
    out = np.zeros_like(x, dtype=np.float32)
    for c in range(12):
        band = x[c]
        band = np.clip(band, np.percentile(band, 1), np.percentile(band, 99))
        std = band.std()
        out[c] = (band - band.mean()) / (1. if std < 1e-6 else std)
    mu = np.asarray(terrain_mu, dtype=np.float64)
    sd = np.asarray(terrain_sd, dtype=np.float64)
    if mu.shape != (2,) or sd.shape != (2,) or not np.isfinite(mu).all() or not np.isfinite(sd).all():
        raise ValueError('terrain_mu / terrain_sd 必须为长度 2 的有限数组')
    if np.any(sd < 1e-6):
        raise ValueError('terrain_sd 过小，固定统计退化')
    for k, c in enumerate(TERRAIN_CHANNELS):
        band = x[c]
        band = np.clip(band, np.percentile(band, 1), np.percentile(band, 99))
        out[c] = ((band.astype(np.float64) - mu[k]) / sd[k]).astype(np.float32)
    return out

def load_terrain_stats(path):
    """加载阶段 0 的 Train 固定地形统计，并校验通道顺序、像素数与数值。"""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f'Terrain stats not found: {p}')
    with np.load(p) as data:
        missing = {'terrain_mu', 'terrain_sd', 'channels'} - set(data.files)
        if missing:
            raise ValueError(f'Terrain stats missing keys: {sorted(missing)}')
        mu = np.asarray(data['terrain_mu'], dtype=np.float64)
        sd = np.asarray(data['terrain_sd'], dtype=np.float64)
        channels = [int(v) for v in np.asarray(data['channels']).ravel()]
        n_pixels = [int(v) for v in np.asarray(data['n_pixels']).ravel()] if 'n_pixels' in data.files else None
        train_list_sha = str(data['train_list_sha256']) if 'train_list_sha256' in data.files else None
    if channels != list(TERRAIN_CHANNELS):
        raise ValueError(f'Terrain stats channel order must be {list(TERRAIN_CHANNELS)}, got {channels}')
    if mu.shape != (2,) or sd.shape != (2,):
        raise ValueError('terrain_mu / terrain_sd must have length 2')
    if not np.isfinite(mu).all() or not np.isfinite(sd).all():
        raise ValueError('terrain_mu / terrain_sd contain non-finite values')
    if np.any(sd < 1e-6):
        raise ValueError(f'terrain_sd too small: {sd.tolist()}')
    return {
        'path': str(p.resolve()),
        'sha256': sha256(p),
        'terrain_mu': mu,
        'terrain_sd': sd,
        'channels': channels,
        'n_pixels': n_pixels,
        'train_list_sha256': train_list_sha,
    }

def apply_normalization(image, mode='n0', terrain_stats=None):
    """统一入口：n0 为历史逐 patch 标准化；n2 为固定 Train 地形统计。"""
    if mode == 'n0':
        return normalize(image)
    if mode == 'n2':
        if terrain_stats is None:
            raise ValueError('n2 requires terrain_stats')
        return normalize_n2(image, terrain_stats['terrain_mu'], terrain_stats['terrain_sd'])
    raise ValueError(f'Unknown normalization mode: {mode}')

def confusion(prediction, target):
    p, y = np.asarray(prediction), np.asarray(target)
    if p.shape != y.shape:
        raise ValueError('Prediction and target shapes differ')
    valid = y != 255
    if not np.isin(y[valid], [0, 1]).all() or not np.isin(p[valid], [0, 1]).all():
        raise ValueError('Expected binary labels (or target ignore label 255)')
    return np.bincount((2*y[valid]+p[valid]).astype(np.int64), minlength=4).reshape(2, 2)

def metrics(cm):
    tn, fp, fn, tp = np.asarray(cm, dtype=np.int64).ravel()
    div = lambda a, b: float(a / b) if b else 0.
    iou, bg = div(tp, tp+fp+fn), div(tn, tn+fp+fn)
    return {'Landslide_IoU': iou, 'Landslide_F1': div(2*tp, 2*tp+fp+fn),
            'Landslide_Precision': div(tp, tp+fp), 'Landslide_Recall': div(tp, tp+fn),
            'mIoU': (iou+bg)/2, 'OA': div(tp+tn, tp+tn+fp+fn),
            'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'TP': int(tp)}

def pairs(root, split):
    names = {'train': ('TrainData', 'mask'), 'val': ('ValidData', 'mask'),
             'test': ('TestData', 'test')}
    folder, mask_folder = names[split]
    images = sorted((Path(root)/folder/'img').glob('image_*.h5'),
                    key=lambda p: int(p.stem.split('_')[-1]))
    result = [(p, Path(root)/folder/mask_folder/p.name.replace('image_', 'mask_')) for p in images]
    if not result or any(not m.exists() for _, m in result):
        raise ValueError(f'Missing images or masks in {folder}')
    return result

def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf8')

def verify_selection_split(split):
    if split != 'val':
        raise ValueError('Checkpoint selection requires validation data; test scores cannot select a model.')
