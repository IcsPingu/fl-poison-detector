"""Context features for label-flip detection.

These features complement raw state_dict statistics with signals that label
flipping changes more reliably: local-global deltas, final-layer class rows and
optional clean public validation behavior.
"""
from __future__ import annotations

import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

_LAYER_ALIASES: List[Tuple[str, List[str]]] = [
    ('conv1', ['conv1.0.weight', 'base.conv1.0.weight']),
    ('conv2', ['conv2.0.weight', 'base.conv2.0.weight']),
    ('fc1', ['fc1.0.weight', 'base.fc1.0.weight']),
    ('fc', ['fc.weight', 'head.weight']),
]
NUM_CLASSES = 10
VAL_PER_CLASS = int(os.environ.get('PUBLIC_VAL_PER_CLASS', '20'))
FLIP_PAIRS = tuple((i, NUM_CLASSES - 1 - i) for i in range(NUM_CLASSES // 2))


def _resolve_key(state_dict: Mapping[str, torch.Tensor], aliases: List[str]) -> str | None:
    return next((key for key in aliases if key in state_dict), None)


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().float()
    b = b.flatten().float()
    den = torch.linalg.norm(a) * torch.linalg.norm(b)
    if den.item() < 1e-12:
        return 0.0
    return float(torch.dot(a, b).div(den).item())


def _delta_stats(local: torch.Tensor, global_: torch.Tensor) -> List[float]:
    delta = (local.float() - global_.float()).flatten()
    global_flat = global_.float().flatten()
    delta_l2 = torch.linalg.norm(delta).item()
    global_l2 = torch.linalg.norm(global_flat).item()
    return [
        float(delta_l2),
        float(delta.abs().max().item()) if delta.numel() else 0.0,
        float(delta.mean().item()) if delta.numel() else 0.0,
        float(delta.std(unbiased=False).item()) if delta.numel() else 0.0,
        _cos(delta, global_flat),
        float(delta_l2 / (global_l2 + 1e-12)),
    ]


def _head_features(
    local_sd: Mapping[str, torch.Tensor],
    global_sd: Mapping[str, torch.Tensor] | None,
) -> List[float]:
    key = _resolve_key(local_sd, ['fc.weight', 'head.weight'])
    if key is None:
        return [0.0] * (NUM_CLASSES * 3 + 5 + len(FLIP_PAIRS) * 4 + 4)
    local = local_sd[key].detach().float()
    if global_sd is None or key not in global_sd or local.ndim != 2:
        return [0.0] * (NUM_CLASSES * 3 + 5 + len(FLIP_PAIRS) * 4 + 4)
    delta = local - global_sd[key].detach().float()
    rows = min(NUM_CLASSES, delta.shape[0])
    norms = torch.linalg.norm(delta[:rows].reshape(rows, -1), dim=1)
    means = delta[:rows].reshape(rows, -1).mean(dim=1)
    coss = torch.tensor([
        _cos(delta[i], global_sd[key][i]) for i in range(rows)
    ], dtype=torch.float32)

    def pad(x: torch.Tensor) -> List[float]:
        vals = x.detach().cpu().numpy().astype(np.float32).tolist()
        return vals + [0.0] * (NUM_CLASSES - len(vals))

    norm_sum = float(norms.sum().item())
    probs = (norms / (norm_sum + 1e-12)).clamp_min(1e-12)
    entropy = float(-(probs * probs.log()).sum().item())
    top_class = float(torch.argmax(norms).item()) if rows else 0.0
    summary = [
        float(norms.max().item()) if rows else 0.0,
        float(norms.min().item()) if rows else 0.0,
        float(norms.std(unbiased=False).item()) if rows else 0.0,
        top_class / max(NUM_CLASSES - 1, 1),
        entropy / math.log(NUM_CLASSES),
    ]
    pair_vals: List[float] = []
    pair_sum_vals: List[float] = []
    pair_anti_vals: List[float] = []
    pair_cos_vals: List[float] = []
    for source, target in FLIP_PAIRS:
        if source < rows and target < rows:
            pair_sum = float((norms[source] + norms[target]).item())
            pair_diff = float(abs(norms[source] - norms[target]).item())
            pair_cos = _cos(delta[source], delta[target])
            pair_anti = _cos(delta[source], -delta[target])
        else:
            pair_sum = pair_diff = pair_cos = pair_anti = 0.0
        pair_vals.extend([pair_sum, pair_diff, pair_cos, pair_anti])
        pair_sum_vals.append(pair_sum)
        pair_anti_vals.append(pair_anti)
        pair_cos_vals.append(pair_cos)
    pair_summary = [
        max(pair_sum_vals) if pair_sum_vals else 0.0,
        max(pair_anti_vals) if pair_anti_vals else 0.0,
        min(pair_cos_vals) if pair_cos_vals else 0.0,
        float(np.std(pair_sum_vals)) if pair_sum_vals else 0.0,
    ]
    return pad(norms) + pad(means) + pad(coss) + summary + pair_vals + pair_summary


class _FedAvgCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 32, 5), nn.ReLU(inplace=True), nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, 5), nn.ReLU(inplace=True), nn.MaxPool2d(2))
        self.fc1 = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(inplace=True))
        self.fc = nn.Linear(512, NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return self.fc(x)


def _canonical_model_state(state_dict: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key
        if new_key.startswith('base.'):
            new_key = new_key[len('base.'):]
        if new_key.startswith('head.'):
            new_key = 'fc.' + new_key[len('head.'):]
        out[new_key] = value.detach().cpu()
    return out


@lru_cache(maxsize=4)
def _public_validation(public_val_dir: str) -> Tuple[torch.Tensor, torch.Tensor]:
    root = Path(public_val_dir)
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    per_class = {i: 0 for i in range(NUM_CLASSES)}
    for path in sorted(root.glob('*.npz'), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem):
        with np.load(path, allow_pickle=True) as npz:
            data = npz['data'].tolist()
        x = np.asarray(data['x'], dtype=np.float32)
        y = np.asarray(data['y'], dtype=np.int64)
        keep = []
        for idx, label in enumerate(y.tolist()):
            label = int(label)
            if label in per_class and per_class[label] < VAL_PER_CLASS:
                keep.append(idx)
                per_class[label] += 1
        if keep:
            xs.append(x[keep])
            ys.append(y[keep])
        if all(v >= VAL_PER_CLASS for v in per_class.values()):
            break
    if not xs:
        raise FileNotFoundError(f'No public validation .npz files in {public_val_dir}')
    X = torch.tensor(np.concatenate(xs), dtype=torch.float32)
    y = torch.tensor(np.concatenate(ys), dtype=torch.long)
    return X, y


@torch.no_grad()
def _eval_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    public_val_dir: str,
    device: torch.device,
) -> Dict[str, np.ndarray | float]:
    X, y = _public_validation(public_val_dir)
    model = _FedAvgCNN().to(device)
    model.load_state_dict(_canonical_model_state(state_dict), strict=False)
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    losses: List[torch.Tensor] = []
    preds: List[torch.Tensor] = []
    margins: List[torch.Tensor] = []
    reverse_margins: List[torch.Tensor] = []
    reverse_prob_deltas: List[torch.Tensor] = []
    reverse_preds: List[torch.Tensor] = []
    for xb, yb in DataLoader(TensorDataset(X, y), batch_size=128, shuffle=False):
        xb = xb.to(device)
        yb = yb.to(device)
        y_flip = (NUM_CLASSES - 1 - yb).clamp_min(0)
        logits = model(xb)
        losses.append(loss_fn(logits, yb).detach().cpu())
        pred = logits.argmax(dim=1)
        preds.append((pred == yb).detach().cpu())
        true_logits = logits.gather(1, yb[:, None]).squeeze(1)
        flip_logits = logits.gather(1, y_flip[:, None]).squeeze(1)
        probs = torch.softmax(logits, dim=1)
        true_probs = probs.gather(1, yb[:, None]).squeeze(1)
        flip_probs = probs.gather(1, y_flip[:, None]).squeeze(1)
        masked = logits.clone()
        masked.scatter_(1, yb[:, None], -1e9)
        margins.append((true_logits - masked.max(dim=1).values).detach().cpu())
        reverse_margins.append((true_logits - flip_logits).detach().cpu())
        reverse_prob_deltas.append((flip_probs - true_probs).detach().cpu())
        reverse_preds.append((pred == y_flip).detach().cpu())
    loss = torch.cat(losses)
    ok = torch.cat(preds).float()
    margin = torch.cat(margins)
    reverse_margin = torch.cat(reverse_margins)
    reverse_prob_delta = torch.cat(reverse_prob_deltas)
    reverse_pred = torch.cat(reverse_preds).float()
    y_cpu = y.cpu()
    acc_by_class = np.zeros(NUM_CLASSES, dtype=np.float32)
    margin_by_class = np.zeros(NUM_CLASSES, dtype=np.float32)
    reverse_margin_by_pair = np.zeros(len(FLIP_PAIRS), dtype=np.float32)
    reverse_prob_delta_by_pair = np.zeros(len(FLIP_PAIRS), dtype=np.float32)
    reverse_pred_by_pair = np.zeros(len(FLIP_PAIRS), dtype=np.float32)
    for c in range(NUM_CLASSES):
        mask = y_cpu == c
        if mask.any():
            acc_by_class[c] = float(ok[mask].mean().item())
            margin_by_class[c] = float(margin[mask].mean().item())
    for idx, (source, target) in enumerate(FLIP_PAIRS):
        mask = (y_cpu == source) | (y_cpu == target)
        if mask.any():
            reverse_margin_by_pair[idx] = float(reverse_margin[mask].mean().item())
            reverse_prob_delta_by_pair[idx] = float(reverse_prob_delta[mask].mean().item())
            reverse_pred_by_pair[idx] = float(reverse_pred[mask].mean().item())
    return {
        'acc': float(ok.mean().item()),
        'loss': float(loss.mean().item()),
        'acc_by_class': acc_by_class,
        'margin_by_class': margin_by_class,
        'reverse_margin_by_pair': reverse_margin_by_pair,
        'reverse_prob_delta_by_pair': reverse_prob_delta_by_pair,
        'reverse_pred_by_pair': reverse_pred_by_pair,
    }


def feature_names() -> List[str]:
    names: List[str] = []
    for layer, _aliases in _LAYER_ALIASES:
        for suffix in ('delta_l2', 'delta_linf', 'delta_mean', 'delta_std', 'delta_cos_global', 'delta_rel_l2'):
            names.append(f'{layer}_{suffix}')
    for prefix in ('head_row_delta_l2', 'head_row_delta_mean', 'head_row_delta_cos'):
        names.extend([f'{prefix}_{i}' for i in range(NUM_CLASSES)])
    names.extend(['head_delta_l2_max', 'head_delta_l2_min', 'head_delta_l2_std', 'head_delta_top_class', 'head_delta_entropy'])
    for source, target in FLIP_PAIRS:
        pair = f'{source}_{target}'
        names.extend([
            f'head_flip_pair_l2_sum_{pair}',
            f'head_flip_pair_l2_diff_{pair}',
            f'head_flip_pair_cos_{pair}',
            f'head_flip_pair_anti_cos_{pair}',
        ])
    names.extend(['head_flip_pair_l2_sum_max', 'head_flip_pair_anti_cos_max', 'head_flip_pair_cos_min', 'head_flip_pair_l2_sum_std'])
    names.extend(['total_delta_l2', 'total_global_l2', 'total_rel_l2', 'total_delta_cos_global'])
    names.extend(['val_local_acc', 'val_global_acc', 'val_acc_delta', 'val_local_loss', 'val_global_loss', 'val_loss_delta'])
    names.extend([f'val_acc_delta_class_{i}' for i in range(NUM_CLASSES)])
    names.extend([f'val_margin_delta_class_{i}' for i in range(NUM_CLASSES)])
    for source, target in FLIP_PAIRS:
        pair = f'{source}_{target}'
        names.extend([
            f'val_reverse_margin_delta_pair_{pair}',
            f'val_reverse_prob_delta_pair_{pair}',
            f'val_reverse_pred_delta_pair_{pair}',
        ])
    return names


N_CONTEXT_FEATURES = len(feature_names())


def extract_context_features(
    local_sd: Mapping[str, torch.Tensor],
    global_sd: Mapping[str, torch.Tensor] | None = None,
    public_val_dir: str | None = None,
    device: torch.device | None = None,
) -> Tuple[np.ndarray, List[str]]:
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vals: List[float] = []
    total_delta: List[torch.Tensor] = []
    total_global: List[torch.Tensor] = []
    for _layer, aliases in _LAYER_ALIASES:
        key = _resolve_key(local_sd, aliases)
        if key is None or global_sd is None or key not in global_sd:
            vals.extend([0.0] * 6)
            continue
        local = local_sd[key].detach().float()
        global_ = global_sd[key].detach().float()
        vals.extend(_delta_stats(local, global_))
        total_delta.append((local - global_).flatten())
        total_global.append(global_.flatten())
    vals.extend(_head_features(local_sd, global_sd))
    if total_delta:
        d = torch.cat(total_delta)
        g = torch.cat(total_global)
        d_l2 = torch.linalg.norm(d).item()
        g_l2 = torch.linalg.norm(g).item()
        vals.extend([float(d_l2), float(g_l2), float(d_l2 / (g_l2 + 1e-12)), _cos(d, g)])
    else:
        vals.extend([0.0] * 4)

    if public_val_dir and global_sd is not None:
        local_eval = _eval_state_dict(local_sd, public_val_dir, device)
        global_eval = _eval_state_dict(global_sd, public_val_dir, device)
        acc_delta = local_eval['acc_by_class'] - global_eval['acc_by_class']
        margin_delta = local_eval['margin_by_class'] - global_eval['margin_by_class']
        reverse_margin_delta = local_eval['reverse_margin_by_pair'] - global_eval['reverse_margin_by_pair']
        reverse_prob_delta = local_eval['reverse_prob_delta_by_pair'] - global_eval['reverse_prob_delta_by_pair']
        reverse_pred_delta = local_eval['reverse_pred_by_pair'] - global_eval['reverse_pred_by_pair']
        vals.extend([
            float(local_eval['acc']),
            float(global_eval['acc']),
            float(local_eval['acc'] - global_eval['acc']),
            float(local_eval['loss']),
            float(global_eval['loss']),
            float(local_eval['loss'] - global_eval['loss']),
        ])
        vals.extend(acc_delta.astype(np.float32).tolist())
        vals.extend(margin_delta.astype(np.float32).tolist())
        for idx in range(len(FLIP_PAIRS)):
            vals.extend([
                float(reverse_margin_delta[idx]),
                float(reverse_prob_delta[idx]),
                float(reverse_pred_delta[idx]),
            ])
    else:
        vals.extend([0.0] * (26 + len(FLIP_PAIRS) * 3))
    arr = np.nan_to_num(np.asarray(vals, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if arr.shape[0] != N_CONTEXT_FEATURES:
        raise ValueError(f'Context feature size {arr.shape[0]} != {N_CONTEXT_FEATURES}')
    return arr, feature_names()


def context_tokens(features: np.ndarray, token_count: int = 128, num_bins: int = 10000) -> List[int]:
    x = np.asarray(features, dtype=np.float32)
    if x.size >= token_count:
        idx = np.linspace(0, x.size - 1, token_count).astype(np.int64)
        x = x[idx]
    else:
        x = np.pad(x, (0, token_count - x.size))
    lo, hi = np.quantile(x, [0.05, 0.95])
    if abs(float(hi - lo)) < 1e-12:
        norm = np.zeros_like(x)
    else:
        norm = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return ((norm * (num_bins - 1)).astype(np.int64) + 1).tolist()
