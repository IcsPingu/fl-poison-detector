"""Treino e avaliacao do detector MLP de updates maliciosos em FL.

Pipeline:
  state_dicts/*.safetensors  ->  features.extract_features (60 dims)
  -> StandardScaler          ->  MLPDetector (60->128->64->2)
  -> early stopping em F1    ->  artefatos em detector_mlp_artifacts/

Saida inclui breakdown de recall por tipo de ataque (benign + 4 maliciosos).
"""
from __future__ import annotations

import copy
import glob
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import load_file
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

try:
    from features import N_FEATURES, extract_features, feature_names
    from context_features import (
        N_CONTEXT_FEATURES,
        extract_context_features,
        feature_names as context_feature_names,
    )
    from split_utils import split_summary, split_train_dev_calib_test_by_client, write_score_diagnostics
except ImportError:
    from .features import N_FEATURES, extract_features, feature_names
    from .context_features import (
        N_CONTEXT_FEATURES,
        extract_context_features,
        feature_names as context_feature_names,
    )
    from .split_utils import split_summary, split_train_dev_calib_test_by_client, write_score_diagnostics

SEED = 42
STATE_DICTS_DIR = os.environ.get('STATE_DICTS_DIR', 'state_dicts')
ARTIFACTS_DIR = Path(os.environ.get('ARTIFACTS_DIR', 'detector_mlp_artifacts'))
OVERSAMPLE_LABEL_FACTOR = max(1, int(os.environ.get('OVERSAMPLE_LABEL_FACTOR', '1')))
LABEL_LOSS_WEIGHT = float(os.environ.get('LABEL_LOSS_WEIGHT', '1.0'))
HIDDEN = (128, 64)
DROPOUT = 0.3
LR = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 60
BATCH_SIZE = 32
PATIENCE = 15
TEST_SIZE = 0.2
CALIB_SIZE = 0.2
DEV_SIZE = 0.2
PUBLIC_VAL_DIR = os.environ.get('PUBLIC_VAL_DIR', '')
TOTAL_FEATURES = N_FEATURES + N_CONTEXT_FEATURES


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MLPDetector(nn.Module):
    def __init__(self, input_dim: int = TOTAL_FEATURES, hidden=(128, 64), dropout: float = 0.3):
        super().__init__()
        h1, h2 = hidden
        self.input_dim = input_dim
        self.hidden = list(hidden)
        self.dropout = dropout
        self.trunk = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, h1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(h2, 2)
        self.label_classifier = nn.Linear(h2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.trunk(x))

    def label_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.label_classifier(self.trunk(x)).squeeze(-1)


def _load_global_state(meta: Dict, state_dir: str) -> Dict[str, torch.Tensor] | None:
    global_ref = meta.get('global_state')
    if not global_ref:
        return None
    path = Path(global_ref)
    if not path.is_absolute():
        path = Path(state_dir) / path
    if not path.exists():
        return None
    return load_file(str(path))


def load_dataset() -> Tuple[np.ndarray, np.ndarray, List[str], List[Dict]]:
    files = [
        f for f in sorted(glob.glob(os.path.join(STATE_DICTS_DIR, '*.safetensors')))
        if os.path.exists(f.replace('.safetensors', '.json'))
    ]
    assert files, f"Nenhum .safetensors em '{STATE_DICTS_DIR}/'."

    X_rows: List[np.ndarray] = []
    y_list: List[int] = []
    types: List[str] = []
    entries: List[Dict] = []
    for f in tqdm(files, desc='extract features', unit='file'):
        sd = load_file(f)
        with open(f.replace('.safetensors', '.json')) as jf:
            meta = json.load(jf)
        global_sd = _load_global_state(meta, STATE_DICTS_DIR)
        base_feats, _ = extract_features(sd)
        ctx_feats, _ = extract_context_features(
            sd,
            global_sd=global_sd,
            public_val_dir=PUBLIC_VAL_DIR or None,
        )
        X_rows.append(np.concatenate([base_feats, ctx_feats]).astype(np.float32))
        y_list.append(int(meta['label']))
        type_ = meta.get('type', 'unknown')
        types.append(type_)
        entries.append({
            'sample_id': os.path.splitext(os.path.basename(f))[0],
            'label': int(meta['label']),
            'labels': int(meta['label']),
            'type': type_,
            'round': int(meta.get('round', 0)),
            'client_id': int(meta.get('client_id', -1)),
        })

    X = np.stack(X_rows).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    return X, y, types, entries


def evaluate(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
    model.eval()
    with torch.no_grad():
        logits = model(X)
        preds = logits.argmax(dim=-1).cpu().numpy()
    y_np = y.cpu().numpy()
    return {
        'accuracy': accuracy_score(y_np, preds),
        'precision': precision_score(y_np, preds, zero_division=0),
        'recall': recall_score(y_np, preds, zero_division=0),
        'f1': f1_score(y_np, preds, zero_division=0),
        'preds': preds.tolist(),
    }


def predict_logits(model: nn.Module, X: torch.Tensor) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(X).detach().cpu().numpy()


def predict_outputs(model: MLPDetector, X: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        logits = model(X)
        label_scores = model.label_logits(X)
    return logits.detach().cpu().numpy(), label_scores.detach().cpu().numpy()


def _threshold_candidates(scores: np.ndarray, n_grid: int = 400) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    eps = max(float(np.ptp(scores)) * 1e-6, 1e-6)
    return np.linspace(float(scores.min()) - eps, float(scores.max()) + eps, n_grid)


def metrics_from_threshold(logits: np.ndarray, labels: np.ndarray, threshold: float) -> Dict:
    preds = ((logits[:, 1] - logits[:, 0]) > threshold).astype(np.int64)
    return {
        'accuracy': float(accuracy_score(labels, preds)),
        'precision': float(precision_score(labels, preds, zero_division=0)),
        'recall': float(recall_score(labels, preds, zero_division=0)),
        'f1': float(f1_score(labels, preds, zero_division=0)),
        'preds': preds.tolist(),
    }


def tune_threshold(logits: np.ndarray, labels: np.ndarray) -> Dict[str, float | List[int]]:
    scores = logits[:, 1] - logits[:, 0]
    eps = max(float(np.ptp(scores)) * 1e-6, 1e-6)
    thresholds = np.linspace(float(scores.min()) - eps, float(scores.max()) + eps, 200)
    best = None
    for threshold in thresholds:
        preds = (scores > threshold).astype(np.int64)
        f1 = f1_score(labels, preds, zero_division=0)
        item = {
            'threshold': float(threshold),
            'accuracy': float(accuracy_score(labels, preds)),
            'precision': float(precision_score(labels, preds, zero_division=0)),
            'recall': float(recall_score(labels, preds, zero_division=0)),
            'f1': float(f1),
            'preds': preds.tolist(),
        }
        if best is None or item['f1'] > best['f1']:
            best = item
    assert best is not None
    return best


def tune_threshold_with_constraint(
    logits: np.ndarray,
    labels: np.ndarray,
    types: List[str],
    max_benign_fpr: float = 0.05,
    objective: str = 'malicious_recall',
) -> Dict[str, float | List[int]]:
    scores = logits[:, 1] - logits[:, 0]
    eps = max(float(np.ptp(scores)) * 1e-6, 1e-6)
    thresholds = np.linspace(float(scores.min()) - eps, float(scores.max()) + eps, 400)
    labels_np = np.asarray(labels)
    types_np = np.asarray(types)
    benign_mask = labels_np == 0
    label_mask = types_np == 'malicious_label'
    best = None

    for threshold in thresholds:
        preds = (scores > threshold).astype(np.int64)
        benign_fpr = float(preds[benign_mask].mean()) if benign_mask.any() else 0.0
        if benign_fpr > max_benign_fpr:
            continue
        malicious_recall = float(recall_score(labels_np, preds, zero_division=0))
        label_recall = float(preds[label_mask].mean()) if label_mask.any() else 0.0
        f1 = float(f1_score(labels_np, preds, zero_division=0))
        precision = float(precision_score(labels_np, preds, zero_division=0))
        key = (
            label_recall if objective == 'label_recall' else malicious_recall,
            malicious_recall,
            f1,
            -benign_fpr,
        )
        item = {
            'threshold': float(threshold),
            'accuracy': float(accuracy_score(labels_np, preds)),
            'precision': precision,
            'recall': malicious_recall,
            'f1': f1,
            'benign_fpr': benign_fpr,
            'malicious_label_recall': label_recall,
            'preds': preds.tolist(),
            '_key': key,
        }
        if best is None or item['_key'] > best['_key']:
            best = item

    if best is None:
        threshold = float(scores.max() + 1e-6)
        preds = (scores > threshold).astype(np.int64)
        best = {
            'threshold': threshold,
            'accuracy': float(accuracy_score(labels_np, preds)),
            'precision': float(precision_score(labels_np, preds, zero_division=0)),
            'recall': float(recall_score(labels_np, preds, zero_division=0)),
            'f1': float(f1_score(labels_np, preds, zero_division=0)),
            'benign_fpr': 0.0,
            'malicious_label_recall': 0.0,
            'preds': preds.tolist(),
            '_key': (0.0, 0.0, 0.0, 0.0),
        }
    best.pop('_key', None)
    return best


def tune_score_threshold_with_constraint(
    scores: np.ndarray,
    labels: np.ndarray,
    types: List[str],
    max_benign_fpr: float = 0.05,
    objective: str = 'label_recall',
) -> Dict[str, float | List[int]]:
    labels_np = np.asarray(labels)
    types_np = np.asarray(types)
    benign_mask = labels_np == 0
    label_mask = types_np == 'malicious_label'
    best = None
    for threshold in _threshold_candidates(np.asarray(scores), n_grid=400):
        preds = (scores > threshold).astype(np.int64)
        benign_fpr = float(preds[benign_mask].mean()) if benign_mask.any() else 0.0
        if benign_fpr > max_benign_fpr:
            continue
        malicious_recall = float(recall_score(labels_np, preds, zero_division=0))
        label_recall = float(preds[label_mask].mean()) if label_mask.any() else 0.0
        f1 = float(f1_score(labels_np, preds, zero_division=0))
        key = (
            label_recall if objective == 'label_recall' else malicious_recall,
            malicious_recall,
            f1,
            -benign_fpr,
        )
        item = {
            'threshold': float(threshold),
            'accuracy': float(accuracy_score(labels_np, preds)),
            'precision': float(precision_score(labels_np, preds, zero_division=0)),
            'recall': malicious_recall,
            'f1': f1,
            'benign_fpr': benign_fpr,
            'malicious_label_recall': label_recall,
            'preds': preds.tolist(),
            '_key': key,
        }
        if best is None or item['_key'] > best['_key']:
            best = item
    if best is None:
        threshold = float(np.max(scores) + 1e-6)
        preds = (scores > threshold).astype(np.int64)
        best = {
            'threshold': threshold,
            'accuracy': float(accuracy_score(labels_np, preds)),
            'precision': float(precision_score(labels_np, preds, zero_division=0)),
            'recall': float(recall_score(labels_np, preds, zero_division=0)),
            'f1': float(f1_score(labels_np, preds, zero_division=0)),
            'benign_fpr': 0.0,
            'malicious_label_recall': 0.0,
            'preds': preds.tolist(),
            '_key': (0.0, 0.0, 0.0, 0.0),
        }
    best.pop('_key', None)
    return best


def tune_combined_thresholds(
    binary_scores: np.ndarray,
    label_scores: np.ndarray,
    labels: np.ndarray,
    types: List[str],
    max_benign_fpr: float = 0.05,
) -> Dict[str, float | List[int]]:
    labels_np = np.asarray(labels)
    types_np = np.asarray(types)
    benign_mask = labels_np == 0
    label_mask = types_np == 'malicious_label'
    best = None
    binary_candidates = _threshold_candidates(binary_scores, n_grid=80)
    label_candidates = _threshold_candidates(label_scores, n_grid=80)
    for binary_threshold in binary_candidates:
        binary_hit = binary_scores > binary_threshold
        for label_threshold in label_candidates:
            preds = np.logical_or(binary_hit, label_scores > label_threshold).astype(np.int64)
            benign_fpr = float(preds[benign_mask].mean()) if benign_mask.any() else 0.0
            if benign_fpr > max_benign_fpr:
                continue
            malicious_recall = float(recall_score(labels_np, preds, zero_division=0))
            label_recall = float(preds[label_mask].mean()) if label_mask.any() else 0.0
            f1 = float(f1_score(labels_np, preds, zero_division=0))
            key = (label_recall, malicious_recall, f1, -benign_fpr)
            item = {
                'binary_threshold': float(binary_threshold),
                'label_threshold': float(label_threshold),
                'accuracy': float(accuracy_score(labels_np, preds)),
                'precision': float(precision_score(labels_np, preds, zero_division=0)),
                'recall': malicious_recall,
                'f1': f1,
                'benign_fpr': benign_fpr,
                'malicious_label_recall': label_recall,
                'preds': preds.tolist(),
                '_key': key,
            }
            if best is None or item['_key'] > best['_key']:
                best = item
    if best is None:
        best = {
            'binary_threshold': float(np.max(binary_scores) + 1e-6),
            'label_threshold': float(np.max(label_scores) + 1e-6),
            'accuracy': float(accuracy_score(labels_np, np.zeros_like(labels_np))),
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'benign_fpr': 0.0,
            'malicious_label_recall': 0.0,
            'preds': np.zeros_like(labels_np).tolist(),
            '_key': (0.0, 0.0, 0.0, 0.0),
        }
    best.pop('_key', None)
    return best


def combined_metrics_from_thresholds(
    binary_scores: np.ndarray,
    label_scores: np.ndarray,
    labels: np.ndarray,
    types: List[str],
    binary_threshold: float,
    label_threshold: float,
) -> Dict:
    labels_np = np.asarray(labels)
    types_np = np.asarray(types)
    preds = np.logical_or(binary_scores > binary_threshold, label_scores > label_threshold).astype(np.int64)
    label_mask = types_np == 'malicious_label'
    benign_mask = labels_np == 0
    return {
        'accuracy': float(accuracy_score(labels_np, preds)),
        'precision': float(precision_score(labels_np, preds, zero_division=0)),
        'recall': float(recall_score(labels_np, preds, zero_division=0)),
        'f1': float(f1_score(labels_np, preds, zero_division=0)),
        'benign_fpr': float(preds[benign_mask].mean()) if benign_mask.any() else 0.0,
        'malicious_label_recall': float(preds[label_mask].mean()) if label_mask.any() else 0.0,
        'preds': preds.tolist(),
    }


def breakdown_by_type(preds: np.ndarray, types_eval: List[str]) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    for t, p in zip(types_eval, preds):
        bucket = out.setdefault(t, {'total': 0, 'predicted_malicious': 0})
        bucket['total'] += 1
        bucket['predicted_malicious'] += int(p == 1)
    return out


def main() -> None:
    set_seed(SEED)
    if not PUBLIC_VAL_DIR or not os.path.isdir(PUBLIC_VAL_DIR):
        raise FileNotFoundError(
            f"PUBLIC_VAL_DIR invalido: {PUBLIC_VAL_DIR!r}. "
            "Use um diretorio public_val limpo e separado do test."
        )
    ARTIFACTS_DIR.mkdir(exist_ok=True)

    print('[1/4] Carregando state_dicts e extraindo features...')
    X, y, types, entries = load_dataset()
    print(f'  Total: {len(y)} amostras | benignos={int((y == 0).sum())} | maliciosos={int((y == 1).sum())}')
    print(f'  Tipos: {sorted(set(types))}')
    print(f'  Feature dim: {X.shape[1]}')

    print('[2/4] Split por clientes disjuntos para treino/dev/calibracao/teste...')
    train_idx, dev_idx, calib_idx, test_idx = split_train_dev_calib_test_by_client(
        entries, dev_size=DEV_SIZE, calib_size=CALIB_SIZE, test_size=TEST_SIZE, seed=SEED
    )
    X_train, X_dev, X_calib, X_test = X[train_idx], X[dev_idx], X[calib_idx], X[test_idx]
    y_train, y_dev, y_calib, y_test = y[train_idx], y[dev_idx], y[calib_idx], y[test_idx]
    y_label_all = np.asarray([1 if t == 'malicious_label' else 0 for t in types], dtype=np.float32)
    y_label_train = y_label_all[train_idx]
    types_dev = [types[i] for i in dev_idx]
    types_calib = [types[i] for i in calib_idx]
    types_test = [types[i] for i in test_idx]
    if OVERSAMPLE_LABEL_FACTOR > 1:
        label_idx = [i for i in train_idx if types[i] == 'malicious_label']
        if label_idx:
            extra_idx = np.repeat(np.asarray(label_idx), OVERSAMPLE_LABEL_FACTOR - 1)
            combined_idx = np.concatenate([train_idx, extra_idx])
            rng = np.random.default_rng(SEED)
            rng.shuffle(combined_idx)
            X_train, y_train = X[combined_idx], y[combined_idx]
            y_label_train = y_label_all[combined_idx]
        print(
            f'  Oversampling malicious_label no treino: '
            f'{len(label_idx)} x {OVERSAMPLE_LABEL_FACTOR}'
        )
    print(f'  Split train: {split_summary(entries, train_idx)}')
    print(f'  Split dev  : {split_summary(entries, dev_idx)}')
    print(f'  Split calib: {split_summary(entries, calib_idx)}')
    print(f'  Split test : {split_summary(entries, test_idx)}')

    scaler = StandardScaler()
    scaler.fit(X[train_idx])
    X_train_s = scaler.transform(X_train).astype(np.float32)
    X_dev_s = scaler.transform(X_dev).astype(np.float32)
    X_calib_s = scaler.transform(X_calib).astype(np.float32)
    X_test_s = scaler.transform(X_test).astype(np.float32)

    Xt_train = torch.from_numpy(X_train_s)
    yt_train = torch.from_numpy(y_train)
    yt_label_train = torch.from_numpy(y_label_train.astype(np.float32))
    Xt_dev = torch.from_numpy(X_dev_s)
    yt_dev = torch.from_numpy(y_dev)
    Xt_calib = torch.from_numpy(X_calib_s)
    yt_calib = torch.from_numpy(y_calib)
    Xt_test = torch.from_numpy(X_test_s)
    yt_test = torch.from_numpy(y_test)

    train_loader = DataLoader(
        TensorDataset(Xt_train, yt_train, yt_label_train),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )

    print('[3/4] Treinando MLPDetector...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MLPDetector(input_dim=TOTAL_FEATURES, hidden=HIDDEN, dropout=DROPOUT).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss()
    label_criterion = nn.BCEWithLogitsLoss()

    Xt_dev_dev = Xt_dev.to(device)
    yt_dev_dev = yt_dev.to(device)
    Xt_calib_dev = Xt_calib.to(device)
    Xt_test_dev = Xt_test.to(device)
    yt_test_dev = yt_test.to(device)

    best_score = (-1.0, -1.0, -1.0)
    best_state = None
    best_epoch = -1
    epochs_without_improve = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for xb, yb, ylb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            ylb = ylb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            label_logits = model.label_logits(xb)
            loss = criterion(logits, yb) + LABEL_LOSS_WEIGHT * label_criterion(label_logits, ylb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        scheduler.step()

        eval_metrics = evaluate(model, Xt_dev_dev, yt_dev_dev)
        logits_dev_epoch, label_scores_dev_epoch = predict_outputs(model, Xt_dev_dev)
        binary_scores_dev_epoch = logits_dev_epoch[:, 1] - logits_dev_epoch[:, 0]
        label_epoch = tune_combined_thresholds(
            binary_scores_dev_epoch,
            label_scores_dev_epoch,
            y_dev,
            types_dev,
            max_benign_fpr=0.05,
        )
        score = (
            float(label_epoch['malicious_label_recall']),
            float(label_epoch['recall']),
            float(eval_metrics['f1']),
        )
        if score > best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1

        if epoch == 1 or epoch % 5 == 0 or epoch == EPOCHS:
            print(
                f'  epoch {epoch:3d}/{EPOCHS} | train_loss={epoch_loss / max(n_batches, 1):.4f} '
                f"| calib f1={eval_metrics['f1']:.4f} acc={eval_metrics['accuracy']:.4f} "
                f"prec={eval_metrics['precision']:.4f} rec={eval_metrics['recall']:.4f} "
                f"| label_fpr05={label_epoch['malicious_label_recall']:.4f} "
                f"| best_label={best_score[0]:.4f}@{best_epoch}"
            )

        if epochs_without_improve >= PATIENCE:
            print(f'  early stopping na epoch {epoch} (sem melhora ha {PATIENCE} epochs)')
            break

    assert best_state is not None
    model.load_state_dict(best_state)

    print('[4/4] Avaliacao final + breakdown por tipo de ataque')
    final = evaluate(model, Xt_test_dev, yt_test_dev)
    preds = np.array(final['preds'])
    logits_calib, label_scores_calib = predict_outputs(model, Xt_calib_dev)
    logits_test, label_scores_test = predict_outputs(model, Xt_test_dev)
    binary_scores_calib = logits_calib[:, 1] - logits_calib[:, 0]
    binary_scores_test = logits_test[:, 1] - logits_test[:, 0]
    tuned_calib = tune_threshold(logits_calib, y_calib)
    tuned = metrics_from_threshold(logits_test, y_test, tuned_calib['threshold'])
    print('\n--- Metricas binarias ---')
    print(f"  accuracy : {final['accuracy']:.4f}")
    print(f"  precision: {final['precision']:.4f}")
    print(f"  recall   : {final['recall']:.4f}")
    print(f"  f1       : {final['f1']:.4f}")
    print(
        f"  threshold: {tuned_calib['threshold']:.4f} "
        f"| tuned_f1={tuned['f1']:.4f} tuned_prec={tuned['precision']:.4f} tuned_rec={tuned['recall']:.4f}"
    )
    print('\n--- classification_report ---')
    report_text = classification_report(
        yt_test.numpy(), preds, target_names=['benign', 'malicious'], zero_division=0
    )
    print(report_text)

    print('--- Breakdown por tipo de ataque ---')
    by_type = breakdown_by_type(preds, types_test)
    by_type_tuned = breakdown_by_type(np.array(tuned['preds']), types_test)
    fpr05_calib = tune_threshold_with_constraint(
        logits_calib, y_calib, types_calib, max_benign_fpr=0.05, objective='malicious_recall'
    )
    fpr05 = metrics_from_threshold(logits_test, y_test, fpr05_calib['threshold'])
    by_type_fpr05 = breakdown_by_type(np.array(fpr05['preds']), types_test)
    label_fpr05_calib = tune_threshold_with_constraint(
        logits_calib, y_calib, types_calib, max_benign_fpr=0.05, objective='label_recall'
    )
    label_fpr05 = metrics_from_threshold(logits_test, y_test, label_fpr05_calib['threshold'])
    by_type_label_fpr05 = breakdown_by_type(np.array(label_fpr05['preds']), types_test)
    label_head_calib = tune_score_threshold_with_constraint(
        label_scores_calib, y_calib, types_calib, max_benign_fpr=0.05, objective='label_recall'
    )
    label_head_preds = (label_scores_test > label_head_calib['threshold']).astype(np.int64)
    label_head_test = {
        'accuracy': float(accuracy_score(y_test, label_head_preds)),
        'precision': float(precision_score(y_test, label_head_preds, zero_division=0)),
        'recall': float(recall_score(y_test, label_head_preds, zero_division=0)),
        'f1': float(f1_score(y_test, label_head_preds, zero_division=0)),
        'benign_fpr': float(label_head_preds[y_test == 0].mean()) if (y_test == 0).any() else 0.0,
        'malicious_label_recall': float(label_head_preds[np.asarray(types_test) == 'malicious_label'].mean()) if (np.asarray(types_test) == 'malicious_label').any() else 0.0,
        'preds': label_head_preds.tolist(),
    }
    by_type_label_head = breakdown_by_type(label_head_preds, types_test)
    combined_calib = tune_combined_thresholds(
        binary_scores_calib, label_scores_calib, y_calib, types_calib, max_benign_fpr=0.05
    )
    combined = combined_metrics_from_thresholds(
        binary_scores_test,
        label_scores_test,
        y_test,
        types_test,
        combined_calib['binary_threshold'],
        combined_calib['label_threshold'],
    )
    by_type_combined = breakdown_by_type(np.asarray(combined['preds']), types_test)
    for t in sorted(by_type):
        b = by_type[t]
        ratio = b['predicted_malicious'] / b['total']
        kind = 'recall' if t != 'benign' else 'FPR'
        print(f"  {t:24s}: predicted_malicious={b['predicted_malicious']}/{b['total']} ({kind}={ratio:.2%})")

    print('\nSalvando artefatos em', ARTIFACTS_DIR)
    torch.save(
        {
            'state_dict': model.state_dict(),
            'input_dim': TOTAL_FEATURES,
            'hidden': list(HIDDEN),
            'dropout': DROPOUT,
            'feature_names': feature_names() + context_feature_names(),
            'base_feature_dim': N_FEATURES,
            'context_feature_dim': N_CONTEXT_FEATURES,
            'has_label_head': True,
        },
        ARTIFACTS_DIR / 'model.pt',
    )
    joblib.dump(scaler, ARTIFACTS_DIR / 'scaler.pkl')
    with open(ARTIFACTS_DIR / 'feature_names.json', 'w') as f:
        json.dump(feature_names() + context_feature_names(), f, indent=2)

    diag_path = ARTIFACTS_DIR / 'score_diagnostics.csv'
    if diag_path.exists():
        diag_path.unlink()
    write_score_diagnostics(
        str(diag_path),
        entries,
        calib_idx,
        'calib',
        logits_calib,
        combined_calib['binary_threshold'],
        label_scores=label_scores_calib,
        label_threshold=combined_calib['label_threshold'],
        combined_preds=combined_calib['preds'],
    )
    write_score_diagnostics(
        str(diag_path),
        entries,
        test_idx,
        'test',
        logits_test,
        combined_calib['binary_threshold'],
        label_scores=label_scores_test,
        label_threshold=combined_calib['label_threshold'],
        combined_preds=combined['preds'],
    )

    report = {
        'best_epoch': best_epoch,
        'best_selection': {
            'metric': 'dev malicious_label recall under benign FPR <= 5%',
            'score': list(best_score),
            'note': 'checkpoint selected on dev split; thresholds selected on calibration split',
        },
        'metrics': {k: final[k] for k in ('accuracy', 'precision', 'recall', 'f1')},
        'tuned': {
            'threshold': tuned_calib['threshold'],
            'accuracy': tuned['accuracy'],
            'precision': tuned['precision'],
            'recall': tuned['recall'],
            'f1': tuned['f1'],
            'by_type': by_type_tuned,
            'note': 'threshold selected on calibration set and reported on held-out test',
        },
        'threshold_fpr05': {
            'threshold': fpr05_calib['threshold'],
            'accuracy': fpr05['accuracy'],
            'precision': fpr05['precision'],
            'recall': fpr05['recall'],
            'f1': fpr05['f1'],
            'benign_fpr': float(np.array(fpr05['preds'])[y_test == 0].mean()) if (y_test == 0).any() else 0.0,
            'malicious_label_recall': float(np.array(fpr05['preds'])[np.asarray(types_test) == 'malicious_label'].mean()) if (np.asarray(types_test) == 'malicious_label').any() else 0.0,
            'by_type': by_type_fpr05,
            'note': 'threshold selected on calibration set with benign FPR <= 5%; metrics reported on held-out test',
        },
        'threshold_label_fpr05': {
            'threshold': label_fpr05_calib['threshold'],
            'accuracy': label_fpr05['accuracy'],
            'precision': label_fpr05['precision'],
            'recall': label_fpr05['recall'],
            'f1': label_fpr05['f1'],
            'benign_fpr': float(np.array(label_fpr05['preds'])[y_test == 0].mean()) if (y_test == 0).any() else 0.0,
            'malicious_label_recall': float(np.array(label_fpr05['preds'])[np.asarray(types_test) == 'malicious_label'].mean()) if (np.asarray(types_test) == 'malicious_label').any() else 0.0,
            'by_type': by_type_label_fpr05,
            'note': 'threshold selected on calibration set with benign FPR <= 5%; metrics reported on held-out test',
        },
        'label_head_fpr05': {
            'threshold': label_head_calib['threshold'],
            'accuracy': label_head_test['accuracy'],
            'precision': label_head_test['precision'],
            'recall': label_head_test['recall'],
            'f1': label_head_test['f1'],
            'benign_fpr': label_head_test['benign_fpr'],
            'malicious_label_recall': label_head_test['malicious_label_recall'],
            'by_type': by_type_label_head,
            'note': 'label-specific head threshold selected on calibration set with benign FPR <= 5%',
        },
        'combined_label_fpr05': {
            'binary_threshold': combined_calib['binary_threshold'],
            'label_threshold': combined_calib['label_threshold'],
            'accuracy': combined['accuracy'],
            'precision': combined['precision'],
            'recall': combined['recall'],
            'f1': combined['f1'],
            'benign_fpr': combined['benign_fpr'],
            'malicious_label_recall': combined['malicious_label_recall'],
            'by_type': by_type_combined,
            'note': 'OR rule: binary_score > binary_threshold or label_score > label_threshold; calibrated with benign FPR <= 5%',
        },
        'by_type': by_type,
        'split_protocol': 'disjoint_client_train_dev_calib_test',
        'split_summary': {
            'train': split_summary(entries, train_idx),
            'dev': split_summary(entries, dev_idx),
            'calib': split_summary(entries, calib_idx),
            'test': split_summary(entries, test_idx),
        },
        'config': {
            'seed': SEED,
            'oversample_label_factor': OVERSAMPLE_LABEL_FACTOR,
            'label_loss_weight': LABEL_LOSS_WEIGHT,
            'hidden': list(HIDDEN),
            'dropout': DROPOUT,
            'lr': LR,
            'weight_decay': WEIGHT_DECAY,
            'epochs': EPOCHS,
            'batch_size': BATCH_SIZE,
            'patience': PATIENCE,
            'test_size': TEST_SIZE,
            'dev_size': DEV_SIZE,
            'calib_size': CALIB_SIZE,
            'public_val_dir': PUBLIC_VAL_DIR,
            'base_feature_dim': N_FEATURES,
            'context_feature_dim': N_CONTEXT_FEATURES,
        },
    }
    with open(ARTIFACTS_DIR / 'report.json', 'w') as f:
        json.dump(report, f, indent=2)

    print(f'\nDONE. Best label_fpr05={best_score[0]:.4f} @ epoch {best_epoch}.')


if __name__ == '__main__':
    main()
