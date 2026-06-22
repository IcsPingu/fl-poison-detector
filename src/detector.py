"""Detector de pesos maliciosos em FL via DistilBERT+LoRA hibrido.

Pipeline:
  state_dicts/*.safetensors -> preprocess_weights (normalizacao per-camada +
  pooling estratificado + bins) + context_features (delta local-global +
  validacao publica) -> DistilBERT+LoRA + ramo tabular -> breakdown por tipo
  de ataque + save modelo final.
"""
from __future__ import annotations

import glob
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import evaluate
import joblib
import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm
from transformers import (
    AutoModel,
    Trainer,
    TrainingArguments,
    set_seed,
)
try:
    from context_features import N_CONTEXT_FEATURES, extract_context_features
    from split_utils import split_summary, split_train_dev_calib_test_by_client, write_score_diagnostics
except ImportError:
    from .context_features import N_CONTEXT_FEATURES, extract_context_features
    from .split_utils import split_summary, split_train_dev_calib_test_by_client, write_score_diagnostics

PAD_ID = 0
NUM_BINS = 10000
MAX_LENGTH = 512
SEED = 42
# Seed do treino fixada na que rendeu o melhor F1 individual (0.892) num
# experimento de ensemble com 5 seeds. Mantida separada de SEED (que controla
# o split estratificado) para nao alterar a particao train/eval.
MODEL_SEED = 15880
STATE_DICTS_DIR = os.environ.get('STATE_DICTS_DIR', 'state_dicts')
PUBLIC_VAL_DIR = os.environ.get('PUBLIC_VAL_DIR', '')
OVERSAMPLE_LABEL_FACTOR = max(1, int(os.environ.get('OVERSAMPLE_LABEL_FACTOR', '1')))
LABEL_LOSS_WEIGHT = float(os.environ.get('LABEL_LOSS_WEIGHT', '1.0'))
TEST_SIZE = 0.2
CALIB_SIZE = 0.2
DEV_SIZE = 0.2
FINAL_MODEL_DIR = os.environ.get('FINAL_MODEL_DIR', './detector_final')
RUN_DIR = os.environ.get('RUN_DIR', './detector_runs/best')
MODEL_NAME = 'distilbert-base-uncased'
CANONICAL_WEIGHT_ALIASES = [
    ('conv1.0.weight', 'base.conv1.0.weight'),
    ('conv2.0.weight', 'base.conv2.0.weight'),
    ('fc1.0.weight', 'base.fc1.0.weight'),
    ('fc.weight', 'head.weight'),
]

# Carregadas em main(); compute_metrics referencia para casar com a assinatura
# (eval_pred) -> dict que o Trainer espera.
_accuracy = None
_f1 = None
_precision = None
_recall = None
_metric_types = None


def _normalize_layer(t: torch.Tensor) -> torch.Tensor:
    # Normalizacao por quantis (q5/q95) e mais robusta a outliers que min/max --
    # importante para detectar `noise`, que estica caudas mas preserva o nucleo.
    lo = torch.quantile(t, 0.05)
    hi = torch.quantile(t, 0.95)
    return ((t - lo) / (hi - lo + 1e-8)).clamp(0.0, 1.0)


def _ordered_weight_items(state_dict):
    used = set()
    ordered = []
    for aliases in CANONICAL_WEIGHT_ALIASES:
        key = next((candidate for candidate in aliases if candidate in state_dict), None)
        if key is not None:
            ordered.append((key, state_dict[key]))
            used.add(key)
    remaining = [
        (k, state_dict[k])
        for k in sorted(state_dict)
        if 'weight' in k and k not in used
    ]
    return ordered + remaining


def preprocess_weights(state_dict, max_length=MAX_LENGTH, num_bins=NUM_BINS) -> List[int]:
    """Normaliza cada camada com 'weight' no nome, concatena e amostra
    `max_length` posicoes uniformemente distribuidas (linspace) sobre o vetor
    inteiro -- garante representacao de todas as camadas, nao so as primeiras.

    Bins ocupam [1, num_bins]; PAD_ID=0 fica reservado pra padding.
    """
    parts = [_normalize_layer(v.flatten().float()) for _, v in _ordered_weight_items(state_dict)]
    if not parts:
        raise ValueError("state_dict sem tensores de peso ('weight'); nao e um update de modelo valido.")
    weights_norm = torch.cat(parts)
    n = len(weights_norm)
    if n >= max_length:
        idx = torch.linspace(0, n - 1, steps=max_length).long()
        sampled = weights_norm[idx]
    else:
        sampled = weights_norm
    binned = (sampled * (num_bins - 1)).long() + 1
    if len(binned) < max_length:
        pad = torch.full((max_length - len(binned),), PAD_ID, dtype=torch.long)
        binned = torch.cat([binned, pad])
    return binned.tolist()


def tokenize_function(examples):
    return {
        'input_ids': examples['inputs'],
        'attention_mask': [[1 if tok != PAD_ID else 0 for tok in ids] for ids in examples['inputs']],
    }


def compute_metrics(eval_pred):
    logits = getattr(eval_pred, 'predictions', None)
    labels = getattr(eval_pred, 'label_ids', None)
    if logits is None or labels is None:
        logits, labels = eval_pred
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    if isinstance(labels, (tuple, list)):
        labels = labels[0]
    labels = np.asarray(labels)
    predictions = np.argmax(logits, axis=-1)
    out = {
        'accuracy': _accuracy.compute(predictions=predictions, references=labels)['accuracy'],
        'f1': _f1.compute(predictions=predictions, references=labels, average='binary')['f1'],
        'precision': _precision.compute(predictions=predictions, references=labels, average='binary')['precision'],
        'recall': _recall.compute(predictions=predictions, references=labels, average='binary')['recall'],
    }
    if _metric_types is not None and len(_metric_types) == len(labels):
        label_fpr05 = tune_threshold_with_constraint(
            logits, labels, list(_metric_types), max_benign_fpr=0.05, objective='label_recall'
        )
        out['label_recall_fpr05'] = label_fpr05['malicious_label_recall']
        out['benign_fpr_at_label_fpr05'] = label_fpr05['benign_fpr']
    return out


def load_entries() -> List[Dict]:
    files = [
        f for f in sorted(glob.glob(os.path.join(STATE_DICTS_DIR, '*.safetensors')))
        if os.path.exists(f.replace('.safetensors', '.json'))
    ]
    assert files, (
        f"Nenhum .safetensors em '{STATE_DICTS_DIR}/'. "
        "Rode o notebook BertModelsclassify.ipynb pra gerar os state_dicts."
    )
    entries: List[Dict] = []
    for f in tqdm(files, desc='preprocess state_dicts', unit='file'):
        sd = load_file(f)
        with open(f.replace('.safetensors', '.json')) as jf:
            meta = json.load(jf)
        global_sd = None
        global_ref = meta.get('global_state')
        if global_ref:
            global_path = os.path.join(STATE_DICTS_DIR, global_ref)
            if os.path.exists(global_path):
                global_sd = load_file(global_path)
        ctx_feats, _ = extract_context_features(
            sd,
            global_sd=global_sd,
            public_val_dir=PUBLIC_VAL_DIR or None,
        )
        entries.append({
            'sample_id': os.path.splitext(os.path.basename(f))[0],
            'inputs': preprocess_weights(sd),
            'context_features': ctx_feats.astype(np.float32).tolist(),
            'labels': int(meta['label']),
            'type': meta.get('type', 'unknown'),
            'round': int(meta.get('round', 0)),
            'client_id': int(meta.get('client_id', -1)),
        })
    return entries


def tune_threshold(logits: np.ndarray, labels: np.ndarray, n_grid: int = 200) -> Dict:
    """Procura o threshold em (logit_mal - logit_ben) que maximiza F1.

    Default do argmax equivale a threshold > 0. Aqui varremos um grid para
    deslocar a fronteira e potencialmente equilibrar precision/recall melhor.
    Aviso: o threshold e tunado no proprio eval -- nao temos val separado --
    entao o ganho relatado e otimista (~uns pontos de F1).
    """
    scores = logits[:, 1] - logits[:, 0]
    eps = max(float(np.ptp(scores)) * 1e-6, 1e-6)
    candidates = np.linspace(float(scores.min()) - eps, float(scores.max()) + eps, n_grid)
    best = {'f1': -1.0, 'threshold': 0.0}
    for t in candidates:
        preds = (scores > t).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best['f1']:
            best = {
                'f1': float(f1),
                'threshold': float(t),
                'precision': float(precision_score(labels, preds, zero_division=0)),
                'recall': float(recall_score(labels, preds, zero_division=0)),
                'preds': preds,
            }
    return best


def tune_threshold_with_constraint(
    logits: np.ndarray,
    labels: np.ndarray,
    types: List[str],
    max_benign_fpr: float = 0.05,
    objective: str = 'malicious_recall',
    n_grid: int = 400,
) -> Dict:
    scores = logits[:, 1] - logits[:, 0]
    eps = max(float(np.ptp(scores)) * 1e-6, 1e-6)
    candidates = np.linspace(float(scores.min()) - eps, float(scores.max()) + eps, n_grid)
    labels_np = np.asarray(labels)
    types_np = np.asarray(types)
    benign_mask = labels_np == 0
    label_mask = types_np == 'malicious_label'
    best = None

    for t in candidates:
        preds = (scores > t).astype(int)
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
            'threshold': float(t),
            'accuracy': float(accuracy_score(labels_np, preds)),
            'precision': precision,
            'recall': malicious_recall,
            'f1': f1,
            'benign_fpr': benign_fpr,
            'malicious_label_recall': label_recall,
            'preds': preds,
            '_key': key,
        }
        if best is None or item['_key'] > best['_key']:
            best = item

    if best is None:
        t = float(scores.max() + 1e-6)
        preds = (scores > t).astype(int)
        best = {
            'threshold': t,
            'accuracy': float(accuracy_score(labels_np, preds)),
            'precision': float(precision_score(labels_np, preds, zero_division=0)),
            'recall': float(recall_score(labels_np, preds, zero_division=0)),
            'f1': float(f1_score(labels_np, preds, zero_division=0)),
            'benign_fpr': 0.0,
            'malicious_label_recall': 0.0,
            'preds': preds,
            '_key': (0.0, 0.0, 0.0, 0.0),
        }
    best.pop('_key', None)
    return best


class HybridBertDetector(nn.Module):
    def __init__(self, context_dim: int = N_CONTEXT_FEATURES, tabular_dim: int = 128):
        super().__init__()
        base = AutoModel.from_pretrained(MODEL_NAME)
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=['q_lin', 'v_lin'],
        )
        self.bert = get_peft_model(base, lora_config)
        hidden = int(base.config.hidden_size)
        self.context_dim = context_dim
        self.tabular_dim = tabular_dim
        self.tabular = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, tabular_dim),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(hidden + tabular_dim, 2),
        )
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden + tabular_dim, hidden + tabular_dim),
            nn.Sigmoid(),
        )
        self.label_classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(hidden + tabular_dim, 1),
        )
        self.loss_fn = nn.CrossEntropyLoss()
        self.label_loss_fn = nn.BCEWithLogitsLoss()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        context_features=None,
        labels=None,
        label_targets=None,
        return_label_logits: bool = False,
    ):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0]
        ctx = context_features.to(dtype=pooled.dtype)
        tab = self.tabular(ctx)
        fused = torch.cat([pooled, tab], dim=1)
        fused = fused * self.fusion_gate(fused)
        logits = self.classifier(fused)
        loss = self.loss_fn(logits, labels) if labels is not None else None
        label_logits = self.label_classifier(fused).squeeze(-1)
        if label_targets is not None:
            aux_loss = self.label_loss_fn(label_logits, label_targets.to(dtype=label_logits.dtype))
            loss = aux_loss * LABEL_LOSS_WEIGHT if loss is None else loss + aux_loss * LABEL_LOSS_WEIGHT
        out = {'loss': loss, 'logits': logits}
        if return_label_logits:
            out['label_logits'] = label_logits
        return out

    def save_hybrid(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        self.bert.save_pretrained(output_dir)
        torch.save(
            {
                'context_dim': self.context_dim,
                'tabular_dim': self.tabular_dim,
                'tabular_state_dict': self.tabular.state_dict(),
                'fusion_gate_state_dict': self.fusion_gate.state_dict(),
                'classifier_state_dict': self.classifier.state_dict(),
                'label_classifier_state_dict': self.label_classifier.state_dict(),
                'model_name': MODEL_NAME,
            },
            os.path.join(output_dir, 'hybrid_head.pt'),
        )


def build_and_train(seed: int, tokenized_train, tokenized_eval, run_dir: str):
    """Treina DistilBERT+LoRA com ramo tabular contextual."""
    set_seed(seed)
    model = HybridBertDetector(context_dim=N_CONTEXT_FEATURES)

    training_args = TrainingArguments(
        output_dir=run_dir,
        num_train_epochs=15,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-4,
        weight_decay=0.01,
        lr_scheduler_type='cosine',
        warmup_ratio=0.06,
        eval_strategy='epoch',
        logging_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model='label_recall_fpr05',
        greater_is_better=True,
        report_to='none',
        seed=seed,
        remove_unused_columns=False,
        label_names=['labels', 'label_targets'],
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    pred_output = trainer.predict(tokenized_eval)
    return trainer, pred_output


def breakdown_by_type(predictions: np.ndarray, types_eval: List[str]) -> Dict[str, Dict[str, int]]:
    grouped: Dict[str, Dict[str, int]] = defaultdict(lambda: {'total': 0, 'predicted_malicious': 0})
    for t, p in zip(types_eval, predictions):
        grouped[t]['total'] += 1
        grouped[t]['predicted_malicious'] += int(p == 1)
    print('--- Breakdown por tipo de ataque ---')
    for t in sorted(grouped):
        b = grouped[t]
        ratio = b['predicted_malicious'] / b['total']
        kind = 'recall' if t != 'benign' else 'FPR'
        print(f"  {t:24s}: predicted_malicious={b['predicted_malicious']}/{b['total']} ({kind}={ratio:.2%})")
    return dict(grouped)


def threshold_metrics(logits: np.ndarray, labels: np.ndarray, threshold: float) -> Dict:
    preds = ((logits[:, 1] - logits[:, 0]) > threshold).astype(int)
    labels_np = np.asarray(labels)
    return {
        'accuracy': float(accuracy_score(labels_np, preds)),
        'precision': float(precision_score(labels_np, preds, zero_division=0)),
        'recall': float(recall_score(labels_np, preds, zero_division=0)),
        'f1': float(f1_score(labels_np, preds, zero_division=0)),
        'preds': preds,
    }


def _threshold_candidates(scores: np.ndarray, n_grid: int = 400) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    eps = max(float(np.ptp(scores)) * 1e-6, 1e-6)
    return np.linspace(float(scores.min()) - eps, float(scores.max()) + eps, n_grid)


def tune_score_threshold_with_constraint(
    scores: np.ndarray,
    labels: np.ndarray,
    types: List[str],
    max_benign_fpr: float = 0.05,
) -> Dict:
    labels_np = np.asarray(labels)
    types_np = np.asarray(types)
    benign_mask = labels_np == 0
    label_mask = types_np == 'malicious_label'
    best = None
    for threshold in _threshold_candidates(scores, n_grid=400):
        preds = (scores > threshold).astype(int)
        benign_fpr = float(preds[benign_mask].mean()) if benign_mask.any() else 0.0
        if benign_fpr > max_benign_fpr:
            continue
        malicious_recall = float(recall_score(labels_np, preds, zero_division=0))
        label_recall = float(preds[label_mask].mean()) if label_mask.any() else 0.0
        f1 = float(f1_score(labels_np, preds, zero_division=0))
        item = {
            'threshold': float(threshold),
            'accuracy': float(accuracy_score(labels_np, preds)),
            'precision': float(precision_score(labels_np, preds, zero_division=0)),
            'recall': malicious_recall,
            'f1': f1,
            'benign_fpr': benign_fpr,
            'malicious_label_recall': label_recall,
            'preds': preds,
            '_key': (label_recall, malicious_recall, f1, -benign_fpr),
        }
        if best is None or item['_key'] > best['_key']:
            best = item
    if best is None:
        threshold = float(np.max(scores) + 1e-6)
        preds = (scores > threshold).astype(int)
        best = {
            'threshold': threshold,
            'accuracy': float(accuracy_score(labels_np, preds)),
            'precision': float(precision_score(labels_np, preds, zero_division=0)),
            'recall': float(recall_score(labels_np, preds, zero_division=0)),
            'f1': float(f1_score(labels_np, preds, zero_division=0)),
            'benign_fpr': 0.0,
            'malicious_label_recall': 0.0,
            'preds': preds,
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
) -> Dict:
    labels_np = np.asarray(labels)
    types_np = np.asarray(types)
    benign_mask = labels_np == 0
    label_mask = types_np == 'malicious_label'
    best = None
    for binary_threshold in _threshold_candidates(binary_scores, n_grid=80):
        binary_hit = binary_scores > binary_threshold
        for label_threshold in _threshold_candidates(label_scores, n_grid=80):
            preds = np.logical_or(binary_hit, label_scores > label_threshold).astype(int)
            benign_fpr = float(preds[benign_mask].mean()) if benign_mask.any() else 0.0
            if benign_fpr > max_benign_fpr:
                continue
            malicious_recall = float(recall_score(labels_np, preds, zero_division=0))
            label_recall = float(preds[label_mask].mean()) if label_mask.any() else 0.0
            f1 = float(f1_score(labels_np, preds, zero_division=0))
            item = {
                'binary_threshold': float(binary_threshold),
                'label_threshold': float(label_threshold),
                'accuracy': float(accuracy_score(labels_np, preds)),
                'precision': float(precision_score(labels_np, preds, zero_division=0)),
                'recall': malicious_recall,
                'f1': f1,
                'benign_fpr': benign_fpr,
                'malicious_label_recall': label_recall,
                'preds': preds,
                '_key': (label_recall, malicious_recall, f1, -benign_fpr),
            }
            if best is None or item['_key'] > best['_key']:
                best = item
    if best is None:
        preds = np.zeros_like(labels_np)
        best = {
            'binary_threshold': float(np.max(binary_scores) + 1e-6),
            'label_threshold': float(np.max(label_scores) + 1e-6),
            'accuracy': float(accuracy_score(labels_np, preds)),
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'benign_fpr': 0.0,
            'malicious_label_recall': 0.0,
            'preds': preds,
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
    preds = np.logical_or(binary_scores > binary_threshold, label_scores > label_threshold).astype(int)
    benign_mask = labels_np == 0
    label_mask = types_np == 'malicious_label'
    return {
        'accuracy': float(accuracy_score(labels_np, preds)),
        'precision': float(precision_score(labels_np, preds, zero_division=0)),
        'recall': float(recall_score(labels_np, preds, zero_division=0)),
        'f1': float(f1_score(labels_np, preds, zero_division=0)),
        'benign_fpr': float(preds[benign_mask].mean()) if benign_mask.any() else 0.0,
        'malicious_label_recall': float(preds[label_mask].mean()) if label_mask.any() else 0.0,
        'preds': preds,
    }


def predict_label_scores(trainer: Trainer, dataset) -> np.ndarray:
    model = trainer.model
    model.eval()
    scores: List[np.ndarray] = []
    dataloader = trainer.get_eval_dataloader(dataset)
    device = trainer.args.device
    with torch.no_grad():
        for batch in dataloader:
            batch = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                context_features=batch['context_features'],
                return_label_logits=True,
            )
            scores.append(outputs['label_logits'].detach().cpu().numpy())
    return np.concatenate(scores, axis=0)


def main() -> None:
    global _accuracy, _f1, _precision, _recall, _metric_types
    set_seed(SEED)
    if not PUBLIC_VAL_DIR or not os.path.isdir(PUBLIC_VAL_DIR):
        raise FileNotFoundError(
            f"PUBLIC_VAL_DIR invalido: {PUBLIC_VAL_DIR!r}. "
            "Use um diretorio public_val limpo e separado do test."
        )

    entries = load_entries()
    n_benign = sum(1 for e in entries if e['labels'] == 0)
    n_mal = len(entries) - n_benign
    types_all = [e['type'] for e in entries]
    print(f'Carregadas {len(entries)} amostras: benignos={n_benign}, maliciosos={n_mal}')
    print(f'Tipos: {sorted(set(types_all))}')

    train_idx, dev_idx, calib_idx, test_idx = split_train_dev_calib_test_by_client(
        entries, dev_size=DEV_SIZE, calib_size=CALIB_SIZE, test_size=TEST_SIZE, seed=SEED
    )
    print(f'Split train: {split_summary(entries, train_idx)}')
    print(f'Split dev  : {split_summary(entries, dev_idx)}')
    print(f'Split calib: {split_summary(entries, calib_idx)}')
    print(f'Split test : {split_summary(entries, test_idx)}')

    scaler = StandardScaler()
    X_ctx_train = np.asarray([entries[i]['context_features'] for i in train_idx], dtype=np.float32)
    scaler.fit(X_ctx_train)

    def make_record(i: int) -> Dict:
        ctx = scaler.transform(
            np.asarray(entries[i]['context_features'], dtype=np.float32).reshape(1, -1)
        )[0].astype(np.float32)
        return {
            'inputs': entries[i]['inputs'],
            'context_features': ctx.tolist(),
            'label_targets': 1.0 if entries[i]['type'] == 'malicious_label' else 0.0,
            'labels': entries[i]['labels'],
        }

    train_records = [make_record(i) for i in train_idx]
    if OVERSAMPLE_LABEL_FACTOR > 1:
        label_records = [
            make_record(i)
            for i in train_idx
            if entries[i]['type'] == 'malicious_label'
        ]
        for _ in range(OVERSAMPLE_LABEL_FACTOR - 1):
            train_records.extend(label_records)
        print(
            f'Oversampling malicious_label no treino: '
            f'{len(label_records)} x {OVERSAMPLE_LABEL_FACTOR}'
        )
    dev_records = [make_record(i) for i in dev_idx]
    calib_records = [make_record(i) for i in calib_idx]
    test_records = [make_record(i) for i in test_idx]
    types_dev = [types_all[i] for i in dev_idx]
    types_calib = [types_all[i] for i in calib_idx]
    types_test = [types_all[i] for i in test_idx]
    print(
        f'Train: {len(train_records)} | Dev: {len(dev_records)} | '
        f'Calib: {len(calib_records)} | Test: {len(test_records)}'
    )

    train_ds = Dataset.from_list(train_records)
    dev_ds = Dataset.from_list(dev_records)
    calib_ds = Dataset.from_list(calib_records)
    test_ds = Dataset.from_list(test_records)
    tokenized_train = train_ds.map(tokenize_function, batched=True, remove_columns=['inputs'])
    tokenized_dev = dev_ds.map(tokenize_function, batched=True, remove_columns=['inputs'])
    tokenized_calib = calib_ds.map(tokenize_function, batched=True, remove_columns=['inputs'])
    tokenized_test = test_ds.map(tokenize_function, batched=True, remove_columns=['inputs'])

    _accuracy = evaluate.load('accuracy')
    _f1 = evaluate.load('f1')
    _precision = evaluate.load('precision')
    _recall = evaluate.load('recall')

    os.makedirs(FINAL_MODEL_DIR, exist_ok=True)
    print(f'\n========== Treinando modelo (seed={MODEL_SEED}) ==========')
    _metric_types = types_dev
    trainer, pred_output = build_and_train(
        MODEL_SEED, tokenized_train, tokenized_dev, RUN_DIR
    )

    _metric_types = None
    calib_output = trainer.predict(tokenized_calib)
    calib_logits = calib_output.predictions
    calib_label_scores = predict_label_scores(trainer, tokenized_calib)
    calib_labels = calib_output.label_ids
    if isinstance(calib_labels, (tuple, list)):
        calib_labels = calib_labels[0]
    tuned = tune_threshold(calib_logits, calib_labels)
    fpr05 = tune_threshold_with_constraint(
        calib_logits, calib_labels, types_calib, max_benign_fpr=0.05, objective='malicious_recall'
    )
    label_fpr05 = tune_threshold_with_constraint(
        calib_logits, calib_labels, types_calib, max_benign_fpr=0.05, objective='label_recall'
    )

    _metric_types = None
    test_output = trainer.predict(tokenized_test)
    logits = test_output.predictions
    label_scores = predict_label_scores(trainer, tokenized_test)
    labels = test_output.label_ids
    if isinstance(labels, (tuple, list)):
        labels = labels[0]
    preds_default = np.argmax(logits, axis=-1)
    acc_default = float(accuracy_score(labels, preds_default))
    f1_default = float(f1_score(labels, preds_default, zero_division=0))
    prec_default = float(precision_score(labels, preds_default, zero_division=0))
    rec_default = float(recall_score(labels, preds_default, zero_division=0))

    print('\n=== Avaliacao final (threshold default = argmax) ===')
    print(f'  acc={acc_default:.4f} F1={f1_default:.4f} prec={prec_default:.4f} rec={rec_default:.4f}')
    by_type_default = breakdown_by_type(preds_default, types_test)

    print('\n=== Threshold calibrado ===')
    print(
        f"  threshold={tuned['threshold']:.4f} | F1={tuned['f1']:.4f} "
        f"prec={tuned['precision']:.4f} rec={tuned['recall']:.4f}  "
        '[calibration set]'
    )
    tuned_test = threshold_metrics(logits, labels, tuned['threshold'])
    fpr05_test = threshold_metrics(logits, labels, fpr05['threshold'])
    label_fpr05_test = threshold_metrics(logits, labels, label_fpr05['threshold'])
    binary_scores_calib = calib_logits[:, 1] - calib_logits[:, 0]
    binary_scores_test = logits[:, 1] - logits[:, 0]
    label_head = tune_score_threshold_with_constraint(
        calib_label_scores, calib_labels, types_calib, max_benign_fpr=0.05
    )
    label_head_preds = (label_scores > label_head['threshold']).astype(int)
    label_head_test = {
        'accuracy': float(accuracy_score(labels, label_head_preds)),
        'f1': float(f1_score(labels, label_head_preds, zero_division=0)),
        'precision': float(precision_score(labels, label_head_preds, zero_division=0)),
        'recall': float(recall_score(labels, label_head_preds, zero_division=0)),
        'benign_fpr': float(label_head_preds[np.asarray(labels) == 0].mean()) if (np.asarray(labels) == 0).any() else 0.0,
        'malicious_label_recall': float(label_head_preds[np.asarray(types_test) == 'malicious_label'].mean()) if (np.asarray(types_test) == 'malicious_label').any() else 0.0,
        'preds': label_head_preds,
    }
    combined_calib = tune_combined_thresholds(
        binary_scores_calib, calib_label_scores, calib_labels, types_calib, max_benign_fpr=0.05
    )
    combined_test = combined_metrics_from_thresholds(
        binary_scores_test,
        label_scores,
        labels,
        types_test,
        combined_calib['binary_threshold'],
        combined_calib['label_threshold'],
    )
    preds_tuned = tuned_test['preds']
    preds_fpr05 = fpr05_test['preds']
    preds_label_fpr05 = label_fpr05_test['preds']
    by_type_tuned = breakdown_by_type(preds_tuned, types_test)
    by_type_fpr05 = breakdown_by_type(preds_fpr05, types_test)
    by_type_label_fpr05 = breakdown_by_type(preds_label_fpr05, types_test)
    by_type_label_head = breakdown_by_type(label_head_test['preds'], types_test)
    by_type_combined = breakdown_by_type(combined_test['preds'], types_test)

    diag_path = os.path.join(FINAL_MODEL_DIR, 'score_diagnostics.csv')
    if os.path.exists(diag_path):
        os.remove(diag_path)
    write_score_diagnostics(
        diag_path,
        entries,
        calib_idx,
        'calib',
        calib_logits,
        combined_calib['binary_threshold'],
        label_scores=calib_label_scores,
        label_threshold=combined_calib['label_threshold'],
        combined_preds=combined_calib['preds'],
    )
    write_score_diagnostics(
        diag_path,
        entries,
        test_idx,
        'test',
        logits,
        combined_calib['binary_threshold'],
        label_scores=label_scores,
        label_threshold=combined_calib['label_threshold'],
        combined_preds=combined_test['preds'],
    )

    trainer.model.save_hybrid(FINAL_MODEL_DIR)
    joblib.dump(scaler, os.path.join(FINAL_MODEL_DIR, 'context_scaler.pkl'))
    with open(os.path.join(FINAL_MODEL_DIR, 'metrics.json'), 'w') as f:
        json.dump(
            {
                'split_seed': SEED,
                'model_seed': MODEL_SEED,
                'oversample_label_factor': OVERSAMPLE_LABEL_FACTOR,
                'label_loss_weight': LABEL_LOSS_WEIGHT,
                'hybrid_context_dim': N_CONTEXT_FEATURES,
                'public_val_dir': PUBLIC_VAL_DIR,
                'split_protocol': 'disjoint_client_train_dev_calib_test',
                'best_selection': {
                    'metric': 'dev malicious_label recall under benign FPR <= 5%',
                    'best_metric': trainer.state.best_metric,
                    'best_model_checkpoint': trainer.state.best_model_checkpoint,
                    'note': 'checkpoint selected on dev split; thresholds selected on calibration split',
                },
                'dev_size': DEV_SIZE,
                'calib_size': CALIB_SIZE,
                'split_summary': {
                    'train': split_summary(entries, train_idx),
                    'dev': split_summary(entries, dev_idx),
                    'calib': split_summary(entries, calib_idx),
                    'test': split_summary(entries, test_idx),
                },
                'default_argmax': {
                    'accuracy': acc_default,
                    'f1': f1_default,
                    'precision': prec_default,
                    'recall': rec_default,
                    'by_type': by_type_default,
                },
                'tuned': {
                    'threshold': tuned['threshold'],
                    'accuracy': tuned_test['accuracy'],
                    'f1': tuned_test['f1'],
                    'precision': tuned_test['precision'],
                    'recall': tuned_test['recall'],
                    'by_type': by_type_tuned,
                    'note': 'threshold selected on calibration set and reported on held-out test',
                },
                'threshold_fpr05': {
                    'threshold': fpr05['threshold'],
                    'accuracy': fpr05_test['accuracy'],
                    'f1': fpr05_test['f1'],
                    'precision': fpr05_test['precision'],
                    'recall': fpr05_test['recall'],
                    'benign_fpr': float(preds_fpr05[np.asarray(labels) == 0].mean()) if (np.asarray(labels) == 0).any() else 0.0,
                    'malicious_label_recall': float(preds_fpr05[np.asarray(types_test) == 'malicious_label'].mean()) if (np.asarray(types_test) == 'malicious_label').any() else 0.0,
                    'by_type': by_type_fpr05,
                    'note': 'threshold selected on calibration set with benign FPR <= 5%; metrics reported on held-out test',
                },
                'threshold_label_fpr05': {
                    'threshold': label_fpr05['threshold'],
                    'accuracy': label_fpr05_test['accuracy'],
                    'f1': label_fpr05_test['f1'],
                    'precision': label_fpr05_test['precision'],
                    'recall': label_fpr05_test['recall'],
                    'benign_fpr': float(preds_label_fpr05[np.asarray(labels) == 0].mean()) if (np.asarray(labels) == 0).any() else 0.0,
                    'malicious_label_recall': float(preds_label_fpr05[np.asarray(types_test) == 'malicious_label'].mean()) if (np.asarray(types_test) == 'malicious_label').any() else 0.0,
                    'by_type': by_type_label_fpr05,
                    'note': 'threshold selected on calibration set with benign FPR <= 5%; metrics reported on held-out test',
                },
                'label_head_fpr05': {
                    'threshold': label_head['threshold'],
                    'accuracy': label_head_test['accuracy'],
                    'f1': label_head_test['f1'],
                    'precision': label_head_test['precision'],
                    'recall': label_head_test['recall'],
                    'benign_fpr': label_head_test['benign_fpr'],
                    'malicious_label_recall': label_head_test['malicious_label_recall'],
                    'by_type': by_type_label_head,
                    'note': 'label-specific head threshold selected on calibration set with benign FPR <= 5%',
                },
                'combined_label_fpr05': {
                    'binary_threshold': combined_calib['binary_threshold'],
                    'label_threshold': combined_calib['label_threshold'],
                    'accuracy': combined_test['accuracy'],
                    'f1': combined_test['f1'],
                    'precision': combined_test['precision'],
                    'recall': combined_test['recall'],
                    'benign_fpr': combined_test['benign_fpr'],
                    'malicious_label_recall': combined_test['malicious_label_recall'],
                    'by_type': by_type_combined,
                    'note': 'OR rule: binary_score > binary_threshold or label_score > label_threshold; calibrated with benign FPR <= 5%',
                },
            },
            f,
            indent=2,
        )
    print(f'\nDONE. Modelo + metrics.json em {FINAL_MODEL_DIR}/')


if __name__ == '__main__':
    main()
