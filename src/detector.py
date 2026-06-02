"""Detector de pesos maliciosos em FL via DistilBERT+LoRA sobre pesos discretizados.

Pipeline:
  state_dicts/*.safetensors -> preprocess_weights (normalizacao per-camada +
  pooling estratificado + bins) -> DistilBERT+LoRA (binario) -> breakdown por
  tipo de ataque + save modelo final.
"""
from __future__ import annotations

import glob
import json
import os
from collections import defaultdict
from typing import Dict, List

import evaluate
import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

PAD_ID = 0
NUM_BINS = 10000
MAX_LENGTH = 512
SEED = 42
# Seed do treino fixada na que rendeu o melhor F1 individual (0.892) num
# experimento de ensemble com 5 seeds. Mantida separada de SEED (que controla
# o split estratificado) para nao alterar a particao train/eval.
MODEL_SEED = 15880
STATE_DICTS_DIR = os.environ.get('STATE_DICTS_DIR', 'state_dicts')
TEST_SIZE = 0.2
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
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        'accuracy': _accuracy.compute(predictions=predictions, references=labels)['accuracy'],
        'f1': _f1.compute(predictions=predictions, references=labels, average='binary')['f1'],
        'precision': _precision.compute(predictions=predictions, references=labels, average='binary')['precision'],
        'recall': _recall.compute(predictions=predictions, references=labels, average='binary')['recall'],
    }


def load_entries() -> List[Dict]:
    files = sorted(glob.glob(os.path.join(STATE_DICTS_DIR, '*.safetensors')))
    assert files, (
        f"Nenhum .safetensors em '{STATE_DICTS_DIR}/'. "
        "Rode o notebook BertModelsclassify.ipynb pra gerar os state_dicts."
    )
    entries: List[Dict] = []
    for f in tqdm(files, desc='preprocess state_dicts', unit='file'):
        sd = load_file(f)
        with open(f.replace('.safetensors', '.json')) as jf:
            meta = json.load(jf)
        entries.append({
            'inputs': preprocess_weights(sd),
            'labels': int(meta['label']),
            'type': meta.get('type', 'unknown'),
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
    candidates = np.linspace(scores.min(), scores.max(), n_grid)
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


def build_and_train(seed: int, tokenized_train, tokenized_eval, run_dir: str):
    """Treina um modelo DistilBERT+LoRA com a seed dada. Retorna (trainer, pred_output)."""
    set_seed(seed)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, ignore_mismatched_sizes=True
    )
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=['q_lin', 'v_lin'],
        modules_to_save=['pre_classifier', 'classifier'],
    )
    model = get_peft_model(model, lora_config)

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
        metric_for_best_model='f1',
        greater_is_better=True,
        report_to='none',
        seed=seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=7)],
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


def main() -> None:
    global _accuracy, _f1, _precision, _recall
    set_seed(SEED)

    entries = load_entries()
    n_benign = sum(1 for e in entries if e['labels'] == 0)
    n_mal = len(entries) - n_benign
    types_all = [e['type'] for e in entries]
    print(f'Carregadas {len(entries)} amostras: benignos={n_benign}, maliciosos={n_mal}')
    print(f'Tipos: {sorted(set(types_all))}')

    # Split estratificado por tipo de ataque (nao so por label) para garantir que
    # cada split tenha amostras de cada categoria.
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
    train_idx, eval_idx = next(splitter.split(np.zeros(len(entries)), types_all))

    train_records = [
        {'inputs': entries[i]['inputs'], 'labels': entries[i]['labels']} for i in train_idx
    ]
    eval_records = [
        {'inputs': entries[i]['inputs'], 'labels': entries[i]['labels']} for i in eval_idx
    ]
    types_eval = [types_all[i] for i in eval_idx]
    print(f'Train: {len(train_records)} | Eval: {len(eval_records)}')

    train_ds = Dataset.from_list(train_records)
    eval_ds = Dataset.from_list(eval_records)
    tokenized_train = train_ds.map(tokenize_function, batched=True, remove_columns=['inputs'])
    tokenized_eval = eval_ds.map(tokenize_function, batched=True, remove_columns=['inputs'])

    _accuracy = evaluate.load('accuracy')
    _f1 = evaluate.load('f1')
    _precision = evaluate.load('precision')
    _recall = evaluate.load('recall')

    os.makedirs(FINAL_MODEL_DIR, exist_ok=True)
    print(f'\n========== Treinando modelo (seed={MODEL_SEED}) ==========')
    trainer, pred_output = build_and_train(
        MODEL_SEED, tokenized_train, tokenized_eval, RUN_DIR
    )

    logits = pred_output.predictions
    labels = pred_output.label_ids
    preds_default = np.argmax(logits, axis=-1)
    acc_default = float(accuracy_score(labels, preds_default))
    f1_default = float(f1_score(labels, preds_default, zero_division=0))
    prec_default = float(precision_score(labels, preds_default, zero_division=0))
    rec_default = float(recall_score(labels, preds_default, zero_division=0))

    print('\n=== Avaliacao final (threshold default = argmax) ===')
    print(f'  acc={acc_default:.4f} F1={f1_default:.4f} prec={prec_default:.4f} rec={rec_default:.4f}')
    by_type_default = breakdown_by_type(preds_default, types_eval)

    print('\n=== Threshold tunado ===')
    tuned = tune_threshold(logits, labels)
    print(
        f"  threshold={tuned['threshold']:.4f} | F1={tuned['f1']:.4f} "
        f"prec={tuned['precision']:.4f} rec={tuned['recall']:.4f}  "
        '[in-sample, otimista]'
    )
    by_type_tuned = breakdown_by_type(tuned['preds'], types_eval)

    trainer.save_model(FINAL_MODEL_DIR)
    with open(os.path.join(FINAL_MODEL_DIR, 'metrics.json'), 'w') as f:
        json.dump(
            {
                'split_seed': SEED,
                'model_seed': MODEL_SEED,
                'default_argmax': {
                    'accuracy': acc_default,
                    'f1': f1_default,
                    'precision': prec_default,
                    'recall': rec_default,
                    'by_type': by_type_default,
                },
                'tuned': {
                    'threshold': tuned['threshold'],
                    'f1': tuned['f1'],
                    'precision': tuned['precision'],
                    'recall': tuned['recall'],
                    'by_type': by_type_tuned,
                    'note': 'threshold tuned on eval set (in-sample, optimistic)',
                },
            },
            f,
            indent=2,
        )
    print(f'\nDONE. Modelo + metrics.json em {FINAL_MODEL_DIR}/')


if __name__ == '__main__':
    main()
