"""ClientCheck (cc): inferencia em producao do detector treinado em detector.py.

Carrega DistilBERT base + adapter LoRA salvo em FINAL_MODEL_DIR e classifica
state_dicts de clientes FL como benign (0) ou malicious (1).

Self-contained: replica preprocess_weights + constantes de detector.py.
Estes sao o CONTRATO de pre-processamento entre treino e inferencia -- se
mudar em detector.py, mudar aqui tambem (e retreinar). Linkadas em:
  detector.py:32-44  (PAD_ID, NUM_BINS, MAX_LENGTH, MODEL_NAME)
  detector.py:54-81  (_normalize_layer + preprocess_weights)

Roda em GPU por default.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import joblib
import numpy as np
import torch
import torch.nn as nn
from peft import PeftModel
from safetensors.torch import load_file
from transformers import AutoModel, AutoModelForSequenceClassification
try:
    from context_features import context_tokens, extract_context_features
except ImportError:
    from .context_features import context_tokens, extract_context_features

PAD_ID = 0
NUM_BINS = 10000
MAX_LENGTH = 512
CONTEXT_TOKEN_COUNT = 128
MODEL_NAME = 'distilbert-base-uncased'
CANONICAL_WEIGHT_ALIASES = [
    ('conv1.0.weight', 'base.conv1.0.weight'),
    ('conv2.0.weight', 'base.conv2.0.weight'),
    ('fc1.0.weight', 'base.fc1.0.weight'),
    ('fc.weight', 'head.weight'),
]


def _public_val_dir() -> str:
    public_val_dir = os.environ.get('PUBLIC_VAL_DIR') or ''
    if not public_val_dir or not os.path.isdir(public_val_dir):
        raise FileNotFoundError(
            f"PUBLIC_VAL_DIR invalido: {public_val_dir!r}. "
            "cc=6 requer public_val limpo e separado do test."
        )
    return public_val_dir


def _normalize_layer(t: torch.Tensor) -> torch.Tensor:
    lo = torch.quantile(t, 0.05)
    hi = torch.quantile(t, 0.95)
    return ((t - lo) / (hi - lo + 1e-8)).clamp(0.0, 1.0)


def _ordered_weight_items(state_dict: Mapping[str, torch.Tensor]):
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


def preprocess_weights(
    state_dict: Mapping[str, torch.Tensor],
    max_length: int = MAX_LENGTH,
    num_bins: int = NUM_BINS,
) -> List[int]:
    parts = [
        _normalize_layer(v.detach().to(dtype=torch.float32).flatten())
        for _, v in _ordered_weight_items(state_dict)
    ]
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


def preprocess_with_context(
    state_dict: Mapping[str, torch.Tensor],
    global_state_dict: Mapping[str, torch.Tensor] | None = None,
    max_length: int = MAX_LENGTH,
    num_bins: int = NUM_BINS,
) -> List[int]:
    ctx_feats, _ = extract_context_features(
        state_dict,
        global_sd=global_state_dict,
        public_val_dir=_public_val_dir(),
    )
    prefix = context_tokens(ctx_feats, token_count=CONTEXT_TOKEN_COUNT, num_bins=num_bins)
    body = preprocess_weights(
        state_dict,
        max_length=max_length - CONTEXT_TOKEN_COUNT,
        num_bins=num_bins,
    )
    return prefix + body


class ClientCheck:
    def __init__(
        self,
        model_dir: str | os.PathLike,
        device: str | None = None,
        use_tuned_threshold: bool = True,
        threshold_key: str = 'threshold_label_fpr05',
    ) -> None:
        self.model_dir = Path(model_dir)
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)

        self.is_hybrid = (self.model_dir / 'hybrid_head.pt').exists()
        required = [
            self.model_dir / 'adapter_config.json',
            self.model_dir / 'adapter_model.safetensors',
            self.model_dir / 'metrics.json',
        ]
        if self.is_hybrid:
            required.extend([
                self.model_dir / 'hybrid_head.pt',
                self.model_dir / 'context_scaler.pkl',
            ])
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(
                f"Detector DistilBERT incompleto em {self.model_dir}. "
                f"Arquivos ausentes: {missing}"
            )

        if self.is_hybrid:
            base = AutoModel.from_pretrained(MODEL_NAME)
            self.bert = PeftModel.from_pretrained(base, str(self.model_dir))
            head = torch.load(self.model_dir / 'hybrid_head.pt', map_location=self.device, weights_only=False)
            context_dim = int(head['context_dim'])
            tabular_dim = int(head['tabular_dim'])
            hidden = int(base.config.hidden_size)
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
            self.label_classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(hidden + tabular_dim, 1),
            )
            self.fusion_gate = None
            if 'fusion_gate_state_dict' in head:
                self.fusion_gate = nn.Sequential(
                    nn.Linear(hidden + tabular_dim, hidden + tabular_dim),
                    nn.Sigmoid(),
                )
                self.fusion_gate.load_state_dict(head['fusion_gate_state_dict'])
            self.tabular.load_state_dict(head['tabular_state_dict'])
            self.classifier.load_state_dict(head['classifier_state_dict'])
            if 'label_classifier_state_dict' in head:
                self.label_classifier.load_state_dict(head['label_classifier_state_dict'])
            else:
                self.label_classifier = None
            self.scaler = joblib.load(self.model_dir / 'context_scaler.pkl')
            self.bert.eval().to(self.device)
            self.tabular.eval().to(self.device)
            self.classifier.eval().to(self.device)
            if self.label_classifier is not None:
                self.label_classifier.eval().to(self.device)
            if self.fusion_gate is not None:
                self.fusion_gate.eval().to(self.device)
        else:
            base = AutoModelForSequenceClassification.from_pretrained(
                MODEL_NAME, num_labels=2, ignore_mismatched_sizes=True
            )
            self.model = PeftModel.from_pretrained(base, str(self.model_dir))
            self._load_legacy_classifier_head()
            self.model.eval().to(self.device)

        self.threshold = None
        self.label_threshold = None
        self.decision_rule = 'binary'
        if use_tuned_threshold:
            metrics_path = self.model_dir / 'metrics.json'
            if not metrics_path.exists():
                raise FileNotFoundError(f"metrics.json ausente em {self.model_dir}")
            with open(metrics_path) as f:
                m = json.load(f)
            combined = m.get('combined_label_fpr05')
            if combined and 'binary_threshold' in combined and 'label_threshold' in combined:
                self.threshold = float(combined['binary_threshold'])
                self.label_threshold = float(combined['label_threshold'])
                self.threshold_key = 'combined_label_fpr05'
                self.decision_rule = 'binary_or_label'
            elif threshold_key not in m or 'threshold' not in m[threshold_key]:
                available = [k for k, v in m.items() if isinstance(v, dict) and 'threshold' in v]
                raise KeyError(
                    f"Threshold '{threshold_key}' ausente em {metrics_path}. "
                    f"Disponiveis: {available}"
                )
            else:
                self.threshold = float(m[threshold_key]['threshold'])
                self.threshold_key = threshold_key
        else:
            self.threshold_key = threshold_key

    def _load_legacy_classifier_head(self) -> None:
        """Compatibilidade com artifacts LoRA antigos que salvaram a head fora de modules_to_save."""
        adapter_path = self.model_dir / 'adapter_model.safetensors'
        if not adapter_path.exists():
            return
        tensors = load_file(str(adapter_path))
        legacy_map = {
            'base_model.model.pre_classifier.weight': self.model.base_model.model.pre_classifier.weight,
            'base_model.model.pre_classifier.bias': self.model.base_model.model.pre_classifier.bias,
            'base_model.model.classifier.weight': self.model.base_model.model.classifier.weight,
            'base_model.model.classifier.bias': self.model.base_model.model.classifier.bias,
        }
        loaded = []
        with torch.no_grad():
            for key, param in legacy_map.items():
                tensor = tensors.get(key)
                if tensor is None:
                    continue
                if tuple(tensor.shape) != tuple(param.shape):
                    raise ValueError(
                        f"Head DistilBERT incompatível em {adapter_path}: "
                        f"{key} tem shape {tuple(tensor.shape)}, esperado {tuple(param.shape)}"
                    )
                param.copy_(tensor.to(device=param.device, dtype=param.dtype))
                loaded.append(key)
        if loaded:
            print(f"[cc] Head DistilBERT carregada do adapter legado: {len(loaded)}/4 tensores.")

    @torch.no_grad()
    def classify(
        self,
        state_dict: Mapping[str, torch.Tensor],
        global_state_dict: Mapping[str, torch.Tensor] | None = None,
    ) -> Dict:
        if self.is_hybrid:
            ids_list = preprocess_weights(state_dict, max_length=MAX_LENGTH, num_bins=NUM_BINS)
        else:
            ids_list = preprocess_with_context(
                state_dict,
                global_state_dict=global_state_dict,
                max_length=MAX_LENGTH,
                num_bins=NUM_BINS,
            )
        input_ids = torch.tensor([ids_list], dtype=torch.long, device=self.device)
        attention_mask = (input_ids != PAD_ID).long()

        if self.is_hybrid:
            ctx_feats, _ = extract_context_features(
                state_dict,
                global_sd=global_state_dict,
                public_val_dir=_public_val_dir(),
                device=self.device,
            )
            ctx_scaled = self.scaler.transform(ctx_feats.reshape(1, -1).astype(np.float32)).astype(np.float32)
            ctx_tensor = torch.from_numpy(ctx_scaled).to(self.device)
            out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            pooled = out.last_hidden_state[:, 0]
            tab = self.tabular(ctx_tensor.to(dtype=pooled.dtype))
            fused = torch.cat([pooled, tab], dim=1)
            if self.fusion_gate is not None:
                fused = fused * self.fusion_gate(fused)
            logits = self.classifier(fused)[0]
            label_score = (
                float(self.label_classifier(fused).squeeze(-1)[0].item())
                if self.label_classifier is not None else float('-inf')
            )
        else:
            out = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = out.logits[0]
            label_score = float('-inf')
        logit_ben = float(logits[0].item())
        logit_mal = float(logits[1].item())
        score = logit_mal - logit_ben

        if self.threshold is not None:
            binary_hit = score > self.threshold
        else:
            binary_hit = logit_mal > logit_ben
        label_hit = bool(self.label_threshold is not None and label_score > self.label_threshold)
        is_mal = bool(binary_hit or label_hit)

        return {
            'label': int(is_mal),
            'is_malicious': bool(is_mal),
            'logit_ben': logit_ben,
            'logit_mal': logit_mal,
            'score': float(score),
            'binary_score': float(score),
            'label_score': float(label_score),
            'threshold': self.threshold,
            'binary_threshold': self.threshold,
            'label_threshold': self.label_threshold,
            'threshold_key': self.threshold_key,
            'binary_hit': bool(binary_hit),
            'label_hit': bool(label_hit),
            'decision_rule': self.decision_rule,
        }

    def is_malicious(
        self,
        state_dict: Mapping[str, torch.Tensor],
        global_state_dict: Mapping[str, torch.Tensor] | None = None,
    ) -> bool:
        return self.classify(state_dict, global_state_dict=global_state_dict)['is_malicious']

    def filter_indices(
        self,
        state_dicts: Sequence[Mapping[str, torch.Tensor]],
        global_state_dicts: Sequence[Mapping[str, torch.Tensor]] | None = None,
    ) -> List[int]:
        """Devolve os indices de state_dicts considerados BENIGNOS (a manter)."""
        if global_state_dicts is None:
            global_state_dicts = [None] * len(state_dicts)
        if len(global_state_dicts) != len(state_dicts):
            raise ValueError("global_state_dicts deve ter o mesmo tamanho de state_dicts.")
        return [
            i for i, (sd, global_sd) in enumerate(zip(state_dicts, global_state_dicts))
            if not self.is_malicious(sd, global_state_dict=global_sd)
        ]
