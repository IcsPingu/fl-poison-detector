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

import torch
from peft import PeftModel
from safetensors.torch import load_file
from transformers import AutoModelForSequenceClassification

PAD_ID = 0
NUM_BINS = 10000
MAX_LENGTH = 512
MODEL_NAME = 'distilbert-base-uncased'
CANONICAL_WEIGHT_ALIASES = [
    ('conv1.0.weight', 'base.conv1.0.weight'),
    ('conv2.0.weight', 'base.conv2.0.weight'),
    ('fc1.0.weight', 'base.fc1.0.weight'),
    ('fc.weight', 'head.weight'),
]


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


class ClientCheck:
    def __init__(
        self,
        model_dir: str | os.PathLike,
        device: str | None = None,
        use_tuned_threshold: bool = True,
    ) -> None:
        self.model_dir = Path(model_dir)
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)

        required = [
            self.model_dir / 'adapter_config.json',
            self.model_dir / 'adapter_model.safetensors',
            self.model_dir / 'metrics.json',
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(
                f"Detector DistilBERT incompleto em {self.model_dir}. "
                f"Arquivos ausentes: {missing}"
            )

        base = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=2, ignore_mismatched_sizes=True
        )
        self.model = PeftModel.from_pretrained(base, str(self.model_dir))
        self._load_legacy_classifier_head()
        self.model.eval().to(self.device)

        self.threshold = None
        if use_tuned_threshold:
            metrics_path = self.model_dir / 'metrics.json'
            with open(metrics_path) as f:
                m = json.load(f)
            self.threshold = float(m.get('tuned', {}).get('threshold', 0.0))

    def _load_legacy_classifier_head(self) -> None:
        """Compatibilidade com artifacts LoRA antigos que salvaram a head fora de modules_to_save."""
        adapter_path = self.model_dir / 'adapter_model.safetensors'
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
    def classify(self, state_dict: Mapping[str, torch.Tensor]) -> Dict:
        ids_list = preprocess_weights(state_dict, max_length=MAX_LENGTH, num_bins=NUM_BINS)
        input_ids = torch.tensor([ids_list], dtype=torch.long, device=self.device)
        attention_mask = (input_ids != PAD_ID).long()

        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits[0]
        logit_ben = float(logits[0].item())
        logit_mal = float(logits[1].item())
        score = logit_mal - logit_ben

        if self.threshold is not None:
            is_mal = score > self.threshold
        else:
            is_mal = logit_mal > logit_ben

        return {
            'label': int(is_mal),
            'is_malicious': bool(is_mal),
            'logit_ben': logit_ben,
            'logit_mal': logit_mal,
            'score': float(score),
        }

    def is_malicious(self, state_dict: Mapping[str, torch.Tensor]) -> bool:
        return self.classify(state_dict)['is_malicious']

    def filter_indices(self, state_dicts: Sequence[Mapping[str, torch.Tensor]]) -> List[int]:
        """Devolve os indices de state_dicts considerados BENIGNOS (a manter)."""
        return [i for i, sd in enumerate(state_dicts) if not self.is_malicious(sd)]
