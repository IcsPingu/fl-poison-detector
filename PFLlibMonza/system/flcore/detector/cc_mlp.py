"""ClientCheck variante MLP+features (cc=7 no MONZA).

Carrega o detector treinado em detector_mlp.py (artefatos em ARTIFACTS_DIR):
  model.pt        -- state_dict + input_dim + hidden + dropout + feature_names
  scaler.pkl      -- StandardScaler ajustado no treino

Reusa `features.extract_features` (60 features statisticas/espectrais/espaciais
sobre as 4 weight layers da FedAvgCNN). Funciona com qualquer state_dict que
contenha `conv1.0.weight`, `conv2.0.weight`, `fc1.0.weight`, `fc.weight`.

Roda em GPU por default.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import joblib
import numpy as np
import torch
import torch.nn as nn

# Garante import do features.py no mesmo diretorio
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from features import N_FEATURES, extract_features, feature_names  # noqa: E402
from context_features import (  # noqa: E402
    N_CONTEXT_FEATURES,
    extract_context_features,
    feature_names as context_feature_names,
)


def _public_val_dir() -> str:
    public_val_dir = os.environ.get('PUBLIC_VAL_DIR') or ''
    if not public_val_dir or not os.path.isdir(public_val_dir):
        raise FileNotFoundError(
            f"PUBLIC_VAL_DIR invalido: {public_val_dir!r}. "
            "cc=7 requer public_val limpo e separado do test."
        )
    return public_val_dir


class _MLPDetector(nn.Module):
    """Replica EXATA da arquitetura em detector_mlp.py:60. Mesma ordem de layers."""

    def __init__(self, input_dim: int = N_FEATURES, hidden=(128, 64), dropout: float = 0.3):
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


class ClientCheckMLP:
    def __init__(
        self,
        artifacts_dir: str | os.PathLike,
        device: str | None = None,
        threshold_key: str = 'threshold_label_fpr05',
    ) -> None:
        self.artifacts_dir = Path(artifacts_dir)
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)

        ckpt = torch.load(self.artifacts_dir / 'model.pt', map_location=self.device, weights_only=False)
        input_dim = ckpt.get('input_dim', N_FEATURES + N_CONTEXT_FEATURES)
        hidden = tuple(ckpt.get('hidden', (128, 64)))
        dropout = float(ckpt.get('dropout', 0.3))
        self.model = _MLPDetector(input_dim=input_dim, hidden=hidden, dropout=dropout)
        self.model.load_state_dict(ckpt['state_dict'])
        self.model.eval().to(self.device)

        self.scaler = joblib.load(self.artifacts_dir / 'scaler.pkl')
        self.feature_names: List[str] = ckpt.get('feature_names', [])
        current_feature_names = feature_names() + context_feature_names()
        expected_dim = len(current_feature_names)
        scaler_dim = int(getattr(self.scaler, 'n_features_in_', input_dim))
        if int(input_dim) != expected_dim or scaler_dim != expected_dim:
            raise ValueError(
                f"Dimensao de features do MLP incompatível com {self.artifacts_dir}: "
                f"checkpoint={input_dim}, scaler={scaler_dim}, atual={expected_dim}. "
                "Retreine o detector MLP com a versão atual de features/context_features."
            )
        if self.feature_names and self.feature_names != current_feature_names:
            raise ValueError(
                f"Feature names do MLP incompatíveis com {self.artifacts_dir}. "
                "Retreine o detector MLP com a versão atual de features.py."
            )
        self.threshold = 0.0
        self.label_threshold = None
        self.decision_rule = 'binary'
        self.threshold_key = threshold_key
        report_path = self.artifacts_dir / 'report.json'
        if not report_path.exists():
            raise FileNotFoundError(f"report.json ausente em {self.artifacts_dir}")
        with open(report_path) as f:
            report = json.load(f)
        combined = report.get('combined_label_fpr05')
        if combined and 'binary_threshold' in combined and 'label_threshold' in combined:
            self.threshold = float(combined['binary_threshold'])
            self.label_threshold = float(combined['label_threshold'])
            self.threshold_key = 'combined_label_fpr05'
            self.decision_rule = 'binary_or_label'
        elif threshold_key not in report or 'threshold' not in report[threshold_key]:
            available = [k for k, v in report.items() if isinstance(v, dict) and 'threshold' in v]
            raise KeyError(
                f"Threshold '{threshold_key}' ausente em {report_path}. "
                f"Disponiveis: {available}"
            )
        else:
            self.threshold = float(report[threshold_key]['threshold'])

    @torch.no_grad()
    def classify(
        self,
        state_dict: Mapping[str, torch.Tensor],
        global_state_dict: Mapping[str, torch.Tensor] | None = None,
    ) -> Dict:
        feats, _ = extract_features(state_dict, device=self.device)
        ctx_feats, _ = extract_context_features(
            state_dict,
            global_sd=global_state_dict,
            public_val_dir=_public_val_dir(),
            device=self.device,
        )
        feats = np.concatenate([feats, ctx_feats]).astype(np.float32)
        feats = feats.reshape(1, -1).astype(np.float32)
        feats_scaled = self.scaler.transform(feats).astype(np.float32)

        x = torch.from_numpy(feats_scaled).to(self.device)
        logits = self.model(x)[0]
        label_score = float(self.model.label_logits(x)[0].item())
        logit_ben = float(logits[0].item())
        logit_mal = float(logits[1].item())
        score = logit_mal - logit_ben
        binary_hit = score > self.threshold
        label_hit = bool(self.label_threshold is not None and label_score > self.label_threshold)
        is_mal = bool(binary_hit or label_hit)
        return {
            'label': int(is_mal),
            'is_malicious': bool(is_mal),
            'logit_ben': logit_ben,
            'logit_mal': logit_mal,
            'score': score,
            'binary_score': score,
            'label_score': label_score,
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
        if global_state_dicts is None:
            global_state_dicts = [None] * len(state_dicts)
        if len(global_state_dicts) != len(state_dicts):
            raise ValueError("global_state_dicts deve ter o mesmo tamanho de state_dicts.")
        return [
            i for i, (sd, global_sd) in enumerate(zip(state_dicts, global_state_dicts))
            if not self.is_malicious(sd, global_state_dict=global_sd)
        ]
