"""Label-flip checker based on clean holdout behavior and final-layer direction.

Fingerprint detectors catch obvious parameter attacks, but label flip can look
statistically benign. This checker uses a clean public holdout to create a
small trusted root update, then flags client updates whose final classifier
direction disagrees with that root and whose class-wise validation loss is a
round outlier.
"""
from __future__ import annotations

import copy
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn


class LabelFlipCheck:
    def __init__(
        self,
        val_loader,
        device: str | torch.device,
        root_lr: float = 0.01,
        root_steps: int = 5,
        min_loss_delta: float = 0.02,
        mad_k: float = 3.0,
        max_final_cos: float = 0.0,
    ) -> None:
        self.val_loader = val_loader
        self.device = torch.device(device)
        self.root_lr = float(root_lr)
        self.root_steps = int(root_steps)
        self.min_loss_delta = float(min_loss_delta)
        self.mad_k = float(mad_k)
        self.max_final_cos = float(max_final_cos)
        self.loss_sum = nn.CrossEntropyLoss(reduction='sum')
        self.loss_none = nn.CrossEntropyLoss(reduction='none')

    def _to_device(self, x, y):
        if type(x) == type([]):
            x[0] = x[0].to(self.device)
        else:
            x = x.to(self.device)
        return x, y.to(self.device)

    @staticmethod
    def _final_weight_key(state_dict) -> str:
        for key in ('head.weight', 'fc.weight', 'base.fc.weight'):
            if key in state_dict:
                return key
        candidates = [k for k in state_dict if k.endswith('fc.weight') or k.endswith('head.weight')]
        if not candidates:
            raise KeyError('Nenhuma camada final encontrada no state_dict.')
        return candidates[-1]

    @torch.no_grad()
    def _class_losses(self, model: torch.nn.Module) -> Dict[int, float]:
        was_training = model.training
        model.eval()
        totals: Dict[int, float] = {}
        counts: Dict[int, int] = {}
        for x, y in self.val_loader:
            x, y = self._to_device(x, y)
            out = model(x)
            losses = self.loss_none(out, y)
            for cls in torch.unique(y):
                mask = y == cls
                c = int(cls.item())
                totals[c] = totals.get(c, 0.0) + float(losses[mask].sum().item())
                counts[c] = counts.get(c, 0) + int(mask.sum().item())
        if was_training:
            model.train()
        return {c: totals[c] / max(counts[c], 1) for c in totals}

    def _root_model(self, global_model: torch.nn.Module) -> torch.nn.Module:
        model = copy.deepcopy(global_model).to(self.device)
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.root_lr)
        steps = 0
        for x, y in self.val_loader:
            x, y = self._to_device(x, y)
            optimizer.zero_grad()
            loss = self.loss_sum(model(x), y) / max(int(y.numel()), 1)
            loss.backward()
            optimizer.step()
            steps += 1
            if steps >= self.root_steps:
                break
        model.eval()
        return model

    @staticmethod
    def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
        av = a.detach().flatten().float()
        bv = b.detach().flatten().float()
        denom = torch.linalg.norm(av) * torch.linalg.norm(bv)
        if float(denom.item()) < 1e-12:
            return 1.0
        return float(torch.dot(av, bv).div(denom).item())

    def score_round(
        self,
        global_model: torch.nn.Module,
        uploaded_models: Iterable[torch.nn.Module],
        uploaded_ids: Iterable[int],
    ) -> Dict[int, Dict[str, float | bool | int]]:
        global_model = global_model.to(self.device)
        root_model = self._root_model(global_model)

        global_sd = global_model.state_dict()
        root_sd = root_model.state_dict()
        final_key = self._final_weight_key(global_sd)
        root_delta = root_sd[final_key].detach() - global_sd[final_key].detach()

        base_class_loss = self._class_losses(global_model)
        rows: List[Tuple[int, float, float, int, float]] = []
        for cid, model in zip(uploaded_ids, uploaded_models):
            model = model.to(self.device)
            sd = model.state_dict()
            client_delta = sd[final_key].detach() - global_sd[final_key].detach()
            final_cos = self._cosine(client_delta, root_delta)
            class_loss = self._class_losses(model)
            deltas = {
                c: class_loss.get(c, base_loss) - base_loss
                for c, base_loss in base_class_loss.items()
            }
            worst_class, worst_delta = max(deltas.items(), key=lambda item: item[1])
            worst_loss = class_loss.get(worst_class, base_class_loss[worst_class])
            rows.append((int(cid), final_cos, float(worst_delta), int(worst_class), float(worst_loss)))

        loss_scores = np.array([r[2] for r in rows], dtype=np.float64)
        if loss_scores.size == 0:
            return {}
        median = float(np.median(loss_scores))
        mad = float(np.median(np.abs(loss_scores - median)))
        outlier_threshold = median + self.mad_k * mad

        out: Dict[int, Dict[str, float | bool | int]] = {}
        for cid, final_cos, worst_delta, worst_class, worst_loss in rows:
            loss_outlier = bool(worst_delta > outlier_threshold and worst_delta > self.min_loss_delta)
            direction_bad = bool(final_cos < self.max_final_cos)
            reject = bool(loss_outlier and direction_bad)
            out[cid] = {
                'reject': reject,
                'loss_outlier': loss_outlier,
                'direction_bad': direction_bad,
                'final_cos': final_cos,
                'worst_class': worst_class,
                'worst_class_loss': worst_loss,
                'worst_class_delta': worst_delta,
                'median_delta': median,
                'mad': mad,
                'outlier_threshold': outlier_threshold,
            }
        return out
