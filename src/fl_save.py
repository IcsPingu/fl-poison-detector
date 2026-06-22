"""Salvamento de state_dicts FL no formato consumido por detector.py / detector_mlp.py.

Schema (espelha bench_grid.py:221-235):
  {out_dir}/{sample_id}.safetensors  -- pesos do cliente
  {out_dir}/{sample_id}.json         -- metadata do cliente
  {out_dir}/global_rXXX.safetensors  -- modelo global antes do round

label: 0=benign, 1=malicious
type:  'benign' | 'malicious_zeros' | 'malicious_random' | 'malicious_shuffle' | 'malicious_label' | 'malicious_noise'

Tensores chegam em GPU (FL roda em CUDA); .detach().cpu() so na hora de serializar.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Iterable, Mapping

import torch
from safetensors.torch import save_file


def save_client_update(
    state_dict: Mapping[str, torch.Tensor],
    label: int,
    type_: str,
    out_dir: str | os.PathLike,
    sample_id: str,
    metadata: Mapping[str, object] | None = None,
) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cpu_sd = {k: v.detach().cpu().contiguous() for k, v in state_dict.items()}

    safe_path = out / f'{sample_id}.safetensors'
    json_path = out / f'{sample_id}.json'

    tmp_safe_path = safe_path.with_suffix(safe_path.suffix + '.tmp')
    tmp_json_path = json_path.with_suffix(json_path.suffix + '.tmp')

    try:
        save_file(cpu_sd, str(tmp_safe_path))
        meta = {'label': int(label), 'type': str(type_)}
        if metadata:
            meta.update(dict(metadata))
        with open(tmp_json_path, 'w') as f:
            json.dump(meta, f)
        os.replace(tmp_safe_path, safe_path)
        os.replace(tmp_json_path, json_path)
    except Exception:
        tmp_safe_path.unlink(missing_ok=True)
        tmp_json_path.unlink(missing_ok=True)
        raise

    return safe_path


def save_global_state(
    state_dict: Mapping[str, torch.Tensor],
    out_dir: str | os.PathLike,
    round_idx: int,
) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    safe_path = out / f'global_r{int(round_idx):03d}.safetensors'
    if safe_path.exists():
        return safe_path
    tmp_safe_path = safe_path.with_suffix(safe_path.suffix + '.tmp')
    cpu_sd = {k: v.detach().cpu().contiguous() for k, v in state_dict.items()}
    try:
        save_file(cpu_sd, str(tmp_safe_path))
        os.replace(tmp_safe_path, safe_path)
    except Exception:
        tmp_safe_path.unlink(missing_ok=True)
        raise
    return safe_path


def save_round_dump(
    uploaded_models: Iterable,
    uploaded_ids: Iterable[int],
    clients_by_id: Dict[int, object],
    index_malicious,
    round_idx: int,
    out_dir: str | os.PathLike,
    global_state_dict: Mapping[str, torch.Tensor] | None = None,
) -> int:
    """Dumpa todos os updates do round atual. Devolve quantos arquivos foram salvos.

    Para cada (model, client_id):
      - is_mal: prefere `client.is_malicious` (flag dinamica do round) ; fallback = client_id em index_malicious
      - type_:  prefere `client.last_attack_type`; fallback 'benign' / 'malicious_unknown'
    """
    index_malicious_set = set(int(x) for x in index_malicious) if index_malicious is not None else set()
    global_path = None
    if global_state_dict is not None:
        global_path = save_global_state(global_state_dict, out_dir, round_idx).name
    n_saved = 0
    for model, cid in zip(uploaded_models, uploaded_ids):
        client = clients_by_id.get(int(cid))
        is_mal_flag = getattr(client, 'is_malicious', None)
        if is_mal_flag is None:
            is_mal = int(cid) in index_malicious_set
        else:
            is_mal = bool(is_mal_flag)

        type_ = getattr(client, 'last_attack_type', None)
        if not type_:
            type_ = 'malicious_unknown' if is_mal else 'benign'

        sample_id = f'r{int(round_idx):03d}_c{int(cid):03d}_{type_}'
        save_client_update(
            model.state_dict(),
            int(is_mal),
            type_,
            out_dir,
            sample_id,
            metadata={
                'round': int(round_idx),
                'client_id': int(cid),
                'global_state': global_path,
            },
        )
        n_saved += 1
    return n_saved
