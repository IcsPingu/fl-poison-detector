"""Shared split and diagnostic helpers for detector training."""
from __future__ import annotations

import csv
from collections import Counter
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


def _client_labels(entries: Sequence[Dict], clients: Sequence[int]) -> List[str]:
    client_labels = []
    for cid in clients:
        types = [e.get('type', 'unknown') for e in entries if int(e.get('client_id', -1)) == cid]
        if 'malicious_label' in types:
            client_labels.append('has_label')
        elif any(str(t).startswith('malicious') for t in types):
            client_labels.append('has_malicious')
        else:
            client_labels.append('benign_only')
    return client_labels


def _split_clients(
    entries: Sequence[Dict],
    clients: Sequence[int],
    holdout_size: float,
    seed: int,
) -> Tuple[set[int], set[int]]:
    client_labels = _client_labels(entries, clients)
    client_arr = np.asarray(clients)
    if len(set(client_labels)) > 1 and min(Counter(client_labels).values()) >= 2:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=holdout_size, random_state=seed)
        keep_pos, holdout_pos = next(splitter.split(np.zeros(len(clients)), client_labels))
        return set(client_arr[keep_pos].tolist()), set(client_arr[holdout_pos].tolist())

    rng = np.random.default_rng(seed)
    shuffled = client_arr.copy()
    rng.shuffle(shuffled)
    n_holdout = max(1, int(round(len(shuffled) * holdout_size)))
    return set(shuffled[n_holdout:].tolist()), set(shuffled[:n_holdout].tolist())


def split_by_client_then_round(
    entries: Sequence[Dict],
    test_size: float = 0.2,
    calib_size: float = 0.2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return train/calib/test indices with disjoint client groups.

    Test and calibration are grouped by client_id. This makes threshold
    calibration depend on clients unseen during model fitting, which is closer
    to the deployment case than calibrating on later rounds of train clients.
    """
    if calib_size + test_size >= 1.0:
        raise ValueError('calib_size + test_size deve ser menor que 1.0.')

    clients = sorted({int(e.get('client_id', -1)) for e in entries})
    train_calib_clients, test_clients = _split_clients(entries, clients, test_size, seed)
    calib_fraction = calib_size / (1.0 - test_size)
    train_clients, calib_clients = _split_clients(
        entries,
        sorted(train_calib_clients),
        calib_fraction,
        seed + 1,
    )

    train_idx = np.asarray([
        i for i, e in enumerate(entries)
        if int(e.get('client_id', -1)) in train_clients
    ], dtype=np.int64)
    calib_idx = np.asarray([
        i for i, e in enumerate(entries)
        if int(e.get('client_id', -1)) in calib_clients
    ], dtype=np.int64)
    test_idx = np.asarray([
        i for i, e in enumerate(entries)
        if int(e.get('client_id', -1)) in test_clients
    ], dtype=np.int64)

    if len(train_idx) == 0 or len(calib_idx) == 0 or len(test_idx) == 0:
        raise ValueError('Split gerou particao vazia; verifique metadata round/client_id.')
    return train_idx, calib_idx, test_idx


def split_train_dev_calib_test_by_client(
    entries: Sequence[Dict],
    dev_size: float = 0.2,
    calib_size: float = 0.2,
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return train/dev/calib/test indices with disjoint client groups.

    Dev is used for checkpoint/early-stopping, calibration only for thresholds,
    and test only for final reporting.
    """
    if dev_size + calib_size + test_size >= 1.0:
        raise ValueError('dev_size + calib_size + test_size deve ser menor que 1.0.')

    clients = sorted({int(e.get('client_id', -1)) for e in entries})
    remaining_clients, test_clients = _split_clients(entries, clients, test_size, seed)
    calib_fraction = calib_size / (1.0 - test_size)
    remaining_clients, calib_clients = _split_clients(
        entries,
        sorted(remaining_clients),
        calib_fraction,
        seed + 1,
    )
    dev_fraction = dev_size / (1.0 - test_size - calib_size)
    train_clients, dev_clients = _split_clients(
        entries,
        sorted(remaining_clients),
        dev_fraction,
        seed + 2,
    )

    def indices_for(client_set: set[int]) -> np.ndarray:
        return np.asarray([
            i for i, e in enumerate(entries)
            if int(e.get('client_id', -1)) in client_set
        ], dtype=np.int64)

    train_idx = indices_for(train_clients)
    dev_idx = indices_for(dev_clients)
    calib_idx = indices_for(calib_clients)
    test_idx = indices_for(test_clients)

    if min(len(train_idx), len(dev_idx), len(calib_idx), len(test_idx)) == 0:
        raise ValueError('Split gerou particao vazia; verifique metadata round/client_id.')
    return train_idx, dev_idx, calib_idx, test_idx


def split_by_client_then_round_legacy(
    entries: Sequence[Dict],
    test_size: float = 0.2,
    calib_size: float = 0.2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Legacy split: unseen clients in test, later train-client rounds in calibration."""
    clients = sorted({int(e.get('client_id', -1)) for e in entries})
    keep_clients, test_clients = _split_clients(entries, clients, test_size, seed)

    keep_idx = np.asarray([
        i for i, e in enumerate(entries)
        if int(e.get('client_id', -1)) in keep_clients
    ], dtype=np.int64)
    test_idx = np.asarray([
        i for i, e in enumerate(entries)
        if int(e.get('client_id', -1)) in test_clients
    ], dtype=np.int64)

    rounds = sorted({int(entries[i].get('round', 0)) for i in keep_idx})
    n_calib_rounds = max(1, int(round(len(rounds) * calib_size)))
    calib_rounds = set(rounds[-n_calib_rounds:])
    calib_idx = np.asarray([
        int(i) for i in keep_idx
        if int(entries[int(i)].get('round', 0)) in calib_rounds
    ], dtype=np.int64)
    train_idx = np.asarray([
        int(i) for i in keep_idx
        if int(entries[int(i)].get('round', 0)) not in calib_rounds
    ], dtype=np.int64)

    if len(train_idx) == 0 or len(calib_idx) == 0 or len(test_idx) == 0:
        raise ValueError('Split gerou particao vazia; verifique metadata round/client_id.')
    return train_idx, calib_idx, test_idx


def split_summary(entries: Sequence[Dict], indices: Iterable[int]) -> Dict:
    idx = [int(i) for i in indices]
    types = [entries[i].get('type', 'unknown') for i in idx]
    clients = {int(entries[i].get('client_id', -1)) for i in idx}
    rounds = [int(entries[i].get('round', 0)) for i in idx]
    return {
        'n': len(idx),
        'clients': len(clients),
        'round_min': min(rounds) if rounds else None,
        'round_max': max(rounds) if rounds else None,
        'by_type': dict(Counter(types)),
    }


def write_score_diagnostics(
    path: str,
    entries: Sequence[Dict],
    indices: Sequence[int],
    split_name: str,
    logits: np.ndarray,
    threshold: float | None = None,
    label_scores: np.ndarray | None = None,
    label_threshold: float | None = None,
    combined_preds: np.ndarray | None = None,
) -> None:
    scores = logits[:, 1] - logits[:, 0]
    preds = (scores > threshold).astype(int) if threshold is not None else np.argmax(logits, axis=1)
    if combined_preds is not None:
        preds = np.asarray(combined_preds).astype(int)
    label_scores_np = None if label_scores is None else np.asarray(label_scores, dtype=np.float32)
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'split', 'sample_id', 'round', 'client_id', 'type', 'label',
                'score', 'logit_benign', 'logit_malicious', 'threshold', 'pred',
                'label_score', 'label_threshold',
            ],
        )
        if f.tell() == 0:
            writer.writeheader()
        for row_pos, entry_idx in enumerate(indices):
            e = entries[int(entry_idx)]
            writer.writerow({
                'split': split_name,
                'sample_id': e.get('sample_id', ''),
                'round': e.get('round', ''),
                'client_id': e.get('client_id', ''),
                'type': e.get('type', 'unknown'),
                'label': int(e.get('labels', e.get('label', 0))),
                'score': float(scores[row_pos]),
                'logit_benign': float(logits[row_pos, 0]),
                'logit_malicious': float(logits[row_pos, 1]),
                'threshold': '' if threshold is None else float(threshold),
                'pred': int(preds[row_pos]),
                'label_score': '' if label_scores_np is None else float(label_scores_np[row_pos]),
                'label_threshold': '' if label_threshold is None else float(label_threshold),
            })
