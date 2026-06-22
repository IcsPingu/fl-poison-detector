"""Benchmark do grid 4x2: 4 variantes de dataset x 2 detectores.

Variantes (USE_PRETRAINED_BASE x HARDEN_ATTACKS):
  1_leakage           : random init        + ataques originais
  2_hard              : random init        + ataques sutis
  3_pretrained_hard   : MNIST-trained base + ataques sutis
  4_pretrained_easy   : MNIST-trained base + ataques originais

Para cada variante:
  - Gera state_dicts em state_dicts_grid/{variante}/
  - Treina e avalia detector.py (DistilBERT+LoRA)
  - Treina e avalia detector_mlp.py (MLP+features)

Tempo total estimado: ~30-40 min na RTX 5060 Ti.
Saida: tabela final em stdout + bench_grid_results.json.
"""
from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from safetensors.torch import save_file


# ============== FedAvgCNN + ataques (copiado do notebook) ==============

class FedAvgCNN(nn.Module):
    def __init__(self, in_features=1, num_classes=10, dim=1024):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features, 32, kernel_size=5, padding=0, stride=1, bias=True),
            nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=(2, 2)),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, padding=0, stride=1, bias=True),
            nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=(2, 2)),
        )
        self.fc1 = nn.Sequential(nn.Linear(dim, 512), nn.ReLU(inplace=True))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.fc(out)
        return out


def model_zeros(model):
    m = copy.deepcopy(model)
    for p in m.parameters():
        p.data.zero_()
    return m


def model_random_uniform(model):
    m = copy.deepcopy(model)
    for p in m.parameters():
        p.data = torch.rand_like(p.data)
    return m


def model_shuffle_full(model):
    m = copy.deepcopy(model)
    for p in m.parameters():
        flat = p.data.view(-1)
        flat[:] = flat[torch.randperm(len(flat))]
    return m


def model_noise(model, snr=10.0):
    m = copy.deepcopy(model)
    for p in m.parameters():
        with torch.no_grad():
            sig = torch.mean(p.data ** 2)
            if sig == 0:
                continue
            pwr = sig / (10 ** (snr / 10))
            p.data.add_(torch.normal(0.0, torch.sqrt(pwr), size=p.shape))
    return m


def model_random_smart(base):
    """Gaussian com sigma da camada original. Preserva mean/std globais."""
    m = copy.deepcopy(base)
    for p in m.parameters():
        sigma = p.data.std().clamp_min(1e-8)
        p.data = torch.randn_like(p.data) * sigma
    return m


def model_shuffle_partial(base, frac):
    """Shuffle de uma fração frac dos pesos por tensor."""
    m = copy.deepcopy(base)
    for p in m.parameters():
        flat = p.data.view(-1)
        n = int(len(flat) * frac)
        if n > 1:
            idx = torch.randperm(len(flat))[:n]
            perm = torch.randperm(n)
            flat[idx] = flat[idx[perm]]
    return m


# ============== Treino do pretrained_base ==============

def train_pretrained_base(epochs=10, batch_size=128, lr=0.01, device=None):
    """Treina FedAvgCNN em MNIST. Retorna modelo na CPU."""
    import torchvision
    from torchvision import transforms
    from torch.utils.data import DataLoader

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = torchvision.datasets.MNIST(
        root='./mnist_data', train=True, download=True, transform=transform
    )
    test_ds = torchvision.datasets.MNIST(
        root='./mnist_data', train=False, download=True, transform=transform
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=2)

    torch.manual_seed(0)
    model = FedAvgCNN().to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    crit = nn.CrossEntropyLoss()
    for ep in range(epochs):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            running += loss.item()
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb).argmax(dim=1)
                correct += (pred == yb).sum().item()
                total += yb.numel()
        print(f'    epoch {ep + 1:2d}/{epochs} loss={running / len(train_loader):.4f} test_acc={correct / total:.4f}')
    return model.cpu()


# ============== Geração de uma variante ==============

ATTACK_TYPES = ['zeros', 'random', 'shuffle', 'noise']


def make_benign(base):
    """Cliente benigno: copia do base + ruído pequeno (proxy de 1 step local)."""
    m = copy.deepcopy(base)
    for p in m.parameters():
        p.data.add_(torch.normal(0.0, 0.01, size=p.shape))
    return m


def make_malicious(base, attack: str, harden: bool):
    """Aplica ataque (versão hard ou easy) em base."""
    if harden:
        # versão sutil
        if attack == 'zeros':
            return model_zeros(base)
        if attack == 'random':
            return model_random_smart(base)
        if attack == 'shuffle':
            frac = float(torch.empty(1).uniform_(0.3, 1.0))
            return model_shuffle_partial(base, frac)
        if attack == 'noise':
            snr = float(torch.empty(1).uniform_(3.0, 15.0))
            return model_noise(base, snr=snr)
    else:
        # versão original
        if attack == 'zeros':
            return model_zeros(base)
        if attack == 'random':
            return model_random_uniform(base)
        if attack == 'shuffle':
            return model_shuffle_full(base)
        if attack == 'noise':
            return model_noise(base, snr=5.0)
    raise ValueError(attack)


def generate_variant(name: str, use_pretrained: bool, harden: bool,
                     dst_dir: Path, pretrained_base: nn.Module,
                     n_samples: int = 1000):
    """Gera state_dicts de uma variante (n benigns + n maliciosos com ataques alternados)."""
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    dst_dir.mkdir(parents=True)

    # Para a variante leakage/easy: usar UM base (random init ou pretrained) compartilhado.
    if use_pretrained:
        shared_base = pretrained_base
    else:
        torch.manual_seed(42)
        shared_base = FedAvgCNN()

    saved = 0
    for i in range(n_samples):
        torch.manual_seed(1000 + i)
        benign = make_benign(shared_base)
        save_file(benign.state_dict(), str(dst_dir / f'benign_{i:04d}.safetensors'))
        with open(dst_dir / f'benign_{i:04d}.json', 'w') as f:
            json.dump({'label': 0, 'type': 'benign'}, f)

        torch.manual_seed(2000 + i)
        attack = ATTACK_TYPES[i % len(ATTACK_TYPES)]
        if harden:
            # cada malicioso parte de um base "fresh" (anti-leakage)
            base_for_attack = make_benign(shared_base)
        else:
            # base compartilhado = leakage
            base_for_attack = shared_base
        mal = make_malicious(base_for_attack, attack, harden)
        save_file(mal.state_dict(), str(dst_dir / f'malicious_{attack}_{i:04d}.safetensors'))
        with open(dst_dir / f'malicious_{attack}_{i:04d}.json', 'w') as f:
            json.dump({'label': 1, 'type': f'malicious_{attack}'}, f)

        saved += 2

    label = ('Pretrained' if use_pretrained else 'RandomInit') + (' + Hard' if harden else ' + Easy')
    print(f'  [{name}] {saved} amostras em {dst_dir}/  ({label})')


# ============== Runner: subprocess pros detectores ==============

SRC_DIR = Path(__file__).resolve().parent  # src/, onde detector.py e detector_mlp.py vivem


def run_detector(name: str, script: str, ds_dir: Path, output_dir: Path,
                 results_file: str) -> Dict:
    """Roda script via subprocess com env. Stream em tempo real (stdout +
    log file). Retorna metricas do JSON salvo."""
    output_dir.mkdir(parents=True, exist_ok=True)
    script_path = SRC_DIR / script
    env = os.environ.copy()
    env['STATE_DICTS_DIR'] = str(ds_dir)
    env['PYTHONUNBUFFERED'] = '1'  # forca prints imediatos no subprocess
    if 'mlp' in script:
        env['ARTIFACTS_DIR'] = str(output_dir)
        tag = 'MLP'
    else:
        env['FINAL_MODEL_DIR'] = str(output_dir)
        env['RUN_DIR'] = str(output_dir / 'hf_runs')
        tag = 'DB '
    log_file = output_dir / 'log.txt'
    prefix = f'      [{name:<19s} {tag}] '
    print(f'  --> {script} log={log_file}')
    t0 = time.time()
    with open(log_file, 'w') as logf:
        proc = subprocess.Popen(
            [sys.executable, '-u', str(script_path)], env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            logf.write(line)
            logf.flush()
            sys.stdout.write(prefix + line)
            sys.stdout.flush()
        proc.wait()
    dt = time.time() - t0
    print(f'  --> {script} done in {dt:.1f}s (returncode={proc.returncode})')

    metrics_path = output_dir / results_file
    if not metrics_path.exists():
        return {'error': f'metrics file ausente: {metrics_path}', 'returncode': proc.returncode}
    with open(metrics_path) as f:
        return json.load(f)


# ============== Orquestrador ==============

def main():
    grid_root = Path('state_dicts_grid')
    output_root = Path('detector_grid_runs')
    grid_root.mkdir(exist_ok=True)
    output_root.mkdir(exist_ok=True)

    # Args opcionais via env
    n_samples = int(os.environ.get('GRID_N_SAMPLES_PER_CLASS', '1000'))
    skip_distilbert = os.environ.get('GRID_SKIP_DISTILBERT', '0') == '1'

    print(f'==== bench_grid: n_samples={n_samples} skip_distilbert={skip_distilbert} ====')

    # 1) Treina pretrained_base (uma vez)
    print('\n[1/3] Treinando pretrained_base em MNIST...')
    pretrained_base = train_pretrained_base(epochs=10)
    print('pretrained_base pronto.\n')

    # 2) Gera os 4 datasets
    variants = [
        ('1_leakage', False, False),
        ('2_hard', False, True),
        ('3_pretrained_hard', True, True),
        ('4_pretrained_easy', True, False),
    ]

    print('[2/3] Gerando 4 variantes...')
    for name, use_pre, harden in variants:
        generate_variant(name, use_pre, harden, grid_root / name, pretrained_base, n_samples)

    # 3) Roda detectores para cada variante
    print('\n[3/3] Treinando detectores em cada variante...')
    results = {}
    for name, _, _ in variants:
        ds = grid_root / name
        results[name] = {}

        if not skip_distilbert:
            db_out = output_root / name / 'distilbert'
            results[name]['distilbert'] = run_detector(
                name, 'detector.py', ds, db_out, 'metrics.json'
            )
        else:
            results[name]['distilbert'] = {'skipped': True}

        mlp_out = output_root / name / 'mlp'
        results[name]['mlp'] = run_detector(
            name, 'detector_mlp.py', ds, mlp_out, 'report.json'
        )

    # ============== Tabela final ==============
    print('\n\n==================== TABELA FINAL ====================\n')
    header = f'{"Variante":<22s}  {"DB F1 (default)":>18s}  {"DB F1 (tunado)":>18s}  {"MLP F1":>10s}'
    print(header)
    print('-' * len(header))
    for name, _, _ in variants:
        r = results[name]
        if 'error' in r.get('distilbert', {}) or 'skipped' in r.get('distilbert', {}):
            db_default = '—'
            db_tuned = '—'
        else:
            db_default = f"{r['distilbert']['default_argmax']['f1']:.4f}"
            db_tuned = f"{r['distilbert']['tuned']['f1']:.4f}"
        if 'error' in r.get('mlp', {}):
            mlp_f1 = '—'
        else:
            mlp_f1 = f"{r['mlp']['metrics']['f1']:.4f}"
        print(f'{name:<22s}  {db_default:>18s}  {db_tuned:>18s}  {mlp_f1:>10s}')

    print('\n\n==================== BREAKDOWN POR ATAQUE (MLP) ====================\n')
    print(f'{"Variante":<22s}  {"FPR":>8s}  {"zeros":>8s}  {"random":>8s}  {"shuffle":>8s}  {"noise":>8s}')
    print('-' * 80)
    for name, _, _ in variants:
        r = results[name]['mlp']
        if 'error' in r:
            continue
        bt = r.get('by_type', {})
        def pct(key, kind='recall'):
            b = bt.get(key, {'total': 0, 'predicted_malicious': 0})
            if b['total'] == 0:
                return '—'
            return f"{b['predicted_malicious'] / b['total']:.0%}"
        print(
            f'{name:<22s}  {pct("benign", "fpr"):>8s}  {pct("malicious_zeros"):>8s}  '
            f'{pct("malicious_random"):>8s}  {pct("malicious_shuffle"):>8s}  '
            f'{pct("malicious_noise"):>8s}'
        )

    # Salva tudo
    with open('bench_grid_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print('\nResultados salvos em bench_grid_results.json')


if __name__ == '__main__':
    main()
