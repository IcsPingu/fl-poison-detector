# HOWTO — pipeline completo (MONZA → dataset → detector → defesa)

Este repo integra **PFLlibMonza** (FL real, em `PFLlibMonza/`) com **jpt** (detectores DistilBERT+MLP, em `src/`). O fluxo é:

1. **MONZA** roda FL com clientes maliciosos e dumpa state_dicts → dataset
2. **jpt** treina detectores (DistilBERT+LoRA e MLP+features) sobre o dataset
3. **MONZA** compara os baselines (`cc=2` cluster cosseno, `cc=3` cosseno+score) com os dois detectores finais (`cc=6` DistilBERT, `cc=7` MLP)

Resultado experimental fechado em [`MONZA_RESULTS.md`](MONZA_RESULTS.md). Análise visual em [`notebook_monza_analysis.ipynb`](notebook_monza_analysis.ipynb).

---

## Setup (1× só)

### 1. Clonar e instalar deps

```bash
git clone https://github.com/81wallace18/fl-poison-detector.git jpt
cd jpt

# Ambiente unico na raiz para jpt + MONZA
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

> **GPU**: o `requirements.txt` unico ja aponta para o indice CUDA 13.0 do PyTorch. Pipeline testado em RTX 5060 Ti, sm_120. Se sua GPU for mais antiga, ajuste o indice/versao do PyTorch e confira que `torch.cuda.is_available()` retorna True.

Todos os comandos abaixo usam essa mesma `.venv` da raiz. Quando o comando roda dentro de `PFLlibMonza/system`, o caminho correto é `../../.venv/bin/python`.

### 2. Gerar particionamento MNIST (100 clientes Dirichlet non-IID)

```bash
cd PFLlibMonza/dataset
../../.venv/bin/python generate_MNIST.py noniid - dir
ls MNIST/train/ | wc -l   # esperado: 100
cd ../..
.venv/bin/python scripts/create_label_flip_train_mal.py --dataset-dir PFLlibMonza/dataset/MNIST --num-classes 10
cd PFLlibMonza/dataset
ls MNIST/train_mal/ | wc -l   # esperado: 100 para ataque malicious_label real
cd ../..
```

Scripts no `PFLlibMonza/dataset/` (`generate_MNIST.py`, `generate_Cifar10.py`, etc) hardcodam `num_clients` no topo do arquivo. Para outras configurações, edite a constante `num_clients`.

> **Importante para label flip**: `PFLlibMonza/system/utils/data_utils.py` agora lê clientes maliciosos de `MNIST/train_mal/`. Se essa pasta não existir, o run falha de propósito. O script `scripts/create_label_flip_train_mal.py` cria o `train_mal` com os mesmos `x` de treino e labels invertidos de forma determinística (`y_flip = num_classes - 1 - y`, no MNIST: 0↔9, 1↔8, ...).

### 3. Verificar GPU e imports

```bash
cd PFLlibMonza/system
../../.venv/bin/python -c "
import torch
from flcore.detector.cc import ClientCheck
from flcore.detector.cc_mlp import ClientCheckMLP
from flcore.detector import fl_save
print('torch:', torch.__version__, '| cuda:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')
print('imports OK')
"
cd ../..
```

Esperado: `cuda: True | GPU: NVIDIA GeForce RTX 5060 Ti` + `imports OK`.

---

## Comando unico para rodar tudo do zero

O script abaixo limpa artefatos gerados, regenera MNIST + `train_mal/`, faz warm-up limpo, dumpa os `state_dicts` com o modelo global anterior de cada round, treina DistilBERT e MLP, roda `cc=2`, `cc=3`, `cc=6` e `cc=7`, executa o notebook e grava os graficos.

```bash
./scripts/run_full_monza.sh
```

Para rodar em background com log automatico:

```bash
./scripts/run_full_monza.sh --background
tail -f rerun_full_*.log
```

Variaveis uteis:

```bash
GLOBAL_ROUNDS=50 DEVICE_ID=0 ./scripts/run_full_monza.sh --background
```

Defaults relevantes para `malicious_label`:

| Variável | Default | Efeito |
|---|---:|---|
| `ROUND_INIT_ATK` | `5` | rounds iniciais sem ataque para pré-treino/warm-up limpo |
| `DUMP_START_ROUND` | `ROUND_INIT_ATK + 1` | primeiro round salvo no dataset dos detectores |
| `PUBLIC_VAL_DIR` | `PFLlibMonza/dataset/MNIST/public_val` | validação pública limpa usada nas features contextuais, separada do `test/` |

## Pipeline (3 passos)

### Passo 1 — Gerar dataset rodando FL com dump

Em `PFLlibMonza/system/`, rodar simulação FL com `--dump_state_dicts <out_dir>` e `--dump_start_round 5`. Cada update de cliente salvo vira um `.safetensors + .json`; cada round salvo também grava `global_rXXX.safetensors` para calcular `delta = local - global`.

```bash
cd PFLlibMonza/system
../../.venv/bin/python main.py -m CNN -data MNIST -nmc 30 -nc 100 -jr 1 -atk all \
    -cc 5 -gr 50 -t 1 -ls 1 -did 0 -rfake 1 \
    -ria 5 \
    --dump_state_dicts ../../state_dicts_monza_cnn_mnist \
    --dump_start_round 5
cd ../..
```

| Flag | Significado |
|---|---|
| `-m CNN` | FedAvgCNN (ou `-m VGG` pra Cifar10) |
| `-data MNIST` | dataset PFLlib particionado |
| `-nc 100` | 100 clientes |
| `-nmc 30` | 30 são maliciosos (~30%) |
| `-atk all` | aleatoriamente sorteia entre {zero, random, shuffle, label} |
| `-cc 5` | sem defesa (modo "gerar dataset puro") |
| `-gr 50` | 50 rounds |
| `-rfake 1` | 100% chance do malicioso atacar a cada round |
| `-ria 5` | ativa ataques só depois do warm-up limpo |
| `--dump_state_dicts <dir>` | salva updates como `r{round:03d}_c{client:03d}_{type}.safetensors + .json` |
| `--dump_start_round 5` | começa a salvar após o warm-up e inclui `global_rXXX.safetensors` |

**Tempo**: ~15-25 min em RTX 5060 Ti. **Tamanho**: ~12 GB (FedAvgCNN ~580k params × 5100 amostras).

**Aceite**:
```bash
ls state_dicts_monza_cnn_mnist/*.json | wc -l                          # ~5000+
ls state_dicts_monza_cnn_mnist | grep -oP 'malicious_\w+|benign' | sort | uniq -c
du -sh state_dicts_monza_cnn_mnist
```

### Passo 2 — Treinar os detectores

Dois detectores em paralelo (paradigmas distintos pra comparação):

#### 2a — Detector DistilBERT+LoRA

> O preprocess do DistilBERT usa ordem canonizada das camadas. Depois de atualizar o código, retreine o artefato DistilBERT antes de comparar `cc=6`.

```bash
STATE_DICTS_DIR=./state_dicts_monza_cnn_mnist \
PUBLIC_VAL_DIR=./PFLlibMonza/dataset/MNIST/public_val \
FINAL_MODEL_DIR=./detector_monza_cnn_mnist \
RUN_DIR=./detector_runs/monza_cnn_mnist \
.venv/bin/python -u src/detector.py
```

**Tempo**: ~15-30 min. **Saída**: `detector_monza_cnn_mnist/` com adapter LoRA, `hybrid_head.pt`, `context_scaler.pkl` e `metrics.json`.

**Aceite**:
- `adapter_model.safetensors`, `hybrid_head.pt`, `context_scaler.pkl` e `metrics.json` devem existir.
- `metrics.json` com `default_argmax.f1` ≥ 0.80.
- `metrics.json` com `default_argmax.by_type` para comparar FPR/recall por tipo de ataque contra o MLP.

#### 2b — Detector MLP (60 features handcrafted)

```bash
STATE_DICTS_DIR=./state_dicts_monza_cnn_mnist \
PUBLIC_VAL_DIR=./PFLlibMonza/dataset/MNIST/public_val \
ARTIFACTS_DIR=./detector_mlp_monza_cnn_mnist \
.venv/bin/python -u src/detector_mlp.py
```

**Tempo**: ~30 segundos (early stop em ~epoch 17). **Saída**: `detector_mlp_monza_cnn_mnist/` (~80 KB).

### Passo 3 — Defesa em produção (`cc=2` / `cc=3` / `cc=6` / `cc=7`)

Re-rodar FL com defesa ativada. `cc=2` e `cc=3` são baselines sem detector treinado; `cc=6` e `cc=7` usam os artefatos treinados:

```bash
cd PFLlibMonza/system
# Cluster cosseno
../../.venv/bin/python main.py -m CNN -data MNIST -nmc 30 -nc 100 -jr 1 -atk all \
    -cc 2 -gr 50 -t 1 -ls 1 -did 0 -rfake 1
# Cosseno + score
../../.venv/bin/python main.py -m CNN -data MNIST -nmc 30 -nc 100 -jr 1 -atk all \
    -cc 3 -gr 50 -t 1 -ls 1 -did 0 -rfake 1
# DistilBERT
../../.venv/bin/python main.py -m CNN -data MNIST -nmc 30 -nc 100 -jr 1 -atk all \
    -cc 6 -gr 50 -t 1 -ls 1 -did 0 -rfake 1 -ria 5 \
    --detector_dir ../../detector_monza_cnn_mnist \
    --bert_threshold_key threshold_label_fpr05
# MLP
../../.venv/bin/python main.py -m CNN -data MNIST -nmc 30 -nc 100 -jr 1 -atk all \
    -cc 7 -gr 50 -t 1 -ls 1 -did 0 -rfake 1 -ria 5 \
    --detector_dir ../../detector_mlp_monza_cnn_mnist \
    --mlp_threshold_key threshold_label_fpr05
cd ../..
```

**Saídas**:
- `PFLlibMonza/system/fpr_frr_results_{2,3,6,7}.csv`: FPR/FRR global por round.
- `PFLlibMonza/system/cc_detail_results_{6,7}.csv`: decisão por cliente/round, com `AttackType`, hits e scores.
- `PFLlibMonza/system/cc_type_results_{6,7}.csv`: FPR benigno e recall por tipo de ataque, incluindo `malicious_label`.
Todos incluem `RunID`; use sempre o último `RunID` para não misturar execuções.

---

## Análise

### CLI rápida

```bash
cd PFLlibMonza/system
for csv in fpr_frr_results_{2,3,6,7}.csv; do
    [ -f "$csv" ] || continue
    echo "=== $csv ==="
    LC_NUMERIC=C awk -F, '
      NR==1 {has_run=($1=="RunID"); next}
      has_run {run=$1; rows[run]=rows[run] $0 "\n"; last=run; next}
      !has_run {rows["legacy"]=rows["legacy"] $0 "\n"; last="legacy"}
      END {printf "%s", rows[last]}
    ' "$csv" \
      | tail -30 \
      | LC_NUMERIC=C awk -F, 'NF==3 {fpr+=$2; frr+=$3; n+=1} NF>=4 {fpr+=$3; frr+=$4; n+=1} END {if(n>0) printf "  FPR_mean=%.4f  FRR_mean=%.4f  (n=%d)\n", fpr/n, frr/n, n}'
done
cd ../..
```

### Notebook visual único

```bash
.venv/bin/jupyter notebook notebook_monza_analysis.ipynb
```

Gera todos os gráficos do projeto em um lugar: FPR/FRR por round, trade-off scatter, métricas offline dos detectores, FPR/recall por tipo de ataque e foco em `malicious_label`. O notebook carrega CSV antigo e novo, escolhe o último `RunID` completo quando existir e compara `cc=2`, `cc=3`, `cc=6` e `cc=7`.

Os PNGs são gerados somente em `analysis_outputs/`. O script completo também remove `plot_*.png` antigos da raiz para evitar confusão com resultados anteriores.

Principais saídas:
- `plot_fpr_frr_by_round.png`
- `plot_tradeoff_fpr_frr.png`
- `plot_detector_metrics.png`
- `analysis_outputs/plot_cc_recall_by_attack_type.png`
- `analysis_outputs/plot_cc_malicious_label_recall.png`

### Scripts opcionais

O notebook acima já roda tudo. Este script fica como alternativa CLI:

```bash
.venv/bin/python scripts/plot_cc_attack_types.py \
    --system-dir PFLlibMonza/system \
    --out-dir analysis_outputs \
    --tail-rounds 30
```

Olhe principalmente:
- `malicious_label` com `Rate` acima de zero.
- `benign` com `Rate` perto ou abaixo de `0.05`.

---

## Estrutura do repo

```
jpt/
├── README.md                    # documentação original (DistilBERT vs MLP detectores)
├── EVOLUTION.md                 # narrativa do desenvolvimento (F1=0.43 → 0.99)
├── RESULTS.md                   # bench atual: 4×2 grid de variantes
├── MONZA_RESULTS.md             # 🆕 resultados experimentais MONZA
├── HOWTO.md                     # 🆕 este arquivo
├── notebook_monza_analysis.ipynb # análise gráfica (cc=2/3/6/7)
├── BertModelsclassify.ipynb     # gerador local de fallback (não-MONZA)
├── analysis_outputs/            # figuras e resumos gerados
├── src/
│   ├── detector.py              # treina DistilBERT+LoRA hibrido (pesos discretizados + features contextuais)
│   ├── detector_mlp.py          # treina MLP+features
│   ├── features.py              # 60 features estatisticas/espectrais (compativel com BaseHeadSplit do PFLlib)
│   ├── bench_grid.py            # bench 4×2 standalone (não usa MONZA)
│   ├── cc.py                    # 🆕 ClientCheck (DistilBERT) — usado pelo MONZA
│   ├── cc_mlp.py                # 🆕 ClientCheckMLP — usado pelo MONZA
│   └── fl_save.py               # 🆕 helper de dump de state_dicts
├── scripts/
│   └── create_label_flip_train_mal.py # cria train_mal/ com label flip deterministico
└── PFLlibMonza/                 # 🆕 fork do PFLlib (FL simulator)
    ├── system/
    │   ├── main.py              # +args --dump_state_dicts e defesas cc=2/3/6/7
    │   ├── flcore/
    │   │   ├── attack/attack.py # ataques zeros/random/shuffle/label
    │   │   ├── clients/         # clientmaliciousavg.py expõe last_attack_type
    │   │   ├── servers/serveravg.py # +cases cc==2/3/6/7
    │   │   ├── trainmodel/models.py # FedAvgCNN, VGG, etc
    │   │   └── detector/        # 🆕 cópia gêmea de cc.py, cc_mlp.py, fl_save.py, features.py
    │   ├── fpr_frr_results_*.csv # outputs por defesa (cc=2,3,6,7 no fluxo principal)
    │   └── run.sh
    └── dataset/                 # precisa existir para gerar/ler MNIST particionado
```

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'cvxpy'"

`PFLlibMonza/system/main.py` importa todos os servers no topo. `serverpac.py` requer cvxpy. Já está no `requirements.txt` da raiz. Se faltar:

```bash
.venv/bin/pip install cvxpy
```

### "FileNotFoundError: '../dataset/MNIST/train/...'"

Dataset MNIST particionado pra `num_clients` errado. Edite `dataset/generate_MNIST.py:13` (`num_clients = 100`) e re-rode `python generate_MNIST.py noniid - dir`. O `check()` em `dataset/utils/dataset_utils.py` regera automaticamente se `num_clients` divergir do `config.json`.

### `cc=6` removendo todos os clientes (FPR=1.0)

O `cc=6` atual usa DistilBERT híbrido: adapter LoRA + `hybrid_head.pt` + `context_scaler.pkl`. Se o artefato estiver incompleto ou antigo, re-treine:

```bash
ls -la detector_monza_cnn_mnist/adapter_model.safetensors \
       detector_monza_cnn_mnist/hybrid_head.pt \
       detector_monza_cnn_mnist/context_scaler.pkl \
       detector_monza_cnn_mnist/metrics.json
```

### Disco insuficiente pra VGG/Cifar10

`-m VGG` gera state_dicts de ~56 MB cada → 100 clients × 50 rounds = 280 GB. Inviável sem amostragem. Opções:

- Reduzir rounds: `-gr 5` (~28 GB).
- Reduzir clientes: `-nc 30`.
- Adicionar amostragem no `fl_save.save_round_dump` (modificar pra dumpar 1 a cada N clientes/rounds).

### `ValueError: Input X contains infinity or a value too large for dtype('float32')`

Features podem explodir em ataques degenerados como `model_zeros`. `features.py:extract_features` já aplica `np.nan_to_num` no final pra sanitizar. Se reaparecer, conferir que o arquivo está atualizado.

---

## Referências

- Plano completo (sessão de desenvolvimento): `/home/wallace/.claude/plans/16-06-27-04-2026-rafael-veiga-lazy-dream.md`
- Bench atual standalone: ver `README.md` e `RESULTS.md` (rodam sem MONZA)
- PFLlib upstream: https://github.com/TsingZ0/PFLlib
- MONZA fork: https://github.com/VeigarGit/PFLlibMonza
