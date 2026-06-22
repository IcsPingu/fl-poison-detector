# fl-poison-detector

Detector binário de **updates maliciosos** em Federated Learning. Recebe um `state_dict` de cliente FL (uma `FedAvgCNN` com ~580k pesos) e classifica como **benigno** ou **malicioso** antes da agregação.

Duas abordagens implementadas e comparadas:
- **`detector.py`** — DistilBERT + LoRA sobre pesos discretizados + ramo tabular com features contextuais
- **`detector_mlp.py`** — MLP sobre features estatísticas dos pesos + delta local-global + validação pública limpa

## TL;DR

| Variante (`pretrained` × `hard`) | DistilBERT F1 | MLP F1 |
|---|---|---|
| 1. Leakage | 0.88 | **1.00** |
| 2. Hard | 0.89 | **0.96** |
| 3. **Pretrained + Hard** (mais realista) | 0.88 | **0.99** |
| 4. Pretrained + Easy | 0.86 | **1.00** |

MLP+features ganha por 0.10–0.15 F1 em todos os cenários.

Validação posterior em FL real (PFLlibMonza, 100 clientes Dirichlet non-IID, 30 maliciosos):

| cc | Defesa | FPR | FRR |
|---|---|---:|---:|
| 6 | NLP DistilBERT | 0.112 | 0.114 |
| **7** | **MLP+features** | **0.000** | **0.156** |

MLP+features ficou como melhor detector final no fluxo normalizado. Detalhes em [`MONZA_RESULTS.md`](MONZA_RESULTS.md).

Documentação:
- [`HOWTO.md`](HOWTO.md) — passo-a-passo do pipeline FL real (gera dataset com MONZA → treina detector → defesas cc=2/cc=3/cc=6/cc=7)
- [`MONZA_RESULTS.md`](MONZA_RESULTS.md) — resultados experimentais em FL real
- [`RESULTS.md`](RESULTS.md) — bench original 4×2 (dataset sintético)
- [`EVOLUTION.md`](EVOLUTION.md) — como o projeto evoluiu
- [`notebook_monza_analysis.ipynb`](notebook_monza_analysis.ipynb) — gráficos comparativos das defesas MONZA (`cc=2`/`cc=3`/`cc=6`/`cc=7`)

Run completo MONZA do zero:

```bash
./scripts/run_full_monza.sh --background
tail -f rerun_full_*.log
```

O run completo usa por padrão `ROUND_INIT_ATK=5` e `DUMP_START_ROUND=6`: os rounds iniciais fazem warm-up limpo, depois o dump salva cada update junto com o modelo global anterior do round. `cc=6` e `cc=7` usam `threshold_label_fpr05` para focar em `malicious_label` mantendo FPR benigno baixo. As features contextuais usam `PFLlibMonza/dataset/MNIST/public_val/`, separado do `test/` usado para avaliação.

## Quick start

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python src/bench_grid.py
```

`bench_grid.py` faz tudo: treina baseline em MNIST, gera 4 variantes do dataset, roda os 2 detectores em cada uma, imprime tabela final + breakdown por ataque, salva tudo em `bench_grid_results.json`. ~30–40 min na RTX 5060 Ti.

Sempre executar a partir da **raiz do projeto** — paths como `state_dicts/`, `mnist_data/` etc. são relativos ao cwd.

O ambiente Python é único: use sempre `.venv/` na raiz. MONZA também deve ser executado com essa venv (`../../.venv/bin/python` quando o cwd for `PFLlibMonza/system`).

Nota sobre `malicious_label`: o runtime MONZA agora exige `PFLlibMonza/dataset/MNIST/train_mal/` para label flip real. O script `scripts/run_full_monza.sh` cria esse diretório automaticamente; para criar manualmente, use `scripts/create_label_flip_train_mal.py`.

## Estrutura

```
.
├── README.md, RESULTS.md, EVOLUTION.md   # docs do bench original (sintético)
├── HOWTO.md, MONZA_RESULTS.md            # docs da integração FL real (PFLlibMonza)
├── requirements.txt, .gitignore
├── BertModelsclassify.ipynb              # notebook ad-hoc com flags de geração
├── notebook_monza_analysis.ipynb         # gráficos comparativos das defesas MONZA
├── bench_grid_results.json               # resultados do bench 4×2
├── src/
│   ├── detector.py                       # DistilBERT+LoRA sobre pesos→bins
│   ├── detector_mlp.py                   # MLP sobre features handcrafted
│   ├── features.py                       # extrator de 60 features de pesos
│   ├── context_features.py               # delta local-global + validação pública
│   ├── bench_grid.py                     # orquestrador 4×2
│   ├── cc.py                             # ClientCheck DistilBERT standalone
│   ├── cc_mlp.py                         # ClientCheckMLP — usado pelo MONZA
│   └── fl_save.py                        # helper de dump de state_dicts
└── PFLlibMonza/                          # fork PFLlib (FL simulator) integrado
    └── system/flcore/detector/           # inferência MONZA: cc/cc_mlp/fl_save/features
```

Saídas geradas em runtime (todas no `.gitignore`, raiz do projeto):

| Diretório | Conteúdo |
|---|---|
| `state_dicts/` ou `state_dicts_grid/{variante}/` | `.safetensors` + `.json` por amostra |
| `detector_final/` | Modelo DistilBERT+LoRA híbrido treinado + scaler contextual + `metrics.json` |
| `detector_mlp_artifacts/` | MLP + scaler + `feature_names.json` + `report.json` |
| `detector_grid_runs/{variante}/{detector}/` | Logs e artefatos por run do grid |
| `mnist_data/` | Cache do MNIST (baixado pelo `bench_grid`) |

## Documentação por arquivo

### `detector.py`

Pipeline DistilBERT+LoRA híbrido:

1. `preprocess_weights(state_dict)` — ordena camadas de forma canônica, pega tensores com `'weight'` no nome, normaliza cada um por quantis (q5/q95) com clamp em [0, 1], concatena, faz **pooling estratificado** via `torch.linspace(0, n-1, 512)` (em vez de truncamento), discretiza em 10000 bins (PAD_ID=0 reservado).
2. `extract_context_features(...)` monta sinais comportamentais para `malicious_label`: delta local-global, deltas da cabeça/classificador e métricas em validação pública MNIST limpa.
3. `tokenize_function` monta `input_ids` + `attention_mask` (1 para tokens não-PAD); as features contextuais entram em paralelo, normalizadas com `StandardScaler`.
4. `build_and_train(seed)` — DistilBERT base + LoRA `r=8` em `q_lin`/`v_lin`; o vetor `[CLS]` é concatenado com um ramo tabular (`LayerNorm -> MLP`) antes da classificação. Treino: 15 epochs, lr=2e-4, weight_decay=0.01, scheduler cosine, warmup 6%, batch=16.
4. `tune_threshold(logits, labels)` — sweep de 200 thresholds em `(logit_mal − logit_ben)`, escolhe o que maximiza F1. Tunado in-sample no eval — métrica é otimista mas marginal.
5. `breakdown_by_type` — recall por tipo de ataque (`zeros`, `random`, `shuffle`, `noise`).

`MODEL_SEED=15880` foi escolhido em experimento de ensemble como o que dá melhor F1 individual. Persistência em `FINAL_MODEL_DIR` + `metrics.json`.

### `detector_mlp.py`

Pipeline MLP:

1. `load_dataset()` — itera `state_dicts/*.safetensors` + `.json`, chama `extract_features`. Resultado: matriz X (N×60) + labels y + types.
2. `stratified_split(types, ...)` — `StratifiedShuffleSplit` por **tipo** de ataque (não só label) — garante que cada split tem amostras de cada categoria.
3. `StandardScaler` ajustado só no treino, persistido em `scaler.pkl`.
4. `MLPDetector` — `BatchNorm1d(60) → Linear(60→128) → ReLU → Dropout(0.3) → Linear(128→64) → ReLU → Dropout(0.3) → Linear(64→2)`. ~13k parâmetros.
5. Treino: AdamW lr=1e-3 wd=1e-4, scheduler `CosineAnnealingLR` por 60 epochs, batch=32, early stopping `patience=15` em F1 do eval, restaura best checkpoint.
6. Avaliação final + `breakdown_by_type` + `report.json` + `feature_names.json` em `ARTIFACTS_DIR`.

### `features.py`

Extrator puro (não tem treino). 4 camadas processadas: `conv1.0.weight`, `conv2.0.weight`, `fc1.0.weight`, `fc.weight`. Conv kernels viram matriz `(out, in·kH·kW)` para SVD/FFT 2D coerentes.

15 features por camada (× 4 camadas = 60):

| Categoria | Feature | O que mede |
|---|---|---|
| Magnitude | `l2`, `linf` | Norma Frobenius e máximo absoluto |
| Distribuição | `mean`, `std`, `kurt`, `zero_ratio`, `p5`, `p95` | Momentos e percentis |
| Entropia | `hist_entropy` | Entropia do histograma de 50 bins |
| Espectral | `sv1`, `sv2`, `sv3` | Top-3 singular values normalizados por Frobenius |
| Frequencial | `fft_hf_ratio` | Razão energia high-freq / low-freq via FFT-2D |
| Espacial | `tv` | Total variation média entre pesos vizinhos |
| Espacial | `autocorr1` | Autocorrelação Pearson lag-1 |

Roda na GPU se disponível (`torch.linalg.svdvals`, `torch.fft.fft2`). Custo ~ms por amostra.

### `bench_grid.py`

Orquestrador único que faz benchmark completo. Etapas:

1. Treina **`pretrained_base`** = `FedAvgCNN` em MNIST (10 epochs, ~1–2 min, cache em `mnist_data/`).
2. Gera 4 datasets em `state_dicts_grid/{1_leakage, 2_hard, 3_pretrained_hard, 4_pretrained_easy}/` (cada um N amostras de cada classe).
3. Roda `detector.py` e `detector_mlp.py` via subprocess para cada variante (8 treinos sequenciais), com env vars apontando pros dirs corretos. Streaming dos logs em tempo real, prefixados com `[variante DB|MLP]`.
4. Lê `metrics.json` / `report.json` de cada run, monta tabela final + breakdown por ataque, salva em `bench_grid_results.json`.

### `BertModelsclassify.ipynb`

Notebook de exploração + geração ad-hoc de dataset:
- **Cell 1**: definições (`FedAvgCNN`, ataques, helpers).
- **Cell 3**: **CONFIG + setup do `pretrained_base`** — flags `USE_PRETRAINED_BASE` (treina em MNIST se True), `HARDEN_ATTACKS` (ataques sutis se True), `N_SAMPLES_PER_CLASS`. Treina baseline conforme flag.
- **Cell 5**: gerador de `state_dicts/`, branching condicional pelo `HARDEN_ATTACKS`.

Útil pra gerar **um dataset específico** sem rodar o grid completo. Os flags do notebook reproduzem qualquer das 4 variantes do `bench_grid.py`.

## Configuração via env vars

| Variável | Usado por | Default | Descrição |
|---|---|---|---|
| `STATE_DICTS_DIR` | `detector.py`, `detector_mlp.py` | `state_dicts` | Pasta de leitura dos `.safetensors` |
| `FINAL_MODEL_DIR` | `detector.py` | `./detector_final` | Pasta de saída do modelo DistilBERT |
| `RUN_DIR` | `detector.py` | `./detector_runs/best` | `output_dir` do `Trainer` HF |
| `ARTIFACTS_DIR` | `detector_mlp.py` | `detector_mlp_artifacts` | Pasta de saída do MLP |
| `GRID_N_SAMPLES_PER_CLASS` | `bench_grid.py` | `1000` | Tamanho de cada variante do grid |
| `GRID_SKIP_DISTILBERT` | `bench_grid.py` | `0` | `1` pula DistilBERT (~3 min total só MLP) |

Exemplo — rodar grid rápido só com MLP:

```bash
GRID_N_SAMPLES_PER_CLASS=200 GRID_SKIP_DISTILBERT=1 .venv/bin/python src/bench_grid.py
```

Exemplo — rodar `detector.py` standalone num dataset custom:

```bash
STATE_DICTS_DIR=meus_dados FINAL_MODEL_DIR=./meu_modelo .venv/bin/python src/detector.py
```

## Reprodutibilidade

- Seeds fixas: `SEED=42` (data split) e `MODEL_SEED=15880` (treino do DistilBERT)
- `torch.backends.cudnn.deterministic=True` no MLP
- `bench_grid.py` é determinístico exceto pela parte de download do MNIST (a primeira vez)
- Resultado esperado bate com `bench_grid_results.json` ±0.01 F1

## Limitações conhecidas

- **Apenas FedAvgCNN**: `features.py` e `detector.py` assumem 4 camadas com `weight` no nome (`conv1.0.weight`, `conv2.0.weight`, `fc1.0.weight`, `fc.weight`). Para outras arquiteturas, ajustar `LAYERS` em `features.py`.
- **Detecção isolada por update**: não usa comparação entre clientes (Krum/Multi-Krum/FoolsGold) nem trajetória multi-round (FLDetector). Defesa complementar, não substituta.
- **Threshold tunado in-sample**: `detector.py:tune_threshold` usa o eval set também pra escolher o threshold. Métrica tunada é otimista.
- **noise SNR alto é ceiling real**: contra benign treinado, ruído com SNR > 10 dB é ~indistinguível por construção.
