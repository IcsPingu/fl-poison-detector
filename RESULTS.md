# Detector de updates maliciosos em FL

> Este documento é o relatório do **bench original** (dataset sintético gerado pelo `BertModelsclassify.ipynb`/`bench_grid.py`). Para resultados em **FL real** (PFLlibMonza, 100 clientes não-IID, 50 rounds com 4 defesas comparadas), ver [`MONZA_RESULTS.md`](MONZA_RESULTS.md).

Estudo comparativo de duas abordagens para detectar **state_dicts maliciosos** enviados por clientes em Federated Learning. Cliente FL é uma `FedAvgCNN` (~580k pesos); detector classifica binariamente um update vindo de cliente como `benigno` ou `malicioso` antes da agregação.

## TL;DR

**MLP sobre 60 features estatísticas/espectrais/espaciais bate DistilBERT+LoRA sobre tokens-de-bins em todas as variantes do dataset, por margem de 0.10 a 0.15 de F1.** No cenário mais realista (modelo treinado em MNIST + ataques sutis), o MLP atinge F1=0.99 com 0% de FPR; o DistilBERT plateia em 0.88.

| Variante (`pretrained` × `hard`) | DistilBERT F1 (tunado) | MLP F1 |
|---|---|---|
| 1. Leakage (random init + ataques originais) | 0.884 | **1.000** |
| 2. Hard (random init + ataques sutis) | 0.895 | **0.964** |
| 3. **Pretrained + Hard** (mais realista) | 0.881 | **0.990** |
| 4. Pretrained + Easy | 0.859 | **1.000** |

## Os dois detectores

### `detector.py` — DistilBERT+LoRA sobre tokens-de-bins
Trata os pesos como uma sequência: normaliza por camada, discretiza em 10000 bins, faz pooling estratificado pra caber em 512 tokens. Cada bin vira um "input_id" do DistilBERT. LoRA `r=8` em `q_lin`/`v_lin`. Threshold tunado pós-treino.

Limitação estrutural: bins são tokens não-ordinais — bin 100 e bin 101 ficam com embeddings independentes. Perde-se a noção de proximidade entre valores de pesos.

### `detector_mlp.py` — MLP sobre 60 features
Calcula 15 features por camada × 4 camadas com `weight` no nome:

- **Estatísticas**: l2, linf, mean, std, kurt, zero_ratio, p5, p95, hist_entropy
- **Espectrais**: top-3 singular values normalizados por Frobenius, FFT high/low ratio
- **Espaciais**: total variation, autocorrelação Pearson lag-1

Vetor de 60 floats → `BatchNorm → 60→128 → ReLU → Dropout → 128→64 → ReLU → Dropout → 64→2`. ~13k params. Treina em ~5s com early stopping.

Features explícitas dão sinal pronto pro MLP — não precisa "redescobrir" estrutura.

## Os 4 datasets do grid

Controlados por dois flags no notebook (`BertModelsclassify.ipynb` cell 3):

| flag | True | False |
|---|---|---|
| `USE_PRETRAINED_BASE` | FedAvgCNN treinado em MNIST por 10 epochs (test_acc ~0.99) | Random init |
| `HARDEN_ATTACKS` | Cada malicioso parte de fresh_base + ruído. Ataques: `random_smart` (Gaussiano com sigma da camada), `shuffle parcial 30-100%`, `noise` SNR uniforme [3, 15] dB | Base compartilhado entre maliciosos. Ataques fixos: U[0,1], shuffle 100%, SNR=5 dB |

Combinação 2×2 = 4 datasets.

## Achados-chave

1. **MLP > DistilBERT em todos os cenários** (gap 0.10-0.15 F1). Features explícitas batem tokenização para esse problema numérico.

2. **DistilBERT plateia em 0.88** independentemente do dataset. Tipo dos ataques, leakage, pretreino — nada disso muda muito. É teto estrutural do paradigma de bins-como-tokens.

3. **Shuffle em random init é detectável** com dados suficientes. Em rodada com 200 amostras tinha recall=0% (parecia indistinguível); com 1000 amostras pula pra 80%. A flutuação estatística entre matrizes random shuffleadas e benignas é pequena mas não-zero — vira sinal aproveitável com escala.

4. **Pretreino traz +0.03 marginais** quando há dados (1000 amostras). Crítico em regime de dados escassos, pequeno em regime amplo.

5. **Limite informacional real**: noise com SNR alto (>10 dB) contra background estruturado. F1 não passa de ~92% pra essa categoria — ruído indistinguível por construção.

## Reproduzir

```bash
# 1) Setup
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 2) Roda o grid completo (4 datasets × 2 detectores, ~30-40 min na RTX 5060 Ti)
.venv/bin/python src/bench_grid.py
```

`bench_grid.py` faz tudo:
- Treina FedAvgCNN em MNIST 1× (cache em `mnist_data/`)
- Gera 4 datasets em `state_dicts_grid/{1_leakage,2_hard,3_pretrained_hard,4_pretrained_easy}/`
- Roda `detector.py` (DistilBERT) e `detector_mlp.py` (MLP) em cada
- Imprime tabela final + breakdown por ataque
- Salva tudo em `bench_grid_results.json`

Logs por run em `detector_grid_runs/{variante}/{distilbert,mlp}/log.txt`.

### Variáveis de ambiente úteis

- `GRID_N_SAMPLES_PER_CLASS=200` — iteração rápida (default 1000)
- `GRID_SKIP_DISTILBERT=1` — só MLP (~3 min total)
- `STATE_DICTS_DIR=...`, `FINAL_MODEL_DIR=...`, `ARTIFACTS_DIR=...` — pra rodar `detector.py`/`detector_mlp.py` standalone em diretório custom

## Estrutura

```
detector.py            DistilBERT+LoRA sobre tokens-de-bins
detector_mlp.py        MLP sobre features
features.py            extract_features(state_dict) -> (np.ndarray[60], names)
bench_grid.py          orquestrador do grid 4×2
BertModelsclassify.ipynb  notebook de exploração + geração de datasets ad-hoc
requirements.txt       deps unificadas para detectores, MONZA e notebooks
```

## Limitações conhecidas

- **FedAvgCNN apenas**: features assumem 4 camadas com `weight` no nome (`conv1.0.weight`, `conv2.0.weight`, `fc1.0.weight`, `fc.weight`). Para outras arquiteturas precisa adaptar `LAYERS` em `features.py`.
- **Detecção isolada por update**: não considera comparação entre clientes (Krum/Multi-Krum) nem trajetória multi-round (FLDetector). Defesa complementar, não substituta.
- **Threshold tunado in-sample** no `detector.py` — métricas tunadas otimistas (não temos val separado).

## Conclusão

Para detectar updates maliciosos em FL com benigno definido como "modelo global treinado + ruído de 1 step local", **features estatísticas/espectrais/espaciais combinadas com um MLP simples são suficientes**: F1 0.96–1.0 nas 4 variantes testadas, com 0–1% de FPR. Transformer sobre tokens-de-bins é dead-end estrutural pra esse problema (~0.88 plateau).

Conclusão validada em FL real na **[Fase 7](EVOLUTION.md#fase-7--integração-com-fl-real-monza--pfllib)**: MLP+features mantém Pareto-superioridade sobre DistilBERT e sobre os baselines do PFLlib (cluster cosseno, cosseno+score). O `cc=8` é uma defesa MONZA posterior ao bench sintético, criada para testar label flip com validação pública. Ver [`MONZA_RESULTS.md`](MONZA_RESULTS.md).
