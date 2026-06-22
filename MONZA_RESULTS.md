# MONZA × jpt — Resultados experimentais

Pipeline completo: **PFLlibMonza** (FL real) gera dataset → **detector DistilBERT/MLP** (jpt) treina → defesa volta como `cc=6/cc=7` no servidor MONZA, comparada com baselines `cc=2` (cluster) e `cc=3` (cosseno+score).

> **Nota pós-resultado**: este relatório preserva os resultados fechados de 2026-04-28 para `cc=2/3/6/7`. O `cc=8` (MLP+validação pública para label flip), `cc=9` (DistilBERT+MLP+label-flip check) e `cc=10` (DistilBERT+MLP+TargetLF) foram adicionados depois e ainda precisam de novo run experimental para ter FPR/FRR reportados.
>
> **Correção importante**: uma auditoria posterior encontrou bug em `utils/data_utils.py`: `is_malicious=True, is_train=True` imprimia "Malicious label", mas acabava lendo `test/` em vez de `train_mal/`. Portanto, os números antigos de `label flip` abaixo devem ser tratados como históricos e precisam ser rerodados após gerar `PFLlibMonza/dataset/MNIST/train_mal/`.

**Data**: 2026-04-28
**Hardware**: RTX 5060 Ti (16GB), CUDA 13, torch 2.11
**Seed**: 42

## Configuração do experimento

| Parâmetro | Valor |
|---|---|
| Modelo | FedAvgCNN (1 in_features, 1024 dim, ~580k params) |
| Dataset | MNIST particionado Dirichlet non-IID (alpha=0.1) |
| Clientes | 100 |
| Maliciosos | 30 (~30%) |
| Rounds | 50 |
| Local epochs | 1 |
| Ataques | `model_zeros` (na verdade, ones), `random_param`, `shuffle_model`, `label_flip` |
| Detecção | aplicada após `receive_models()`, antes de `aggregate_parameters()` |
| Quarentena | clientes detectados ficam `2*N_quarantined` rounds fora |

## Geração do dataset (passo 1 do pipeline)

Rodada com `-cc 5` (sem defesa) por 50 rounds:
- **5100 amostras** salvas em `state_dicts_monza_cnn_mnist/` (12 GB)
- Distribuição: 70.6% benign, 7.4% zeros, 6.9% random, 7.2% shuffle, 7.9% label
- Cada amostra: `r{round:03d}_c{client:03d}_{type}.safetensors` + `.json`

## Detectores treinados (passo 2)

### NLP — DistilBERT + LoRA
- **F1 = 0.838**, Precision = 0.978, Recall = 0.733
- Treino: 15 epochs, early stopping patience=7, ~15 min
- Modelo: 280 MB (base + adapter LoRA com `modules_to_save`)
- Bug fixado nesta sessão: `LoraConfig` precisava `modules_to_save=['pre_classifier','classifier']` para o head ser serializado no adapter (`detector.py:153`)

### MLP — 60 features + MLP
- **F1 = 0.851**, Precision = 1.000, Recall = 0.740
- Treino: early stopping na epoch 17 (best @ epoch 2), ~30 segundos
- Modelo: 80 KB (60→128→64→2)

### Recall por tipo de ataque (eval set in-sample)

| Ataque | NLP recall | MLP recall |
|---|---|---|
| zeros | 100% | 100% |
| random | 100% | 100% |
| shuffle | 100% | 98.6% |
| **label flip** | **6.2%** | **4.9%** |
| benign FPR | 1.67% | **0.00%** |

## Defesas em produção (passo 3)

50 rounds de FL com `-rfake 1` (cliente malicioso ataca todo round). Métricas FPR/FRR computadas pelo MONZA via quarentena.

| cc | Defesa | FPR (média 30 últimos rounds) | FRR (média 30 últimos rounds) |
|---|---|---:|---:|
| 2 | Cluster cosseno (PFLlib) | **0.0000** | 0.2622 |
| 3 | Cosseno + score (PFLlib) | 0.0533 | 0.1144 |
| 6 | NLP DistilBERT (este trabalho) | 0.1124 | 0.1144 |
| **7** | **MLP + features (este trabalho)** | **0.0000** | **0.1556** |

`cc=8`, `cc=9` e `cc=10` não aparecem nesta tabela porque não existiam no run original. Eles devem ser avaliados separadamente com `PFLlibMonza/system/fpr_frr_results_{8,9,10}.csv`.

**Vencedor: cc=7 (MLP+features)** — Pareto-ótimo. Zero falsos positivos (não pune benignos) e captura 84% dos maliciosos. Supera todos os 3 baselines em pelo menos uma das duas métricas.

## Achados

### 1. MLP+features supera baselines do PFLlib em FL real

cc=7 tem **FPR = 0%** (idêntico ao cc=2 cluster) **e FRR = 15.6%** (melhor que cc=2 que tem 26%, comparável a cc=3 que tem 11% mas com FPR=5%). Trade-off ótimo.

### 2. DistilBERT+LoRA é desperdício para essa tarefa

Mesmo com bug fixado e classifier head salvo corretamente, o NLP em produção tem **FPR=11%** — pune 1 a cada 9 benignos. Custo prático alto em FL real. F1 do treino similar ao MLP (0.83 vs 0.85), mas não generaliza tão bem do eval set in-sample para a distribuição rolando do FL.

Tradeoff prático: para 100 clientes com 30 maliciosos, cc=6 fica banindo ~8 benignos por round.

### 3. Label flip exige rerun com `train_mal/` correto

No run histórico, dois paradigmas radicalmente diferentes (DistilBERT learning vs handcrafted features SVD/FFT/momentos) chegaram ao mesmo teto: **~5% recall** em label flip. Depois foi encontrado o bug de leitura descrito no topo deste arquivo, então essa leitura não deve ser usada como conclusão final.

A hipótese técnica continua válida: label flip tende a manter distribuição de pesos parecida com benign, mas otimiza função errada. Por isso, o rerun deve comparar fingerprint (`cc=6/7`) contra sinais de comportamento (`cc=8/9/10`).

Detecção de label flip exige outra abordagem:
- Comparação de gradientes entre clientes
- Validation set hold-out público (`cc=8`, adicionado depois deste relatório)
- Margem para rótulo invertido em holdout limpo (`cc=10`, adicionado depois deste relatório)
- Comparação cross-round de comportamento do modelo

### 4. Bug do LoRA: `modules_to_save` é necessário pra classification heads

Bug encontrado em `detector.py:153` (linha original sem `modules_to_save`). Afetou o run inicial: classifier head saía aleatório no carregamento via `PeftModel.from_pretrained`. Em produção, NLP marcou todos os 100 clientes como maliciosos no round 1.

Lição: ao usar LoRA com tasks que precisam de heads novos (classification, regression), sempre inclua os heads em `modules_to_save`. PEFT só persiste os módulos LoRA por padrão — o resto do modelo (head incluído) volta dos pesos do checkpoint base.

### 5. Baselines do PFLlib não pegam ataques sutis bem

- `cc=2` (cluster cosseno) é conservador demais: FPR=0% mas perde 26% dos maliciosos.
- `cc=3` (cosseno + score) é meio-termo: FPR=5%, FRR=11%.

Em geral, métodos baseados em cosseno entre updates dependem de assumir que maliciosos são "outliers angulares". Para `model_ones`, `random_param` etc isso vale, mas para `label flip` (pesos similares a benign), os baselines também perdem.

## Ranking pelo critério honesto

Score combinado (FPR + FRR, menor = melhor):

1. **cc=7 (MLP)**: 0.000 + 0.156 = **0.156** 🏆
2. cc=3 (cosseno+score): 0.053 + 0.114 = 0.168
3. cc=6 (NLP): 0.112 + 0.114 = 0.226
4. cc=2 (cluster): 0.000 + 0.262 = 0.262

## Caveats

1. **Eval in-sample**: detectores treinados e avaliados no mesmo conjunto MONZA. F1 reportado é otimista. Sem cross-validation pra outro dataset/seed.
2. **Histórico `model_zeros` no MONZA**: nos resultados antigos, `model_zeros` usava `torch.ones`, não `torch.zeros`. O pipeline atual zera de verdade; portanto essa categoria precisa de novo dump e retreino antes de comparar com este relatório.
3. **`model_noise` bugado** (`attack.py:52` early return): `-atk all` cobre 4 categorias, não 5. Em produção real, ataques com noise alto vão passar pelo detector.
4. **VGG/Cifar10 cortado** por disco (~280GB inviável). Sem evidência de generalização além de FedAvgCNN/MNIST.
5. **Single seed (42)** — sem intervalo de confiança nos números do detector.
6. **Threshold tunado in-sample** no NLP — métrica otimista (mantida pra paridade com bench original).

## Trabalho futuro

- **Cross-validation seed**: re-rodar com seeds 123, 456, 789 e reportar média ± std.
- **Cifar10/VGG**: requer liberar 100+ GB ou implementar amostragem no `fl_save` (dump 1 a cada N rounds).
- **Validar `cc=8`**: rodar MLP+validação pública contra `label_flip` e comparar FPR/FRR com `cc=7`.
- **Combinar cc=2 + cc=7**: cluster filtra grosseiros, MLP filtra residuais. Ensemble pode ter FPR+FRR menor que ambos individualmente.
- **Resistência a poisoning adaptativo**: testar atacantes que conhecem o detector e tentam evitar.

## Reprodutibilidade

Comandos completos em `/home/wallace/.claude/plans/16-06-27-04-2026-rafael-veiga-lazy-dream.md` (Fases C.0 → E.4).

Artefatos preservados:
- `state_dicts_monza_cnn_mnist/` (12 GB) — dataset gerado
- `detector_monza_cnn_mnist/` (~3 MB) — DistilBERT+LoRA treinado
- `detector_mlp_monza_cnn_mnist/` (~80 KB) — MLP treinado
- `PFLlibMonza/system/fpr_frr_results_{2,3,6,7}.csv` — resultados FL deste relatório
- `PFLlibMonza/system/fpr_frr_results_{8,9,10}.csv` — esperado em novo run com defesas pós-relatório

Análise visual em `notebook_monza_analysis.ipynb`.
