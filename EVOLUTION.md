# Evolução do projeto: lógica e metodologia

Como saímos de **F1=0.43 com bugs** para **F1=0.99 num benchmark realista**. Cada fase identificou um problema distinto — alguns eram bugs, outros decisões metodológicas, outros descobertas que reescreveram a abordagem.

## Ponto de partida

`detector.py` original usava **DistilBERT+LoRA** sobre os pesos discretizados em bins (cada peso vira um inteiro de 0 a 9999, vetor de 512 ints vira `input_ids`). Treino com 200 amostras (100 benignos + 100 maliciosos) por 3 epochs. Resultado: F1=0.43, confiança aleatória.

Diagnóstico inicial revelou múltiplos bugs e más decisões:

1. **`attention_mask` mascarava bin 0 válido** — `padding_id=0` colidia com o menor peso normalizado. Pesos no bin mínimo eram tratados como padding e ignorados.
2. **`BitsAndBytesConfig` com kwarg inválido** — `torch_dtype` não é argumento desse config; deveria ir no `from_pretrained`.
3. **Sem `if __name__ == '__main__'`** — importar `detector.py` num REPL disparava o treino.
4. **Truncamento brutal** dos pesos pra caber em 512 tokens — pegava os primeiros 512, descartando 99,9% (basicamente a `conv1` toda e nada de `conv2`/`fc1`/`fc`).
5. **Hiperparâmetros default do HuggingFace** — 3 epochs, lr=5e-5, sem weight decay nem scheduler, sem early stopping. Default é pra LLMs gigantes em datasets enormes; nosso caso é MLP-equivalente em 200 amostras.

## Fase 1 — fixes técnicos no DistilBERT

Corrigimos os bugs e ajustamos o pipeline. As mudanças com impacto:

- **Pooling estratificado** via `torch.linspace(0, n-1, 512)` em vez de truncamento — passa a representar todas as camadas (conv1+conv2+fc1+fc) de forma proporcional.
- **Normalização per-camada por quantis** (q5/q95) em vez de min/max global — cada camada mantém resolução em sua própria escala, e q5/q95 é robusto a outliers.
- **Hiperparâmetros realistas**: 15 epochs, lr=2e-4, weight_decay=0.01, scheduler cosine, warmup 6%, early stopping com `patience=7`, batch=16.
- **Threshold tuning pós-treino** (sweep de 200 thresholds em `logit_mal − logit_ben`).
- **Estrutura limpa** com `main()` + `if __name__`, seeds fixas (`SEED=42`, `MODEL_SEED=15880`).

Resultado: F1 **0.43 → 0.89** no dataset original (que ainda tinha leakage — voltaremos a isso).

Insight importante dessa fase: rodamos um **ensemble de 5 seeds** esperando ganho. Ensemble piorou o F1 (0.85 vs 0.89 do melhor individual). Conclusão: erros individuais são correlacionados, ensemble não ajudou. Ficamos com o melhor individual.

## Fase 2 — paradigma alternativo: MLP+features

Pesquisamos a literatura de FL Byzantine detection (FLDetector KDD'22, OptiGradTrust 2025, FedDMC, FLAIR, SVD+IsoForest 2024). Padrão claro: **features estatísticas/espectrais por camada → MLP** é o approach que funciona em FL.

Construímos `features.py` com 13 features por camada (52 totais, depois 60):

- Estatísticas: `l2`, `linf`, `mean`, `std`, `kurt`, `zero_ratio`, `p5`, `p95`, `hist_entropy`
- Espectrais: top-3 singular values normalizados por Frobenius (capturam shuffle), FFT high/low ratio (captura noise)

`detector_mlp.py` usa um MLP simples (60→128→64→2, ~13k parâmetros) sobre StandardScaler. Treino: AdamW, CosineAnnealingLR, early stopping em F1, em ~5s.

Resultado: **F1=1.00** com 0% FPR no dataset original. Aparentemente perfeito.

**Insight estrutural**: DistilBERT trata bin 100 e bin 101 como tokens não-relacionados — perde a ordinalidade dos valores. Com features explícitas, o MLP só precisa combinar — não precisa "redescobrir" relações entre pesos.

## Fase 3 — descoberta do leakage e endurecimento

F1=1.00 era **ilusão**. Análise cuidadosa do dataset revelou:

- **Os 200 maliciosos compartilhavam o mesmo `base_model`** (variável global na cell 3 do notebook). Cada um era apenas uma versão do `base_model` com um ataque aplicado.
- O `shuffle_model(base_model)` produzia 50 amostras com **stats globalmente idênticas** (mean, std, min, max — porque shuffle preserva tudo). Detectar isso era trivial: o modelo aprendia a "fingerprint" do `base_model` e flagava qualquer derivado.
- `random_param` substituía pesos por `U[0,1]` — distribuição com mean=0.5 e kurtose=1.8, drasticamente diferente da `kaiming_uniform_` de qualquer init. Trivial.

Endurecemos os ataques no notebook:

- **`fresh_base` por amostra**: cada malicioso parte de um `FedAvgCNN()` próprio (anti-leakage).
- **`random_smart`**: Gaussiano com sigma da própria camada. Preserva `mean/std/percentiles`; só quebra estrutura espacial.
- **`shuffle parcial`**: permuta uma fração `frac ∈ U[0.3, 1.0]` dos pesos por tensor.
- **`noise SNR variável`**: SNR uniforme `[3, 15] dB` (antes era fixo em 5dB; SNR=15 = ruído ~10% do sinal, muito sutil).

Resultado: ambos detectores **desabaram**.
- DistilBERT: F1 **0.89 → 0.43** (pior que aleatório!)
- MLP: F1 **1.00 → 0.86**, com `shuffle` recall = **0%**

`shuffle` em random init virou impossível com features estatísticas: TV, autocorrelação, momentos, percentis — todos invariantes (em expectativa) sob permutação de uma matriz random.

## Fase 4 — baseline realista

Pesquisa em mais profundidade (Yunis "Spectral Dynamics" 2024, FedLLM-Bench NeurIPS'24, "Permutation Invariant Functions" 2025) revelou:

- Em FL real, ataques **não acontecem no round 0** com modelo random. Ocorrem após 5–50 rondas, quando os pesos já têm estrutura espacial mensurável.
- Random init é artefato de benchmarks acadêmicos. Em deployment, baseline é sempre treinado.
- Yunis mostrou: após ~10 epochs em MNIST, singular values divergem do Marchenko-Pastur law, total variation cai >30%, autocorr lag-1 sobe pra >0.3.

Solução: **treinar `pretrained_base = FedAvgCNN` em MNIST por 10 epochs antes** de qualquer geração de amostras. Cada cliente FL (benigno e malicioso) parte desse modelo treinado. Variação entre amostras vem do ruído pequeno (proxy de 1 step local de SGD).

Adicionamos também 2 features espaciais que só fazem sentido com pesos estruturados:
- **`tv`** (total variation média) — pesos suaves (treinados) têm TV baixa; shuffle aumenta.
- **`autocorr1`** (Pearson lag-1) — treinado tem autocorr 0.3+; shuffle leva a 0.

Resultado: MLP **F1=0.97** no dataset realista (random init + hardened + pretrained). `shuffle` voltou pra **100%**, `noise` foi pra 80% (perde alguns SNR muito altos), os outros 100%.

## Fase 5 — bench grid 4×2

Pra mapear o espaço completamente, criamos `bench_grid.py` que cruza 2 flags ortogonais:

| | `HARDEN_ATTACKS=False` | `HARDEN_ATTACKS=True` |
|---|---|---|
| `USE_PRETRAINED_BASE=False` | **Leakage** (random init + ataques originais) | **Hard** (random init + ataques sutis) |
| `USE_PRETRAINED_BASE=True` | **Pretrained+Easy** (treinado + originais) | **Pretrained+Hard** (treinado + sutis) |

Subimos `N_SAMPLES_PER_CLASS` para 1000 (1600 train / 400 eval — intervalo de confiança ~3x mais apertado) e rodamos os 2 detectores em cada variante.

Resultados:

| Variante | DistilBERT F1 | MLP F1 |
|---|---|---|
| 1. Leakage | 0.88 | **1.00** |
| 2. Hard | 0.89 | **0.96** |
| 3. Pretrained+Hard | 0.88 | **0.99** |
| 4. Pretrained+Easy | 0.86 | **1.00** |

MLP venceu em todas as variantes por gap consistente de 0.10–0.15 F1. DistilBERT plateia perto de 0.88 independente do dataset — teto estrutural.

## Fase 6 — surpresa metodológica

Em fase 4 (200 amostras), MLP teve `shuffle=0%` no dataset hard sem pretreino. Concluímos com base em pesquisa que era "informacionalmente indistinguível".

Em fase 5 (1000 amostras), o mesmo cenário deu `shuffle=80%`.

**A claim teórica era direcionalmente correta mas absoluta demais.** Autocorr e TV de matrizes random shuffleadas têm flutuação estatística pequena (não exatamente zero) — abaixo do limiar de detecção com 200 amostras, vira sinal aproveitável com 1000. Lição: distinção informacional binária ("é detectável" vs "não é") esconde uma escala. Com dados suficientes, sinais sutis viram aproveitáveis.

## Decisões metodológicas relevantes

- **Split estratificado por tipo de ataque**, não só por label — garante que cada split tem amostras de cada categoria (benign + 4 ataques) na proporção correta. Sem isso, splits desbalanceados podem mascarar problemas de generalização.
- **Threshold tunado in-sample no `detector.py`** — sweep no eval set, otimista. Mantido por simplicidade; ganho marginal (~+0.01 F1).
- **MLP+features venceu DistilBERT por motivos estruturais**: tarefa numérica + dataset pequeno + features explícitas. DistilBERT é ferramenta errada; tokens-de-bins perdem ordinalidade.
- **Pretreino traz +0.03 F1 em regime amplo (1000 amostras)** mas é crítico em regime escasso (200). Em regime amplo, features estatísticas já discriminam bem mesmo sobre random init.
- **Limite informacional real**: noise com SNR alto contra benign treinado. Aproximadamente 80–92% recall máxima nessa categoria — não há feature pra capturar.

## Fase 7 — integração com FL real (MONZA / PFLlib)

Até a Fase 6, todo o pipeline rodava em **dataset sintético**: gerador local no notebook (`BertModelsclassify.ipynb`) instanciava `FedAvgCNN` random ou pretrained, aplicava ataques sintéticos, e o detector aprendia. O threat model era cuidadoso, mas a *origem* dos pesos era um bench acadêmico — nada de cliente FL real treinando em data não-IID.

A pergunta natural que faltava: **o detector treinado em pesos sintéticos converge num cenário FL real?** Vale como defesa em produção, ou era artefato da geração?

Integramos com [`PFLlibMonza`](https://github.com/VeigarGit/PFLlibMonza) — fork do PFLlib (FL simulator) com ataques `zero/random/shuffle/label_flip`. Pipeline novo:

1. MONZA roda FL real com 100 clientes Dirichlet non-IID (alpha=0.1) sobre MNIST, 30 maliciosos atacando todo round, 50 rounds. Cada update é dumpado.
2. `detector.py` e `detector_mlp.py` treinam sobre esse dataset (5100 amostras).
3. Detector treinado é carregado de volta no servidor MONZA num novo `cc==6` (NLP) e `cc==7` (MLP), filtrando clientes maliciosos antes de `aggregate_parameters()`.
4. Pós-Fase 7, foram adicionados `cc==8` (MLP+validação pública), `cc==9` (DistilBERT+MLP+label-flip check) e `cc==10` (DistilBERT+MLP+targeted label-flip com delta global-cliente). `cc==10` é o experimento focado em melhorar `malicious_label`.

### Achados

**No treino dos detectores** (eval set in-sample):

| Métrica | NLP (DistilBERT+LoRA) | MLP (60 features) |
|---|---|---|
| F1 | 0.83 | **0.85** |
| Precision | 0.97 | **1.00** |
| Recall | 0.73 | 0.74 |

Bate com a banda esperada do bench Fase 5 (DistilBERT plateia em ~0.88, MLP ~0.99 em variantes leakage; aqui em MNIST FL real ficaram 0.83/0.85 — coerente com cenário mais realista que pretrained_hard).

**Em produção (defesa rolando, média últimos 30 dos 50 rounds)**:

| cc | Defesa | FPR | FRR | Score (FPR+FRR) |
|---|---|---:|---:|---:|
| 2 | Cluster cosseno (PFLlib baseline) | 0.000 | 0.262 | 0.262 |
| 3 | Cosseno + score (PFLlib baseline) | 0.053 | 0.114 | 0.168 |
| 6 | NLP DistilBERT | 0.112 | 0.114 | 0.226 |
| **7** | **MLP + features** | **0.000** | **0.156** | **0.156** 🏆 |

**MLP+features Pareto-supera os 2 baselines do PFLlib em produção FL real**: zero falsos positivos (não pune benignos) e melhor FRR que cluster.

### Insights novos (não redundantes com Fases 1–6)

**1. Distribution shift treino → produção é menor pro MLP que pro NLP.** O MLP eval F1=0.85 traduz pra FPR=0%/FRR=15.6% em produção. O NLP eval F1=0.83 *mas* em produção tem FPR=11% — tem 8× mais falsos positivos por round. Hipótese: features handcrafted (SVD, FFT, momentos) capturam invariantes estatísticos *de uma distribuição de pesos plausível*; tokens-de-bins do DistilBERT memoriza muito mais a distribuição empírica do training set, que diverge dos pesos novos a cada round.

**2. Label flip exige sinal de comportamento e rerun com `train_mal/` correto.** Os detectores por fingerprint (DistilBERT vs MLP+features SVD/FFT) chegaram a ~5% recall em label flip no run histórico, mas depois encontramos bug em `utils/data_utils.py`: clientes maliciosos de label flip liam `test/` em vez de `train_mal/`. Então o número antigo é só histórico. A hipótese técnica permanece: label flip pode manter pesos estatisticamente parecidos com benign, mas otimiza função errada. Detecção precisa outro paradigma — validation hold-out, margem para rótulo invertido, gradient-based, ou comparação cross-round. `cc=10` adiciona um score direcionado para reverse label flip usando holdout limpo e delta da camada final contra o modelo global.

**3. Defesa não precisa pegar 100% — só baixar a fração de poison abaixo do threshold de tolerância do FedAvg.** No `cc=5` (sem defesa, gerou o dataset), modelo nunca convergiu (best acc 0.12 com 30% poison todo round). Com cc=7 filtrando, fração de poison cai pra ~7%, e MNIST/FedAvgCNN converge sem problema. *Lição prática*: F1 do detector não é a métrica final — FPR e FRR em produção, e como elas interagem com o algoritmo de agregação, é o que importa.

**4. Bug do LoRA (descoberto na sessão MONZA): `modules_to_save` é necessário pra classification heads.** `LoraConfig(target_modules=['q_lin','v_lin'])` salva só as matrizes A/B do LoRA — `pre_classifier` e `classifier` voltam como inicialização aleatória ao carregar o adapter. Detector NLP rodou efetivamente com head random no primeiro deploy (FPR=1.0, marcou todos os 100 clientes como maliciosos no round 1). Fix: incluir o head em `modules_to_save`. Lição transferível para qualquer LoRA com tarefa nova de classificação/regressão.

**5. MLP é Pareto-melhor agora também em produção.** Confirma e reforça o achado da Fase 5: MLP+features 30× mais rápido pra treinar (~30s vs 15min), 3500× menor (~80KB vs 280MB), e com FPR=0% vs 11% do NLP em FL real. NLP só faria sentido se generalizasse melhor pra arquiteturas não vistas — não testamos por restrição de disco (VGG/Cifar10 exigiria ~280GB de state_dicts dumpados).

### Caveats herdados e novos

- Eval in-sample (otimista) — caveat herdado da Fase 5.
- Histórico: `model_zeros` no MONZA usava `torch.ones`, não `torch.zeros` (`attack.py:14`) — detector aprendeu a categoria errada. O pipeline atual zera de verdade; essa categoria precisa de novo dump e retreino antes de comparar.
- `model_noise` bugado no MONZA (`attack.py:52` early return) — `-atk all` cobre 4 categorias, não 5.
- Single seed (42) — sem CI nos números.
- VGG/Cifar10 cortado por disco — sem evidência de generalização além de FedAvgCNN/MNIST.
- `cc=8/9/10` ainda sem métrica fechada pós-correção de `train_mal` — não comparar como vencedores antes de novo run.

Detalhes em [`MONZA_RESULTS.md`](MONZA_RESULTS.md). Análise visual em [`notebook_monza_analysis.ipynb`](notebook_monza_analysis.ipynb).

## Fora do escopo (futuro possível)

- **Krum / Multi-Krum / FoolsGold** — defesas que comparam updates entre clientes em vez de classificar isoladamente. Diferente paradigma; complementar.
- **Ataques mais sofisticados**: backdoor, label flipping (já incluído mas não detectado), model-poisoning direcionado. O baseline atual cobre 4–5 ataques sintéticos; expansão é direta.
- **Arquiteturas maiores** (ResNet, transformer) — `features.py` precisaria adaptar `LAYERS` (já tolerante a `BaseHeadSplit` com `_resolve_layers`). Funções permanecem válidas.
- **Contrastive learning** entre rounds consecutivos — pega ataques on-off (cliente honesto por X rondas, ataca em Y).
- **Anchor-based defenses** com modelo pré-treinado público como referência.
- **Ensemble cc=2 + cc=7** — cluster filtra grosseiros, MLP filtra residuais. Pode ter FPR+FRR menor que ambos individualmente.
- **Validar cc=8/9/10** — medir se validação pública, label-flip check e targeted label-flip reduzem FRR em label flip sem perder o FPR=0% do cc=7.

## Lições

1. **F1=1.0 quase sempre indica leakage** ou benchmark inadequado. Desconfie sempre.
2. **Honestidade de threat model importa mais que sofisticação do modelo**. F1=0.99 no realista vale mais que F1=1.0 no leakage.
3. **Reaproveitar engenharia clássica de features** quando aplicável. SVD, FFT, autocorr resolveram problemas que DistilBERT não consegue por design.
4. **Distinções informacionais binárias mentem**. Tudo é questão de SNR vs número de amostras.
5. **Quando o paradigma errado, polir não resolve**. DistilBERT plateia em 0.88 independente do tuning. Feature engineering + MLP simples sobe pra 0.99 sem sweep.
6. **F1 do detector não é a métrica final em FL**. FPR e FRR em produção, e como interagem com a tolerância do algoritmo de agregação, é o que define se a defesa serve. F1=0.83 com FPR=0% (cc=7) é melhor que F1=0.83 com FPR=11% (cc=6) na prática.
7. **Distribution shift treino → produção penaliza modelos overparametrizados**. DistilBERT (66M params) overfita à distribuição empírica do training set. MLP+features (13k params) opera sobre invariantes estatísticos e generaliza melhor. Em FL real, o detector vê pesos que o eval set nunca viu.
8. **LoRA com classification heads exige `modules_to_save` explícito**. Default só salva matrizes A/B das atenções; o head treinado fica órfão. Bug silencioso — F1 in-memory não bate com F1 ao recarregar.
9. **Label flip precisa sinal de comportamento, não só sinal de pesos**. `cc=8/9/10` adicionam validação pública, direção da camada final, margem para rótulo invertido e score direcionado por delta global-cliente para observar efeito do update no holdout limpo.
