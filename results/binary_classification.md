# Binary Classification — Offensive vs. Non-offensive

Task: classify each comment as **offensive** or **non-offensive**. Models are trained on
**Hate-BR** and then applied to the **ITED-BR** election corpus.

Metrics: **P** = precision, **C** = recall, **F1** = F1-score, **Macro-F** = macro-averaged
F1.

## Inference on ITED-BR

Models ranked by Macro-F. Both candidate sets (Lula and Bolsonaro) were merged into a single
evaluation set.

| Model | Non-offensive P | C | F1 | Offensive P | C | F1 | Macro-F |
|---|---|---|---|---|---|---|---|
| **BERTimbau** | 0.64 | 0.70 | 0.67 | 0.85 | 0.81 | 0.83 | **0.75** |
| Sabiá-3 | 0.59 | 0.71 | 0.65 | 0.85 | 0.77 | 0.81 | 0.73 |
| AutoML — MOL+GE | 0.54 | 0.67 | 0.60 | 0.83 | 0.74 | 0.78 | 0.69 |
| AutoML — MOL+GE+GEP+Lex | 0.54 | 0.67 | 0.60 | 0.83 | 0.74 | 0.78 | 0.69 |
| AutoML — MOL+GE+GEP+Lex+IP | 0.55 | 0.65 | 0.60 | 0.82 | 0.75 | 0.78 | 0.69 |
| AutoML — MOL+GE+GEP | 0.53 | 0.68 | 0.60 | 0.83 | 0.72 | 0.77 | 0.68 |
| AutoML — Baseline | 0.47 | 0.62 | 0.53 | 0.79 | 0.67 | 0.73 | 0.63 |
| AutoML — MOL | 0.45 | 0.70 | 0.55 | 0.81 | 0.61 | 0.70 | 0.62 |
| Perspective API | 0.37 | 0.93 | 0.53 | 0.94 | 0.42 | 0.59 | 0.56 |

**Discussion.**

- The fine-tuned **BERTimbau** achieved the best Macro-F (0.75), closely followed by the
  **Sabiá-3** LLM (0.73) — only 2 points behind.
- The **AutoML** models combining MOL with sentiment-analysis features reached 0.68–0.69,
  trailing the advanced models by just 4–6 points despite being far simpler and cheaper to
  run.
- The largest AutoML gain came from adding **GoEmotions (GE)** to the MOL-only model: +7
  points (0.62 → 0.69). Further features (GEP, Lex, IP) left results virtually unchanged.
- Used in isolation, the **MOL** feature *degraded* performance relative to the Baseline
  (0.62 vs. 0.63).
- The **Perspective API** ranked lowest (0.56). Its 0.4 TOXICITY threshold produced the
  highest precision on the offensive class (0.94) and the highest recall on the
  non-offensive class (0.93), but severely hurt offensive-class recall (0.42) and
  non-offensive precision (0.37).

## Training on Hate-BR

Reference performance on the Hate-BR test split (15% of the corpus), before applying the
models to ITED-BR.

### AutoML (incremental feature configurations)

| Model | Non-offensive P | C | F1 | Offensive P | C | F1 | Macro-F |
|---|---|---|---|---|---|---|---|
| Baseline | 0.85 | 0.91 | 0.88 | 0.90 | 0.83 | 0.86 | **0.87** |
| MOL+GE+GEP+Lex+IP | 0.85 | 0.88 | 0.87 | 0.87 | 0.84 | 0.86 | 0.86 |
| MOL+GE | 0.84 | 0.88 | 0.86 | 0.86 | 0.83 | 0.85 | 0.85 |
| MOL+GE+GEP+Lex | 0.85 | 0.88 | 0.86 | 0.87 | 0.84 | 0.85 | 0.85 |
| MOL+GE+GEP | 0.84 | 0.87 | 0.86 | 0.86 | 0.83 | 0.85 | 0.85 |
| MOL | 0.76 | 0.81 | 0.79 | 0.80 | 0.74 | 0.77 | 0.78 |

On Hate-BR there is little variation across feature configurations, suggesting the
sentiment-analysis features contributed little over the feature-less baseline in-domain.
As in the ITED-BR results, the MOL feature applied alone degraded performance across all
metrics.

### BERTimbau

| Class | P | C | F1 | Support |
|---|---|---|---|---|
| Non-offensive | 0.92 | 0.93 | 0.92 | 524 |
| Offensive | 0.92 | 0.92 | 0.92 | 514 |
| **Macro-F** | | | **0.92** | 1,038 |
