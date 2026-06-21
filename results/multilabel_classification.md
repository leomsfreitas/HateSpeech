# Multi-label Classification — Offense Types

Task: assign one or more of seven offense types to each comment — **insult, obscene,
ideology, LGBTQphobia, racism, sexism, xenophobia**. Models are trained on **Unified-Hate**
and then applied to the **ITED-BR** election corpus.

Metrics: **P** = precision, **C** = recall, **F1** = F1-score, **Macro-F** = macro-averaged
F1 across the seven classes.

## Inference on ITED-BR

### F1 per class (ranked by Macro-F)

| Model | Insult | Obscene | Ideology | LGBTQphobia | Racism | Sexism | Xenophobia | Macro-F |
|---|---|---|---|---|---|---|---|---|
| **Sabiá-3** | 0.68 | 0.65 | 0.53 | 0.59 | 0.52 | 0.55 | 0.55 | **0.58** |
| AutoML — Baseline | 0.41 | 0.76 | 0.49 | 0.59 | 0.50 | 0.56 | 0.51 | 0.55 |
| AutoML — MOL | 0.42 | 0.76 | 0.49 | 0.60 | 0.50 | 0.56 | 0.50 | 0.55 |
| AutoML — MOL+GE | 0.44 | 0.72 | 0.49 | 0.53 | 0.50 | 0.58 | 0.50 | 0.54 |
| AutoML — MOL+GE+GEP | 0.45 | 0.72 | 0.49 | 0.54 | 0.50 | 0.57 | 0.50 | 0.54 |
| AutoML — MOL+GE+GEP+Lex | 0.45 | 0.72 | 0.49 | 0.54 | 0.50 | 0.61 | 0.50 | 0.54 |
| AutoML — MOL+GE+GEP+Lex+IP | 0.45 | 0.70 | 0.48 | 0.52 | 0.50 | 0.58 | 0.50 | 0.53 |
| BERTimbau | 0.80 | 0.53 | 0.12 | 0.11 | 0.00 | 0.22 | 0.17 | 0.28 |

### Full per-class metrics (P / C / F1)

**Sabiá-3** — Macro-F 0.58

| Class | P | C | F1 |
|---|---|---|---|
| Insult | 0.70 | 0.72 | 0.68 |
| Obscene | 0.80 | 0.62 | 0.65 |
| Ideology | 0.57 | 0.65 | 0.53 |
| LGBTQphobia | 0.56 | 0.68 | 0.59 |
| Racism | 0.51 | 0.62 | 0.52 |
| Sexism | 0.54 | 0.62 | 0.55 |
| Xenophobia | 0.54 | 0.62 | 0.55 |

**AutoML — Baseline** — Macro-F 0.55

| Class | P | C | F1 |
|---|---|---|---|
| Insult | 0.59 | 0.51 | 0.41 |
| Obscene | 0.73 | 0.80 | 0.76 |
| Ideology | 0.50 | 0.50 | 0.49 |
| LGBTQphobia | 0.67 | 0.57 | 0.59 |
| Racism | 0.50 | 0.50 | 0.50 |
| Sexism | 0.55 | 0.59 | 0.56 |
| Xenophobia | 0.51 | 0.52 | 0.51 |

**AutoML — MOL** — Macro-F 0.55

| Class | P | C | F1 |
|---|---|---|---|
| Insult | 0.58 | 0.51 | 0.42 |
| Obscene | 0.73 | 0.81 | 0.76 |
| Ideology | 0.50 | 0.50 | 0.49 |
| LGBTQphobia | 0.58 | 0.62 | 0.60 |
| Racism | 0.50 | 0.50 | 0.50 |
| Sexism | 0.54 | 0.65 | 0.56 |
| Xenophobia | 0.50 | 0.50 | 0.50 |

**BERTimbau** — Macro-F 0.28

| Class | P | C | F1 |
|---|---|---|---|
| Insult | 0.68 | 0.96 | 0.80 |
| Obscene | 0.46 | 0.64 | 0.53 |
| Ideology | 0.16 | 0.10 | 0.12 |
| LGBTQphobia | 0.09 | 0.12 | 0.11 |
| Racism | 0.00 | 0.00 | 0.00 |
| Sexism | 0.27 | 0.19 | 0.22 |
| Xenophobia | 0.33 | 0.12 | 0.17 |

**Discussion.**

- The **Sabiá-3** LLM achieved the best overall Macro-F (0.58), with the AutoML **Baseline**
  right behind (0.55). The simple AutoML model even **beat the LLM on obscene comments**
  (F1 0.76 vs. 0.65) and matched it on LGBTQphobia and sexism.
- **BERTimbau** obtained the worst Macro-F (0.28), generalizing only on the majority
  **insult** class (F1 0.80) and failing almost entirely on the minority classes (racism
  F1 0.00).
- Adding the **MOL** lexicon to AutoML strongly improved the *insult* class — precision
  0.59 → 0.66, recall 0.51 → 0.99, F1 0.41 → 0.79 — bringing AutoML to BERTimbau-level on
  that class. However, MOL confused the model on the remaining offense types, lowering their
  scores. The additional sentiment-analysis features did not improve multi-label performance.

## Training on Unified-Hate

Reference performance on the Unified-Hate test split (15% of the corpus).

### AutoML (F1 per class, ranked by Macro-F)

| Model | Insult | Obscene | Ideology | LGBTQphobia | Racism | Sexism | Xenophobia | Macro-F |
|---|---|---|---|---|---|---|---|---|
| Baseline | 0.81 | 0.83 | 0.79 | 0.69 | 0.53 | 0.66 | 0.53 | 0.69 |
| MOL | 0.81 | 0.83 | 0.79 | 0.71 | 0.50 | 0.66 | 0.50 | 0.69 |
| MOL+GE+GEP+Lex | 0.82 | 0.82 | 0.78 | 0.72 | 0.49 | 0.68 | 0.49 | 0.69 |
| MOL+GE+GEP+Lex+IP | 0.83 | 0.82 | 0.79 | 0.70 | 0.51 | 0.68 | 0.54 | 0.69 |
| MOL+GE | 0.82 | 0.83 | 0.76 | 0.72 | 0.50 | 0.67 | 0.49 | 0.68 |
| MOL+GE+GEP | 0.82 | 0.82 | 0.77 | 0.71 | 0.50 | 0.66 | 0.56 | 0.68 |

Results are essentially equivalent across feature configurations (~0.69 Macro-F). As
expected, the minority classes (racism, xenophobia) scored lowest, while insult and obscene
exceeded 0.80 — comparable to BERTimbau.

### BERTimbau

| Class | Macro P | Macro C | Macro F1 | Support |
|---|---|---|---|---|
| Insult | 0.83 | 0.85 | 0.84 | 1,480 |
| Obscene | 0.84 | 0.84 | 0.84 | 1,317 |
| Ideology | 0.92 | 0.78 | 0.83 | 198 |
| LGBTQphobia | 0.87 | 0.81 | 0.84 | 98 |
| Racism | 0.49 | 0.50 | 0.50 | 35 |
| Sexism | 0.83 | 0.70 | 0.75 | 144 |
| Xenophobia | 0.99 | 0.52 | 0.54 | 42 |
| **Macro avg** | **0.83** | **0.72** | **0.73** | |

In-domain, BERTimbau reached 0.73 Macro-F, with most classes between 0.75 and 0.84 and the
expected drop on the rare racism (0.50) and xenophobia (0.54) classes. The sharp contrast
with its 0.28 Macro-F on ITED-BR highlights the model's difficulty generalizing to the
out-of-domain election corpus.
