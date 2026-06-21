# Results

Detailed evaluation results for all approaches, on both the **training corpora** (Hate-BR
for binary, Unified-Hate for multi-label) and the **target corpus ITED-BR** (3,400 annotated
tweets from the 2022 Brazilian presidential elections). ITED-BR is the central evaluation
target of the project.

Metrics: **P** = precision, **C** = recall, **F1** = F1-score, **Macro-F** = macro-averaged
F1 across classes.

## Contents

- [Binary classification](binary_classification.md) — offensive vs. non-offensive.
- [Multi-label classification](multilabel_classification.md) — seven offense types.

## Overview — ITED-BR (Macro-F)

| Approach | Binary | Multi-label | Compute cost |
|---|---|---|---|
| BERTimbau | **0.75** | 0.28 | High (GPU fine-tuning) |
| Sabiá-3 (LLM) | 0.73 | **0.58** | API (paid inference) |
| AutoML — MOL+GE | 0.69 | 0.54 | Low (CPU) |
| AutoML — MOL+GE+GEP+Lex | 0.69 | 0.54 | Low (CPU) |
| AutoML — MOL+GE+GEP+Lex+IP | 0.69 | 0.53 | Low (CPU) |
| AutoML — MOL+GE+GEP | 0.68 | 0.54 | Low (CPU) |
| AutoML — Baseline | 0.63 | 0.55 | Low (CPU) |
| AutoML — MOL | 0.62 | 0.55 | Low (CPU) |
| Perspective API | 0.56 | — | API |

## Highlights

- **Binary task.** Best result with the fine-tuned **BERTimbau** (0.75 Macro-F), followed by
  the **Sabiá-3** LLM (0.73). Traditional AutoML models — far simpler and cheaper — reach
  0.69, only 4–6 points behind the advanced models. The **Perspective API** scored lowest
  (0.56): its 0.4 TOXICITY threshold maximized precision on offensive comments (0.94) but
  badly hurt recall on the offensive class (0.42).
- **Multi-label task.** Best result with **Sabiá-3** (0.58), with the AutoML **Baseline**
  right behind (0.55). **BERTimbau** collapsed to 0.28, generalizing only on the majority
  *insult* class (F1 0.80). Adding the MOL lexicon to AutoML strongly improved *insult*
  (F1 0.41 → 0.45, recall 0.51 → 0.99) but confused the model on the other offense types.
- **Feature contribution.** Sentiment-analysis features were most useful in the **binary**
  task: adding GoEmotions (GE) to MOL gave a +7-point Macro-F gain (0.62 → 0.69), supporting
  the hypothesis that offensive comments are inherently negative in sentiment.
- **Rare classes.** Racism and xenophobia remained the hardest categories to detect across
  all approaches, reflecting their scarcity in the data.
