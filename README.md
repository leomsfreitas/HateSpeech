# Exploring Sentiment Analysis Techniques for Offensive Language Classification in Brazilian Politics

> Comparison of AutoML, BERT and LLM approaches for detecting offensive language and hate speech in tweets from the 2022 Brazilian presidential elections.

## Background

Social media has become a central space for contemporary political debate, enabling the
rapid circulation of opinions, information and electoral strategies. This expansion came
hand in hand with a sharp rise in offensive comments and hate speech — insults, racism,
xenophobia and other forms of verbal violence that degrade public discourse and deepen
political polarization.

Although incitement to hatred is a crime in Brazil, its enforcement on social platforms
remains limited. Investigating the spread of online offensive language is therefore
essential to understanding how interaction, power and conflict dynamics manifest in
digital environments.

This project investigates the use of **sentiment analysis (SA)** techniques for the
automatic classification of offensive language and hate speech in tweets about Brazilian
politics. The guiding hypothesis is that offensive comments tend to express negative
sentiments and emotions, and that the presence of negative terms and emotions can
therefore support the classification of offensive language. The work was developed within
the **Interfaces** research group (UFSCar), as part of a regular FAPESP project
(2022/03090-0) aimed at the large-scale analysis of Brazilian political data.

Two classification tasks are addressed:

1. **Binary classification** — whether a comment is offensive or not.
2. **Multi-label classification** — which type(s) of offense a comment carries, including
   hate-speech categories.

## Corpus

The target corpus, **ITED-BR**, is a reference set of **3,400 tweets** manually annotated
for this research, drawn from the [Interfaces Twitter Elections Dataset](https://doi.org/10.1371/journal.pone.0317289)
(Iasulaitis et al., 2025) — a collection of ~228 million tweets from the 2022 Brazilian
presidential elections. The 3,400 tweets relate to the **second round** of the election
(1,700 mentioning Bolsonaro and 1,700 mentioning Lula), published between October 2 and 30,
2022.

Annotation was carried out by nine researchers in two stages: first labeling each comment
as offensive/non-offensive, then assigning one or more offense types to offensive comments.
Inter-annotator agreement (Krippendorff's Alpha) reached **0.68** (Lula set) and **0.72**
(Bolsonaro set).

**Class distribution (ITED-BR).** Offense-type percentages are computed over the offensive
comments of each set; a comment may carry more than one type.

| Offense | Bolsonaro set | Lula set | Total |
|---|---|---|---|
| Insult | 1,188 (95.4%) | 1,036 (96.2%) | 2,224 (95.8%) |
| Ideology | 244 (20.5%) | 193 (18.0%) | 437 (18.8%) |
| Obscene | 106 (8.9%) | 70 (6.5%) | 176 (7.6%) |
| Others | 24 (1.4%) | 14 (0.8%) | 38 (1.1%) |
| Xenophobia | 12 (1.1%) | 5 (0.4%) | 17 (0.7%) |
| Sexism | 11 (0.9%) | 5 (0.4%) | 16 (0.7%) |
| LGBTQphobia | 8 (0.7%) | 0 (0.0%) | 8 (0.3%) |
| Racism | 3 (0.2%) | 1 (0.09%) | 4 (0.2%) |
| **Offensive** | **1,245 (73.2%)** | **1,076 (63.3%)** | **2,321 (68.3%)** |
| **Non-offensive** | **455 (26.8%)** | **624 (36.7%)** | **1,079 (31.7%)** |

**Training corpora.** Models are trained on public Portuguese corpora before being applied
to ITED-BR. The binary models use **Hate-BR** (Vargas et al., 2022; 7k Instagram comments).
The multi-label models use **Unified-Hate**, built by merging **TOLD-BR**, **OLID-BR** and
**Hate-BR** under a common label scheme — **15,555 comments** after de-duplication.

| Class | TOLD-BR | OLID-BR | Hate-BR | Total | Original classes |
|---|---|---|---|---|---|
| Insult | 4,371 | 5,589 | — | 9,864 (63.4%) | Insult |
| Obscene | 6,639 | 2,311 | — | 8,819 (56.7%) | Obscene, Profanity |
| Ideology | — | 1,221 | 496 | 1,704 (10.9%) | Ideology, Partyism |
| LGBTQphobia | 344 | 375 | 17 | 731 (4.7%) | LGBTQ+phobia, Homophobia |
| Racism | 138 | 92 | 8 | 236 (1.5%) | Racism |
| Sexism | 463 | 461 | 97 | 1,011 (6.5%) | Misogyny, Sexism |
| Xenophobia | 151 | 128 | 1 | 279 (1.8%) | Xenophobia |

## Approaches Evaluated

### AutoML (AutoGluon)

Traditional machine-learning models trained with [AutoGluon](https://github.com/autogluon/autogluon),
combined with sentiment-analysis features. To measure each feature's contribution, an
**incremental evaluation** is performed — starting from a feature-less *baseline* and adding
one attribute at a time, yielding six configurations:
`Baseline → MOL → MOL+GE → MOL+GE+GEP → MOL+GE+GEP+Lex → MOL+GE+GEP+Lex+IP`.

The features are:

- **MOL** — *Multilingual Offensive Lexicon* (Vargas et al., 2021): weighted score of
  offensive terms found in the text.
- **GE** — *GoEmotions* probability vector (28 emotions) from a BERTimbau model fine-tuned
  for emotion analysis (Hammes & Freitas, 2021).
- **GEP** — *GoEmotions polarity*: emotions above a 0.3 threshold mapped to +1/0/-1
  (Assi et al., 2022) and summed.
- **Lex** — proportions of positive/negative/neutral terms using three sentiment lexicons
  (LIWC-Br, WordNetAffectBR, SentiLex-PT).
- **IP** — a single *polarity index* combining the three Lex proportions via logistic
  regression.

### BERTimbau

The [BERTimbau Large](https://huggingface.co/neuralmind/bert-large-portuguese-cased)
Transformer (Souza et al., 2020), fine-tuned separately for the binary and multi-label
tasks. Hyperparameters: batch size 32, 3 epochs, learning rate 2e-5, weight decay 0.01,
warmup ratio 0.1, max sequence length 128, label smoothing 0.1, early-stopping patience 2.

### Sabiá-3 (LLM)

A generative large language model trained specifically for Brazilian Portuguese
([Maritaca AI](https://www.maritaca.ai/)). It is prompted with task descriptions; a
**few-shot** strategy outperformed zero-shot on both tasks and was the one adopted.

### Perspective API

The [Perspective API](https://perspectiveapi.com/) (Jigsaw/Google), used **only for the
binary task** via its `TOXICITY` attribute. Comments scoring ≥ **0.4** (threshold defined
empirically) are classified as offensive.

## Results

Summary on the **ITED-BR** corpus (Macro-F). Full per-class metrics and the training-stage
results are in **[results/README.md](results/README.md)**.

| Approach | Binary Macro-F | Multi-label Macro-F | Compute cost |
|---|---|---|---|
| **BERTimbau** | **0.75** | 0.28 | High (GPU fine-tuning) |
| **Sabiá-3 (LLM)** | 0.73 | **0.58** | API (paid inference) |
| AutoML — best config | 0.69 | 0.55 | Low (CPU) |
| AutoML — Baseline | 0.63 | 0.55 | Low (CPU) |
| Perspective API | 0.56 | — | API |

**Key findings.** In the **binary** task, the fine-tuned BERTimbau leads (0.75), followed
closely by the Sabiá-3 LLM (0.73); the much simpler and cheaper AutoML models reach 0.69,
trailing by only 4–6 points. In the **multi-label** task the LLM leads (0.58), with the
AutoML Baseline right behind (0.55), while BERTimbau collapses to 0.28 — generalizing only
on the majority *insult* class. Sentiment-analysis features helped most in the binary task
(+6 points over baseline), supporting the project's initial hypothesis. Rare classes
(racism, xenophobia) remained the hardest to detect across all approaches.

## Repository Structure

```
.
├── data/
│   ├── ited-br/              # target corpus: 3,400 annotated election tweets
│   │   ├── processed/        # cleaned ited-br.csv
│   │   └── features/         # mol, ge, gep, lex, ip feature tables
│   ├── hate-br/              # training corpus — binary task
│   ├── unified-hate/         # training corpus — multi-label task (TOLD+OLID+Hate)
│   ├── told-br/, olid-br/    # source corpora for Unified-Hate
│   └── lexicons/             # LIWC-Br, SentiLex-PT, WordNetAffectBR, MOL
├── notebooks/
│   ├── 1_analysis.ipynb              # exploratory data analysis
│   ├── 2a_filtered_corpus.ipynb      # filter & sample raw election tweets
│   ├── 2b_pre_processing.ipynb       # clean/normalize all corpora
│   ├── 2c_unified_corpus.ipynb       # merge into Unified-Hate
│   ├── 3a_feat_lex.ipynb             # Lex feature
│   ├── 3b_feat_ge.ipynb              # GoEmotions (GE) feature
│   ├── 3c_feat_gep.ipynb             # GoEmotions polarity (GEP) feature
│   ├── 3d_feat_ip.ipynb              # polarity index (IP) feature
│   ├── 3e_feat_mol.ipynb             # MOL feature
│   ├── 4a_automl_binary.ipynb        # train AutoML — binary (Hate-BR)
│   ├── 4a_automl_multilabel.ipynb    # train AutoML — multi-label (Unified-Hate)
│   ├── 4b_bert.ipynb                 # fine-tune BERTimbau — both tasks
│   ├── 5a_automl_inference.ipynb     # AutoML inference on ITED-BR
│   ├── 5b_bert_inference.ipynb       # BERTimbau inference on ITED-BR
│   ├── 5c_sabia.ipynb                # Sabiá-3 inference on ITED-BR
│   ├── 5d_perspective.ipynb          # Perspective API inference on ITED-BR
│   └── others/                       # auxiliary lexicon/feature ETL notebooks
├── src/
│   ├── pipeline/   # automl.py, bert.py, sabia.py, perspective.py
│   ├── models/     # automl.py (MultilabelPredictor), bert.py
│   └── utils/      # features, report, split, preprocessing, prompt, map, path
├── models/         # trained AutoGluon and BERTimbau models
├── results/        # detailed metrics (see results/README.md)
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone https://github.com/leomsfreitas/HateSpeech.git
cd HateSpeech
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## How to Run

The notebooks are numbered in execution order. A typical end-to-end run is:

1. **Pre-processing** — `2b_pre_processing.ipynb`, then `2c_unified_corpus.ipynb`.
2. **Feature extraction** — `3a_feat_lex` → `3b_feat_ge` → `3c_feat_gep` → `3d_feat_ip` →
   `3e_feat_mol`.
3. **Training** — `4a_automl_binary`, `4a_automl_multilabel`, `4b_bert`.
4. **Inference on ITED-BR** — `5a_automl_inference`, `5b_bert_inference`, `5c_sabia`,
   `5d_perspective`.

> The Sabiá-3 and Perspective API notebooks require API keys (Maritaca AI and Jigsaw/Google,
> respectively).

## Available Models

The trained models are too large to host on GitHub and are distributed separately. Only the
fine-tuned BERTimbau models and the best-performing AutoML configuration per task are
provided.

| Model | Task | Best config | ITED-BR Macro-F | Download |
|---|---|---|---|---|
| BERTimbau | Binary | — | 0.75 | _(Google Drive link coming soon)_ |
| BERTimbau | Multi-label | — | 0.28 | _(Google Drive link coming soon)_ |
| AutoML | Binary | MOL+GE | 0.69 | _(Google Drive link coming soon)_ |
| AutoML | Multi-label | Baseline | 0.55 | _(Google Drive link coming soon)_ |

## Citation

```bibtex
@misc{freitas2026ofensivo,
  title  = {Explorando Técnicas de Análise de Sentimento na Classificação de
            Linguagem Ofensiva no Contexto da Política Brasileira},
  author = {Freitas, Leo Marques Sabino de},
  year   = {2026},
  url    = {https://github.com/leomsfreitas/HateSpeech}
}
```

## Authors

Leo Marques Sabino de Freitas — [@leomsfreitas](https://github.com/leomsfreitas)

## Acknowledgments

This research was developed within the **Interfaces** research group (UFSCar) under the
supervision of **Eloize Rossi Marques Seno**, as part of a regular **FAPESP** project
(2022/03090-0). The author thanks **PACTEC/IFSP** for the research scholarship that made
this work possible, and **Maritaca AI** for the credits granted for the use of the Sabiá-3
LLM.

---

> The author declares the use of the **Opus 4.8** model to generate the README files of this
> project.
