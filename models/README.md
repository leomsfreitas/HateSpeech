# Models

This folder is where the classification models are saved during training. Running the
training notebooks (`4a_automl_*`, `4b_bert`) writes the trained predictors here:

```
models/
├── automl/                              # AutoGluon predictors (binary/, multilabel/)
├── bert/                                # fine-tuned BERTimbau (binary/, multilabel/)
└── BERTimbau_large_GoEmotions_portuguese/   # GoEmotions model (see below)
```

The trained models are not versioned in git (see the repository `.gitignore`), as they are
too large for GitHub.

## Required: GoEmotions Portuguese

The feature-extraction notebooks `3b_feat_ge` and `3c_feat_gep` depend on the **GoEmotions
Portuguese** model — a BERTimbau model fine-tuned for emotion analysis in Portuguese.

Download it from the author's repository and place it in this folder under
`BERTimbau_large_GoEmotions_portuguese/`:

**https://github.com/Luzo0/GoEmotions_portuguese**

The expected local path is `./models/BERTimbau_large_GoEmotions_portuguese`, as referenced
in [`src/utils/path.py`](../src/utils/path.py). Without this model, the GE and GEP feature
notebooks will not run.
