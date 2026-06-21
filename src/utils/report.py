from sklearn.metrics import classification_report
import pandas as pd
import numpy as np

def multilabel_report(y_true, y_pred, label_names=None):
    if isinstance(y_true, np.ndarray):
        n_labels = y_true.shape[1]
        cols = label_names if label_names is not None else [str(i) for i in range(n_labels)]
        y_true = pd.DataFrame(y_true, columns=cols)
        y_pred = pd.DataFrame(y_pred, columns=cols)

    rows = []
    empty = {"precision": 0.0, "recall": 0.0, "f1-score": 0.0, "support": 0}

    for label in y_true.columns:
        report = classification_report(
            y_true[label],
            y_pred[label],
            labels=[0, 1],
            output_dict=True,
            zero_division=0
        )

        for cls in [0, 1]:
            stats = report.get(cls, report.get(str(cls), empty))
            rows.append({
                "label": label,
                "class": cls,
                "support": stats["support"],
                "precision": stats["precision"],
                "recall": stats["recall"],
                "f1": stats["f1-score"],
            })

        for avg in ["macro avg", "weighted avg"]:
            stats = report.get(avg, empty)
            rows.append({
                "label": label,
                "class": avg.split()[0],
                "support": stats["support"],
                "precision": stats["precision"],
                "recall": stats["recall"],
                "f1": stats["f1-score"],
            })

    return pd.DataFrame(rows)