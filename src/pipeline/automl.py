from dataclasses import dataclass
from autogluon.tabular import TabularPredictor
import pandas as pd

from src.models.automl import MultilabelPredictor
from src.utils.split import multiclass_split, multilabel_split


@dataclass
class AutoGluonConfig:
    model_path: str
    label_columns: list[str] | None = None
    stratify_column: str = "insult"
    test_size: float = 0.35
    val_size: float = 15 / 35
    seed: int = 21
    presets: str = "medium_quality"
    time_limit: int = 900
    eval_metrics: list[str] | None = None


class PipelineAutoGluonMultilabel(MultilabelPredictor):
    def __init__(self, config: AutoGluonConfig, df: pd.DataFrame, problem_types: list[str] = None):
        self.config = config

        label_columns = config.label_columns if config.label_columns is not None else df.columns[1:-1].tolist()
        eval_metrics = config.eval_metrics if config.eval_metrics is not None else ["f1_macro"] * len(label_columns)
        
        if problem_types is None:
            problem_types = ["binary"] * len(label_columns)

        super().__init__(labels=label_columns, eval_metrics=eval_metrics, path=config.model_path, problem_types=problem_types)

    def split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        return multilabel_split(df, self.config.label_columns, self.config.test_size, self.config.val_size)

    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
        super().fit(
            train_data=train_df,
            tuning_data=val_df,
            presets=self.config.presets,
            time_limit=self.config.time_limit,
        )

    def evaluate(self, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        y_pred = super().predict(test_df)
        y_true = test_df[self.labels]
        return y_true, y_pred

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        return super().predict(df)


class PipelineAutoGluonBinary:
    def __init__(self, config: AutoGluonConfig, df: pd.DataFrame):
        self.config = config
        self.label = config.label_columns[0]
        eval_metric = config.eval_metrics[0]

        self.predictor = TabularPredictor(
            label=self.label,
            problem_type="binary",
            eval_metric=eval_metric,
            path=config.model_path,
        )

    def split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        return multiclass_split(df, self.label, self.config.test_size, self.config.val_size, self.config.seed)

    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
        self.predictor.fit(
            train_data=train_df,
            tuning_data=val_df,
            presets=self.config.presets,
            time_limit=self.config.time_limit,
        )

    def evaluate(self, test_df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        y_pred = self.predictor.predict(test_df)
        y_true = test_df[self.label]
        return y_true, y_pred

    def predict(self, df: pd.DataFrame) -> pd.Series:
        return self.predictor.predict(df)
