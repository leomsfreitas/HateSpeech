import json
import time
import requests
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from itertools import chain

from src.utils.split import multiclass_split


@dataclass
class PerspectiveConfig:
    api_key: str
    label_column: str
    threshold: float = 0.4
    chunk_size: int = 600
    sleep_interval: int = 60
    attributes: list[str] = field(default_factory=lambda: ["TOXICITY"])
    test_size: float = 0.35
    val_size: float = 15 / 35
    seed: int = 21


class PipelinePerspective:
    def __init__(self, config: PerspectiveConfig):
        self.config = config
        self.url = f"https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze?key={config.api_key}"

    def split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        return multiclass_split(df, self.config.test_size, self.config.val_size, self.config.seed)

    def _analyze(self, text: str) -> dict | None:
        payload = {
            "comment": {"text": text},
            "languages": ["pt"],
            "requestedAttributes": {attr: {} for attr in self.config.attributes},
        }
        try:
            response = requests.post(
                self.url,
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload),
            )
            response.raise_for_status()
            result = response.json()["attributeScores"]
            return {attr: result[attr]["summaryScore"]["value"] for attr in result}
        except Exception as e:
            print(f"Erro: {e}")
            return None

    def predict(self, df: pd.DataFrame, text_column: str) -> np.ndarray:
        scores_list = []
        for i in range(0, len(df), self.config.chunk_size):
            chunk = df[text_column].iloc[i:i + self.config.chunk_size]
            scores_list.append(chunk.apply(self._analyze))
            time.sleep(self.config.sleep_interval)
        flat = list(chain.from_iterable(scores_list))
        scores = pd.DataFrame(flat, index=df.index)
        return (scores[self.config.attributes[0]] >= self.config.threshold).astype(int).values