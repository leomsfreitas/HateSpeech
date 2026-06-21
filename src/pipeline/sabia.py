import json
import time
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path
from openai import OpenAI
from sklearn.metrics import classification_report



@dataclass
class SabiaConfig:
    api_key: str
    label_columns: list[str]
    prompt=str,
    model: str = "sabia-3"
    max_tokens: int = 2
    temperature: float = 0
    workdir: str | None = None
    test_size: float = 0.2
    val_size: float = 0.1
    seed: int = 21


class PipelineSabiaBinary:
    def __init__(self, config: SabiaConfig):
        self.config = config
        self.label_column = config.label_columns[0]
        self.client = OpenAI(
            api_key=config.api_key,
            base_url="https://chat.maritaca.ai/api",
        )

    def _classify(self, text: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": self.config.prompt},
                    {"role": "user", "content": text},
                ],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"erro: {e}"

    def _compute_metrics(self, labels, preds) -> dict:
        report = classification_report(
            labels,
            preds,
            labels=[0, 1],
            output_dict=True,
            zero_division=0,
        )
        macro = report["macro avg"]
        return {
            f"{self.label_column}_precision": round(macro["precision"], 4),
            f"{self.label_column}_recall": round(macro["recall"], 4),
            f"{self.label_column}_f1": round(macro["f1-score"], 4),
            f"{self.label_column}_support": int(report["1"]["support"]),
        }

    def predict(self, df: pd.DataFrame, text_column: str) -> pd.Series:
        preds = df[text_column].apply(self._classify).astype(int)
        return pd.Series(preds, name=self.label_column, index=df.index)

    def evaluate(self, df: pd.DataFrame, text_column: str) -> dict:
        preds = self.predict(df, text_column)
        return self._compute_metrics(df[self.config.label_column].values, preds.values)


class PipelineSabiaMultilabel:
    def __init__(self, config: SabiaConfig):
        self.config = config
        self.client = OpenAI(
            api_key=config.api_key,
            base_url="https://chat.maritaca.ai/api",
        )

    def _build_jsonl(self, df: pd.DataFrame, text_column: str, output_path: str):
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for i, text in enumerate(df[text_column].astype(str)):
                row = {
                    "custom_id": f"tweet-{i}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": self.config.model,
                        "messages": [
                            {"role": "system", "content": self.config.prompt},
                            {"role": "user", "content": text},
                        ],
                        "max_tokens": self.config.max_tokens,
                        "temperature": self.config.temperature,
                    },
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _create_batch(self, input_path: str) -> str:
        with open(input_path, "rb") as f:
            file = self.client.files.create(file=f, purpose="batch")
        batch = self.client.batches.create(
            input_file_id=file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        return batch.id

    def _wait_batch(self, batch_id: str, poll_interval: int = 30):
        while True:
            batch = self.client.batches.retrieve(batch_id)
            print("status:", batch.status)
            if batch.status in {"completed", "failed", "expired", "cancelled"}:
                return batch
            time.sleep(poll_interval)

    def _download_file(self, file_id: str, output_path: str):
        content = self.client.files.content(file_id)
        Path(output_path).write_text(content.text, encoding="utf-8")

    def _parse_response(self, text: str) -> dict:
        result = {label: 0 for label in self.config.label_columns}
        for part in text.split(","):
            if ":" not in part:
                continue
            key, value = part.split(":", 1)
            key = key.strip()
            if key in result:
                try:
                    result[key] = int(value.strip())
                except ValueError:
                    result[key] = 0
        return result

    def _read_output(self, output_path: str) -> pd.DataFrame:
        rows = []
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                custom_id = item.get("custom_id")
                error = item.get("error")
                if error:
                    rows.append({
                        "custom_id": custom_id,
                        "raw_response": None,
                        "error": json.dumps(error, ensure_ascii=False),
                        **{label: 0 for label in self.config.label_columns},
                    })
                    continue
                content = item["response"]["body"]["choices"][0]["message"]["content"]
                pred = self._parse_response(content)
                rows.append({"custom_id": custom_id, "raw_response": content, "error": None, **pred})
        return pd.DataFrame(rows)

    def _compute_metrics(self, labels, preds) -> dict:
        metrics = {}
        all_precision, all_recall, all_f1 = [], [], []
        for task_idx, col in enumerate(self.config.label_columns):
            report = classification_report(
                labels[:, task_idx],
                preds[:, task_idx],
                labels=[0, 1],
                output_dict=True,
                zero_division=0,
            )
            macro = report["macro avg"]
            metrics[f"{col}_precision"] = round(macro["precision"], 4)
            metrics[f"{col}_recall"] = round(macro["recall"], 4)
            metrics[f"{col}_f1"] = round(macro["f1-score"], 4)
            metrics[f"{col}_support"] = int(report["1"]["support"])
            all_precision.append(macro["precision"])
            all_recall.append(macro["recall"])
            all_f1.append(macro["f1-score"])
        metrics["macro_avg_precision"] = round(np.mean(all_precision), 4)
        metrics["macro_avg_recall"] = round(np.mean(all_recall), 4)
        metrics["macro_avg_f1"] = round(np.mean(all_f1), 4)
        return metrics

    def predict(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        if self.config.workdir is None:
            raise ValueError("workdir precisa ser definido no config para rodar o batch.")
        
        workdir = Path(self.config.workdir)
        workdir.mkdir(parents=True, exist_ok=True)

        input_path = f"{workdir}/input.jsonl"
        output_path = f"{workdir}/output.jsonl"

        self._build_jsonl(df, text_column, input_path)
        batch_id = self._create_batch(input_path)
        batch = self._wait_batch(batch_id)

        if batch.status != "completed":
            raise RuntimeError(f"Batch terminou com status={batch.status}")

        self._download_file(batch.output_file_id, output_path)
        pred_df = self._read_output(output_path)

        df = df.reset_index(drop=True).copy()
        df["custom_id"] = [f"tweet-{i}" for i in range(len(df))]
        return df.merge(pred_df, on="custom_id", how="left")

    def evaluate(self, df: pd.DataFrame, text_column: str) -> dict:
        result = self.predict(df, text_column)
        labels = df[self.config.label_columns].values
        preds = result[self.config.label_columns].values
        return self._compute_metrics(labels, preds)