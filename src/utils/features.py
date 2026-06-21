import pandas as pd
from typing import List

def load_features(base_path: str, feature_paths: List[str]) -> pd.DataFrame:
    df = pd.read_csv(base_path)
    for path in feature_paths:
        df = df.merge(pd.read_csv(path), left_index=True, right_index=True)
    return df