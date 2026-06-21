import re
import pandas as pd

def _clean(text: str) -> str | None:
    text = re.sub(r"\brt\b|\bretweet\b", '', text, flags=re.IGNORECASE)
    text = re.sub(r"@[A-Za-z0-9_]+", '@user', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r" +", ' ', text).strip()
    return text if text != "" else None

def _normalize(text: str, abbrev_dict: dict) -> str | None:
    if not isinstance(text, str): 
            return None
        
    for abbrev, expr in abbrev_dict.items():
        text = re.sub(r'\b' + re.escape(abbrev) + r'\b', expr, text, flags=re.IGNORECASE)
    return text

def preprocess_text(df: pd.DataFrame, text_col: str, abbrev_dict: dict = None) -> pd.DataFrame:
    df = df.copy()
    df[text_col] = df[text_col].apply(_clean)
    if abbrev_dict:
        df[text_col] = df[text_col].apply(lambda x: _normalize(x, abbrev_dict))
    
    return df.dropna(subset=[text_col]).reset_index(drop=True)