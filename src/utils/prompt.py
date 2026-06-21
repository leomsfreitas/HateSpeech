from pathlib import Path

load_prompt = lambda path: Path(path).read_text(encoding="utf-8")