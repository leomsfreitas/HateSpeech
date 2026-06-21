from pathlib import Path

DATA_PATH = Path("./data")
GO_EMOTIONS_PATH = Path("./models/BERTimbau_large_GoEmotions_portuguese")

_directories = [
    DATA_PATH,
    GO_EMOTIONS_PATH
]

for directory in _directories:
    if not directory.exists():
        raise RuntimeError(f"Diretório obrigatório não encontrado: {directory.resolve()}")