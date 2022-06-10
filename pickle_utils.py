import pickle
import os
from pathlib import Path
from typing import Any, Union


def load_pickle(filepath: Union[os.PathLike, Path]) -> Any:
    with open(filepath, "rb") as fh:
        return pickle.load(fh)

def save_pickle(obj: Any, filepath: Union[os.PathLike, Path]):
    filepath = Path(filepath)
    if not filepath.parent.is_dir():
        filepath.parent.mkdir(parents=True)
    with open(filepath, "wb") as fh:
        pickle.dump(obj, fh)
