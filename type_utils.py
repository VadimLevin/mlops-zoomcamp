import dataclasses
from dataclasses import dataclass
from typing import Union, Callable, Dict, Protocol

import numpy as np
from scipy.sparse import spmatrix


@dataclass
class Dataset:
    features: Union[np.ndarray, spmatrix]
    values: np.ndarray


@dataclass
class DatasetMetrics:
    dataset_name: str
    predicted_values: np.ndarray
    metrics: Dict[str, float] = dataclasses.field(default_factory=dict)


MetricType = Callable[[np.ndarray, np.ndarray], float]


class ModelType(Protocol):
    def predict(self, features: Union[np.ndarray, spmatrix]) -> np.ndarray:
        ...
