# src/models/base.py

from abc import ABC, abstractmethod
import numpy as np


class BaseModel(ABC):
    """
    Base interface for all models
    """

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    def save(self, path: str):
        """
        Optional override if model needs custom save
        """
        pass

    def load(self, path: str):
        """
        Optional override if model needs custom load
        """
        pass