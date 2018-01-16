from abc import ABC, abstractmethod

class ModelFn(ABC):
    @staticmethod
    @abstractmethod
    def model_fn(features, labels, mode, params):
        pass
