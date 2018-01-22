from abc import ABC, abstractmethod

class Model(ABC):
    @staticmethod
    @abstractmethod
    def model_fn(features, labels, mode, params):
        pass
