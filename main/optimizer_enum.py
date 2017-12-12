from enum import Enum

class Optimizers(Enum):
    ADAM = "adam"
    RMSPROP = "rmsprop"
    ADADELTA = "adadelta"

    @staticmethod
    def get_optimizer(optimizer_name):
        return getattr(Optimizers, optimizer_name)
