from enum import Enum

class Optimizers(Enum):
    ADAM = "adam"
    RMSPROP = "rmsprop"
    ADADELTA = "adadelta"
    MOMENTUM = "momentum"

    @staticmethod
    def get_optimizer(optimizer_name):
        return getattr(Optimizers, optimizer_name)
