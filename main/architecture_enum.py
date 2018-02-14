from enum import Enum

class Architectures(Enum):
    CNNMDLSTM = "cnnmdlstm"

    @staticmethod
    def get_optimizer(optimizer_name):
        return getattr(Architectures, optimizer_name)
