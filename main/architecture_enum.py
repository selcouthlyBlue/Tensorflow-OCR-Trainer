from enum import Enum

class Architectures(Enum):
    CNNMDLSTM = "cnnmdlstm"
    GRIDLSTM = "gridlstm"

    @staticmethod
    def get_optimizer(optimizer_name):
        return getattr(Architectures, optimizer_name)
