from enum import Enum

class LayerTypes(Enum):
    CONV2D = "conv2d"
    MAX_POOL2D = "max_pool2d"
    COLLAPSE_TO_RNN_DIMS = "collapse_to_rnn_dims"
    MDRNN = "mdrnn"
    BIRNN = "birnn"
    L2_NORMALIZE = "l2_normalize"
    BATCH_NORM = "batch_norm"
    DROPOUT = "dropout"

class PaddingTypes(Enum):
    SAME = "same"
    VALID = "valid"

class CellTypes(Enum):
    LSTM = "LSTM"
    GRU = "GRU"
    GLSTM = "GLSTM"

class ActivationFunctions(Enum):
    TANH = "tanh"
    RELU = "relu"
    RELU6 = "relu6"

class Optimizers(Enum):
    ADAM = "adam"
    MOMENTUM = "momentum"
    ADADELTA = "adadelta"
    RMSPROP = "rmsprop"

class Metrics(Enum):
    LABEL_ERROR_RATE = "label_error_rate"

class Losses(Enum):
    CTC = "ctc"

class OutputLayers(Enum):
    CTC_DECODER = "ctc_decoder"
