import tfutils as network_utils
from ModelFn import ModelFn

from tensorflow.contrib.learn import ModeKeys
from tensorflow.python.training.training_util import get_global_step
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib


class GridRNNModelFn(ModelFn):
    def __init__(self, input_shape, num_hidden_units, num_classes, learning_rate, optimizer):
        self.params = {
            "input_shape": input_shape,
            "num_hidden_units": num_hidden_units,
            "num_classes": num_classes,
            "learning_rate": learning_rate,
            "optimizer": optimizer
        }

    @staticmethod
    def model_fn(features, labels, mode, params):
        input_layer = network_utils.reshape(features["x"], params["input_shape"])
        seq_lens = network_utils.reshape(features["seq_lens"], [-1])
        sparse_labels = network_utils.dense_to_sparse(labels, eos_token=80)
        net = network_utils.bidirectional_grid_lstm(inputs=input_layer, num_hidden=params["num_hidden_units"])
        net = network_utils.get_time_major(inputs=net,
                                           num_classes=params["num_classes"],
                                           batch_size=network_utils.get_shape(input_layer)[0],
                                           num_hidden_units=params["num_hidden_units"] * 2)
        net = network_utils.transpose(net, (1, 0, 2))

        loss = None
        train_op = None

        if mode != ModeKeys.INFER:
            loss = network_utils.ctc_loss(inputs=net, labels=sparse_labels, sequence_length=seq_lens)

        if mode == ModeKeys.TRAIN:
            optimizer = network_utils.get_optimizer(learning_rate=params["learning_rate"],
                                                    optimizer_name=params["optimizer"])
            train_op = optimizer.minimize(loss=loss, global_step=get_global_step())

        decoded, log_probabilities = network_utils.ctc_beam_search_decoder(inputs=net, sequence_length=seq_lens)
        dense_decoded = network_utils.sparse_to_dense(decoded, name="output")

        predictions = {
            "decoded": dense_decoded,
            "probabilities": log_probabilities
        }

        eval_metric_ops = {
            "label_error_rate": network_utils.label_error_rate(y_pred=decoded, y_true=sparse_labels)
        }

        return model_fn_lib.ModelFnOps(mode=mode,
                                       predictions=predictions,
                                       loss=loss,
                                       train_op=train_op,
                                       eval_metric_ops=eval_metric_ops)
