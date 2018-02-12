import tfutils as network_utils

from Model import Model


class GridRNNCTCModel(Model):
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
        net = network_utils.get_logits(inputs=net,
                                       num_classes=params["num_classes"],
                                       num_steps=net.shape[1],
                                       num_hidden_units=params["num_hidden_units"] * 2,
                                       mode=mode,
                                       use_batch_norm=True)

        loss = None
        train_op = None

        if not network_utils.is_inference(mode):
            loss = network_utils.ctc_loss(labels=sparse_labels, inputs=net, sequence_length=seq_lens)
            network_utils.add_to_summary("loss", loss)

        if network_utils.is_training(mode):
            train_op = network_utils.create_train_op(loss, params["learning_rate"], params["optimizer"])

        decoded, log_probabilities = network_utils.ctc_beam_search_decoder(inputs=net, sequence_length=seq_lens)
        dense_decoded = network_utils.sparse_to_dense(decoded, name="output")
        label_error_rate = network_utils.label_error_rate(y_pred=decoded, y_true=sparse_labels)
        network_utils.add_to_summary("label_error_rate", label_error_rate)

        predictions = {
            "decoded": dense_decoded,
            "probabilities": log_probabilities,
            "label_error_rate": label_error_rate
        }

        return network_utils.create_model_fn(mode=mode,
                                             predictions=predictions,
                                             loss=loss,
                                             train_op=train_op)
