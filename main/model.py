import tfutils as network_utils
import tensorflow as tf

from optimizer_enum import Optimizers

class Model(object):
    def __init__(self):
        self.num_time_steps = 1596
        self.input_dimension = 48
        self.inputs = network_utils.input_data([None, self.num_time_steps, self.input_dimension], name="input")
        self.labels = network_utils.label_data(name="label")
        self.seq_lens = network_utils.input_data([None], name="seq_len", input_type=tf.int32)

    def _inference(self):
        model = network_utils.bidirectional_lstm(self.inputs, 50, return_seq=True)
        model = network_utils.bidirectional_lstm(model, 100, return_seq=True)
        model = network_utils.bidirectional_lstm(model, 200)
        return model

    def loss(self):
        y_predict = self._inference()
        loss = network_utils.ctc_loss(predictions=y_predict, labels=self.labels, sequence_length=self.seq_lens)
        decoded = network_utils.decode(inputs=y_predict, sequence_length=self.seq_lens)
        label_error_rate = network_utils.label_error_rate(y_pred=decoded[0], y_true=self.labels)
        return loss, label_error_rate

    def optimize(self, optimizer_name, learning_rate):
        return network_utils.optimize(loss=self.loss, optimizer=Optimizers.get_optimizer(optimizer_name), learning_rate=learning_rate)
