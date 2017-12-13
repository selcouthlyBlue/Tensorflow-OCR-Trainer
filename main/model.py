import tfutils as network_utils

class Model(object):
    def __init__(self):
        self.num_classes = 80
        self.input_dimension = 48
        self.inputs = network_utils.input_data([None, None, self.input_dimension], name="input")
        self.labels = network_utils.sparse_input_data()
        self.seq_lens = network_utils.input_data([None], name="seq_len", input_type=network_utils.get_type('int32'))
        self.learning_rate = 0.01

    def _inference(self):
        model = network_utils.bidirectional_lstm(self.inputs, 25)
        logits = network_utils.get_time_major(model, self.num_classes, network_utils.get_shape(self.inputs)[0], 200)
        return logits

    def loss(self):
        y_predict = self._inference()
        loss = network_utils.ctc_loss(predictions=y_predict, labels=self.labels, sequence_length=self.seq_lens)
        cost = network_utils.cost(loss)
        decoded = network_utils.decode(inputs=y_predict, sequence_length=self.seq_lens)
        label_error_rate = network_utils.label_error_rate(y_pred=decoded[0], y_true=self.labels)
        return loss, label_error_rate, cost

    def optimize(self, optimizer):
        return network_utils.optimize(optimizer=optimizer, learning_rate=self.learning_rate)
