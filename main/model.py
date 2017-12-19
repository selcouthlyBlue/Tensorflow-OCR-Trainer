import tfutils as network_utils

class Model(object):
    def __init__(self):
        self.num_classes = 80
        self.input_dimension = 48
        self.batch_size = 1
        self.inputs = network_utils.input_data([None, None, self.input_dimension], name="input")
        self.labels = network_utils.sparse_input_data()
        self.seq_lens = network_utils.input_data([None], name="seq_len", input_type=network_utils.get_type('int32'))
        self.learning_rate = 0.001

    def inference(self):
        model = network_utils.bidirectional_lstm(self.inputs, [25, 50, 100])
        logits = network_utils.get_time_major(model, self.num_classes, network_utils.get_shape(self.inputs)[0], 200)
        return logits

    def loss(self, logits):
        loss = network_utils.ctc_loss(predictions=logits, labels=self.labels, sequence_length=self.seq_lens)
        return loss

    def cost(self, logits):
        loss = network_utils.ctc_loss(predictions=logits, labels=self.labels, sequence_length=self.seq_lens)
        return network_utils.cost(loss)

    def training(self, loss, optimizer_name):
        train_op = network_utils.optimize(loss=loss, optimizer_name=optimizer_name, learning_rate=self.learning_rate)
        return train_op

    def label_error_rate(self, logits):
        decoded = network_utils.decode(logits, self.seq_lens)
        return network_utils.label_error_rate(decoded, self.labels)
