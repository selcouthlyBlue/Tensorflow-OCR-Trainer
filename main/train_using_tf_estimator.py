import dataset_utils
import tensorflow as tf
import numpy as np

from tensorflow.contrib import grid_rnn, learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

tf.logging.set_verbosity(tf.logging.INFO)

def grid_rnn_fn(features, labels, mode):
    input_layer = tf.reshape(features["x"], [-1, 1596, 48])
    seq_lens = tf.reshape(features["seq_lens"], [-1])
    indices = tf.where(tf.not_equal(labels, tf.constant(0, dtype=tf.int32)))
    values = tf.gather_nd(labels, indices)
    sparse_labels = tf.SparseTensor(indices, values, dense_shape=tf.shape(labels, out_type=tf.int64))
    cell_fw = grid_rnn.Grid2LSTMCell(num_units=128)
    cell_bw = grid_rnn.Grid2LSTMCell(num_units=128)
    bidirectional_grid_rnn = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, input_layer, dtype=tf.float32)
    outputs = tf.reshape(bidirectional_grid_rnn[0], [-1, 256])

    W = tf.Variable(tf.truncated_normal([256,
                                         80],
                                        stddev=0.1, dtype=tf.float32), name='W')
    b = tf.Variable(tf.constant(0., dtype=tf.float32, shape=[80], name='b'))

    logits = tf.matmul(outputs, W) + b
    logits = tf.reshape(logits, [tf.shape(input_layer)[0], -1, 80])
    logits = tf.transpose(logits, (1, 0, 2))

    loss = None
    train_op = None

    if mode != learn.ModeKeys.INFER:
        loss = tf.nn.ctc_loss(inputs=logits, labels=sparse_labels, sequence_length=seq_lens)

    if mode == learn.ModeKeys.TRAIN:
        train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss=loss,
                                                                        global_step=tf.train.get_global_step())

    decoded, log_probabilities = tf.nn.ctc_beam_search_decoder(inputs=logits, sequence_length=seq_lens)
    dense_decoded = tf.sparse_to_dense(tf.to_int32(decoded[0].indices),
                                       tf.to_int32(decoded[0].values),
                                       tf.to_int32(decoded[0].dense_shape),
                                       name="output")

    predictions = {
        "decoded": dense_decoded,
        "probabilities": log_probabilities
    }

    return model_fn_lib.ModelFnOps(mode=mode,
                                   predictions=predictions,
                                   loss=loss,
                                   train_op=train_op)

def main(_):
    image_paths, labels = dataset_utils.read_dataset_list('../test/dummy_labels_file.txt')
    data_dir = "../test/dummy_data/"
    images = dataset_utils.read_images(data_dir=data_dir, image_paths=image_paths, image_extension='png')
    print('Done reading images')
    images = dataset_utils.resize(images, (1596, 48))
    images = dataset_utils.transpose(images)
    labels = dataset_utils.encode(labels)
    x_train, x_test, y_train, y_test = dataset_utils.split(features=images, test_size=0.5, labels=labels)
    x_train_seq_lens = dataset_utils.get_seq_lens(x_train)
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(x_train),
           "seq_lens": np.array(x_train_seq_lens)},
        y=np.array(y_train),
        num_epochs=1,
        shuffle=True,
        batch_size=1
    )

    classifier = learn.Estimator(model_fn=grid_rnn_fn, model_dir="/tmp/grid_rnn_ocr_model")
    classifier.fit(input_fn=train_input_fn)


if __name__ == '__main__':
    tf.app.run(main=main)
