import dataset_utils
import tensorflow as tf
import numpy as np

from tensorflow.contrib import grid_rnn, learn, layers, framework

def grid_rnn_fn(features, labels, mode):
    input_layer = tf.reshape(features["x"], [-1, 48, 1596])
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
        loss = tf.nn.ctc_loss(inputs=logits, labels=labels, sequence_length=320)

    if mode == learn.ModeKeys.TRAIN:
        train_op = layers.optimize_loss(loss=loss, global_step=framework.get_global_step(),
                                        learning_rate=0.001,
                                        optimizer="Adam")

    decoded, log_probabilities = tf.nn.ctc_beam_search_decoder(inputs=logits, sequence_length=320)

    predictions = {
        "decoded": decoded,
        "probabilities": log_probabilities
    }

    return learn.estimators.ModelFnOps(mode=mode,
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
    sparse_y_train = dataset_utils.convert_to_sparse(y_train)
    sparse_y_train = tf.SparseTensor(indices=sparse_y_train[0],
                                     values=sparse_y_train[1],
                                     dense_shape=sparse_y_train[2])

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(x_train)},
        y=sparse_y_train,
        num_epochs=1,
        shuffle=True,
        batch_size=1
    )

    classifier = learn.Estimator(model_fn=grid_rnn_fn, model_dir="/tmp/grid_rnn_ocr_model")
    classifier.fit(input_fn=train_input_fn)


if __name__ == '__main__':
    tf.app.run(main=main)
