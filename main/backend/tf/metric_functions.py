import tensorflow as tf

from backend.tf.util_ops import dense_to_sparse


def create_metric(values):
    return tf.metrics.mean(values)


def label_error_rate(y_pred, y_true, num_classes):
    y_true = dense_to_sparse(y_true, num_classes)
    return tf.reduce_mean(tf.edit_distance(tf.cast(y_pred, tf.int32), y_true), name="label_error_rate")
