import tensorflow as tf


def create_eval_metric(values):
    return tf.metrics.mean(values)


def label_error_rate(y_pred, y_true, name="label_error_rate"):
    return tf.reduce_mean(tf.edit_distance(tf.cast(y_pred, tf.int32), y_true), name=name)
