import tensorflow as tf
import json
import losses
import ctc_ops
import metric_functions

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
from tensorflow.contrib import slim
from tensorflow.contrib.learn import ModeKeys
from util_ops import feed

tf.logging.set_verbosity(tf.logging.INFO)

def _get_loss(loss, labels, inputs, num_classes):
    if loss == "ctc":
        return losses.ctc_loss(labels=labels,
                               inputs=inputs,
                               eos_token=num_classes)
    raise NotImplementedError(loss + " loss not implemented")


def _sparse_to_dense(sparse_tensor, name="sparse_to_dense"):
    return tf.sparse_to_dense(tf.to_int32(sparse_tensor.indices),
                              tf.to_int32(sparse_tensor.values),
                              tf.to_int32(sparse_tensor.dense_shape),
                              name=name)


def _get_optimizer(learning_rate, optimizer_name):
    if optimizer_name == "momentum":
        return tf.train.MomentumOptimizer(learning_rate,
                                          momentum=0.9,
                                          use_nesterov=True)
    elif optimizer_name == "adam":
        return tf.train.AdamOptimizer(learning_rate)
    elif optimizer_name == "adadelta":
        return tf.train.AdadeltaOptimizer(learning_rate)
    elif optimizer_name == "rmsprop":
        return tf.train.RMSPropOptimizer(learning_rate)
    raise NotImplementedError(optimizer_name + " optimizer not supported")


def run_experiment(model_config_file, train_input_fn, checkpoint_dir, validation_input_fn=None,
                   validation_steps=100):
    validation_monitor = learn.monitors.ValidationMonitor(input_fn=validation_input_fn,
                                                          every_n_steps=validation_steps)
    estimator = learn.Estimator(model_fn=_model_fn,
                                params=json.load(open(model_config_file, 'r')),
                                model_dir=checkpoint_dir,
                                config=learn.RunConfig(save_checkpoints_steps=validation_steps))
    estimator.fit(input_fn=train_input_fn, monitors=[validation_monitor])


def input_fn(x_feed_dict, y, num_epochs=1, shuffle=True, batch_size=1):
    return tf.estimator.inputs.numpy_input_fn(x=x_feed_dict,
                                              y=y,
                                              shuffle=shuffle,
                                              num_epochs=num_epochs,
                                              batch_size=batch_size)


def _add_to_summary(name, value):
    tf.summary.scalar(name, value)


def _create_train_op(loss, learning_rate, optimizer):
    optimizer = _get_optimizer(learning_rate, optimizer)
    return slim.learning.create_train_op(loss, optimizer, global_step=tf.train.get_or_create_global_step())


def _create_model_fn(mode, predictions, loss=None, train_op=None, eval_metric_ops=None):
    return model_fn_lib.ModelFnOps(mode=mode,
                                   predictions=predictions,
                                   loss=loss,
                                   train_op=train_op,
                                   eval_metric_ops=eval_metric_ops)


def _get_output(inputs, output_layer):
    if output_layer == "ctc_decoder":
        decoded, _ = ctc_ops.ctc_beam_search_decoder(inputs)
        return _sparse_to_dense(decoded, name="output")
    raise NotImplementedError(output_layer + " not implemented")


def _get_metric(metrics, y_pred, y_true, num_classes):
    metrics_dict = {}
    for metric in metrics:
        if metric == "label_error_rate":
            y_pred, _ = ctc_ops.ctc_beam_search_decoder(y_pred)
            ler = metric_functions.label_error_rate(y_pred, y_true, num_classes)
            _add_to_summary(metric, ler)
            metrics_dict[metric] = metric_functions.create_metric(ler)
        else:
            raise NotImplementedError(metric + " metric not implemented")
    return metrics_dict


def _model_fn(features, labels, mode, params):
    features = features["x"]

    network = params["network"]
    metrics = params["metrics"]
    output_layer = params["output_layer"]
    loss = params["loss"]
    learning_rate = params["learning_rate"]
    optimizer = params["optimizer"]
    num_classes = params["num_classes"]

    for layer in network:
        features = feed(features, layer, is_training=mode==ModeKeys.TRAIN)

    outputs = _get_output(features, output_layer)
    predictions = {
        "outputs": outputs
    }
    if mode==ModeKeys.INFER:
        return _create_model_fn(mode, predictions=predictions)

    loss = _get_loss(loss, labels=labels, inputs=features, num_classes=num_classes)
    metrics = _get_metric(metrics, y_pred=features, y_true=labels, num_classes=num_classes)

    if mode==ModeKeys.EVAL:
        return _create_model_fn(mode, predictions=predictions, loss=loss,
                                eval_metric_ops=metrics)

    assert mode==ModeKeys.TRAIN

    train_op = _create_train_op(loss,
                                learning_rate=learning_rate,
                                optimizer=optimizer)
    return _create_model_fn(mode,
                            predictions=predictions,
                            loss=loss,
                            train_op=train_op)
