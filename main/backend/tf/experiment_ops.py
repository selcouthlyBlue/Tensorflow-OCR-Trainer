import tensorflow as tf
import numpy as np

from sklearn.model_selection import train_test_split
from tensorflow.contrib import learn
from tensorflow.contrib import slim
from tensorflow.contrib.learn import ModeKeys
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

from backend.tf import ctc_ops, losses, metric_functions
from backend.tf.util_ops import feed, dense_to_sparse, get_sequence_lengths

tf.logging.set_verbosity(tf.logging.INFO)

def _get_loss(loss, labels, inputs, num_classes):
    if loss == "ctc":
        inputs = ctc_ops.convert_to_ctc_dims(inputs,
                                             num_classes=num_classes,
                                             num_steps=inputs.shape[1],
                                             num_outputs=inputs.shape[-1])
        labels = dense_to_sparse(labels, token_to_ignore=-1)
        return losses.ctc_loss(labels=labels,
                               inputs=inputs,
                               sequence_length=get_sequence_lengths(inputs))
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


def run_experiment(params, features, labels, checkpoint_dir,
                   num_classes, batch_size=1, num_epochs=1,
                   save_checkpoint_every_n_epochs=1, test_fraction=None):
    x_train = features
    y_train = labels
    num_steps_per_epoch = len(x_train) // batch_size
    monitors = []
    if test_fraction:
        x_train, x_validation, y_train, y_validation = train_test_split(
            x_train,
            y_train,
            test_size=test_fraction
        )
        num_steps_per_epoch = len(x_train) // batch_size
        validation_monitor = learn.monitors.ValidationMonitor(
            input_fn=_input_fn(x_validation,
                               y_validation,
                               batch_size,
                               num_epochs,
                               shuffle=False),
            every_n_steps=save_checkpoint_every_n_epochs * num_steps_per_epoch)
        monitors.append(validation_monitor)
        print('Number of training samples:', len(x_train))
        print('Number of validation samples', len(x_validation))
    params['num_classes'] = num_classes
    params['log_step_count_steps'] = num_steps_per_epoch
    estimator = learn.Estimator(model_fn=_model_fn,
                                params=params,
                                model_dir=checkpoint_dir,
                                config=learn.RunConfig(
                                    save_checkpoints_steps=num_steps_per_epoch,
                                    log_step_count_steps=num_steps_per_epoch,
                                    save_summary_steps=num_steps_per_epoch)
                                )
    estimator.fit(input_fn=_input_fn(x_train,
                                     y_train,
                                     batch_size),
                  monitors=monitors,
                  steps=num_epochs * num_steps_per_epoch)


def _input_fn(features, labels, batch_size=1, num_epochs=None, shuffle=True):
    return tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(features)},
        y=np.array(labels, dtype=np.int32),
        batch_size=batch_size,
        num_epochs=num_epochs,
        shuffle=shuffle
    )


def _add_to_summary(name, value):
    tf.summary.scalar(name, value)


def _create_train_op(loss, learning_rate, optimizer):
    optimizer = _get_optimizer(learning_rate, optimizer)
    return slim.learning.create_train_op(loss, optimizer, global_step=tf.train.get_or_create_global_step())


def _create_model_fn(mode, predictions, loss=None, train_op=None, eval_metric_ops=None, training_hooks=None):
    return model_fn_lib.ModelFnOps(mode=mode,
                                   predictions=predictions,
                                   loss=loss,
                                   train_op=train_op,
                                   eval_metric_ops=eval_metric_ops,
                                   training_hooks=training_hooks)


def _get_output(inputs, output_layer, num_classes):
    if output_layer == "ctc_decoder":
        inputs = ctc_ops.convert_to_ctc_dims(inputs,
                                             num_classes=num_classes,
                                             num_steps=inputs.shape[1],
                                             num_outputs=inputs.shape[-1])
        decoded, _ = ctc_ops.ctc_beam_search_decoder(inputs)
        return _sparse_to_dense(decoded, name="output")
    raise NotImplementedError(output_layer + " not implemented")


def _get_metrics(metrics, y_pred, y_true, num_classes, log_step_count_steps=100):
    metrics_dict = {}
    training_hooks = []
    for metric in metrics:
        if metric == "label_error_rate":
            y_pred = ctc_ops.convert_to_ctc_dims(y_pred,
                                                 num_classes=num_classes,
                                                 num_steps=y_pred.shape[1],
                                                 num_outputs=y_pred.shape[-1])
            y_pred, _ = ctc_ops.ctc_beam_search_decoder(y_pred)
            y_true = dense_to_sparse(y_true, token_to_ignore=-1)
            value = metric_functions.label_error_rate(y_pred,
                                                      y_true,
                                                      metric)
        else:
            raise NotImplementedError(metric + " metric not implemented")
        _add_to_summary(metric, value)
        training_hooks.append(tf.train.LoggingTensorHook({metric: metric},
                                                         every_n_iter=log_step_count_steps))
        metrics_dict[metric] = metric_functions.create_metric(value)
    return metrics_dict, training_hooks


def _model_fn(features, labels, mode, params):
    features = features["x"]

    network = params["network"]
    metrics = params["metrics"]
    output_layer = params["output_layer"]
    loss = params["loss"]
    learning_rate = params["learning_rate"]
    optimizer = params["optimizer"]
    num_classes = params["num_classes"]
    log_step_count_steps = params['log_step_count_steps']

    for layer in network:
        features = feed(features, layer, is_training=mode==ModeKeys.TRAIN)

    outputs = _get_output(features, output_layer, num_classes)
    predictions = {
        "outputs": outputs
    }
    if mode==ModeKeys.INFER:
        return _create_model_fn(mode, predictions=predictions)

    loss = _get_loss(loss, labels=labels, inputs=features, num_classes=num_classes)
    _add_to_summary("loss", loss)
    metrics, training_hooks = _get_metrics(metrics,
                                           y_pred=features,
                                           y_true=labels,
                                           num_classes=num_classes,
                                           log_step_count_steps=log_step_count_steps)

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
                            train_op=train_op,
                            training_hooks=training_hooks)
