import numpy as np
import tensorflow as tf
import logging

from tensorflow.python.estimator.export.export import ServingInputReceiver
from tensorflow.python.tools import freeze_graph

from trainer.backend.GraphKeys import Optimizers
from trainer.backend.GraphKeys import OutputLayers
from trainer.backend.GraphKeys import Metrics
from trainer.backend.GraphKeys import Losses

from trainer.backend.tf import ctc_ops, losses, metric_functions
from trainer.backend.tf.replicate_model_fn import TowerOptimizer
from trainer.backend.tf.util_ops import feed, dense_to_sparse, get_sequence_lengths
from trainer.backend.tf.ValidationHook import ValidationHook

from tensorflow.contrib import slim
from tensorflow.contrib.opt import NadamOptimizer

tf.logging.set_verbosity(tf.logging.INFO)


def _get_loss(loss, labels, inputs, num_classes):
    if loss == Losses.CTC.value:
        ctc_inputs = ctc_ops.convert_to_ctc_dims(inputs,
                                                 num_classes=num_classes,
                                                 num_steps=inputs.shape[1],
                                                 num_outputs=inputs.shape[-1])
        labels = dense_to_sparse(labels, token_to_ignore=-1)
        return losses.ctc_loss(labels=labels,
                               inputs=ctc_inputs,
                               sequence_length=get_sequence_lengths(inputs))
    raise NotImplementedError(loss + " loss not implemented")


def _sparse_to_dense(sparse_tensor, name="sparse_to_dense"):
    return tf.sparse_to_dense(tf.to_int32(sparse_tensor.indices),
                              tf.to_int32(sparse_tensor.dense_shape),
                              tf.to_int32(sparse_tensor.values),
                              name=name)


def _get_optimizer(learning_rate, optimizer_name):
    if optimizer_name == Optimizers.MOMENTUM.value:
        return tf.train.MomentumOptimizer(learning_rate,
                                          momentum=0.9,
                                          use_nesterov=True)
    if optimizer_name == Optimizers.ADAM.value:
        return tf.train.AdamOptimizer(learning_rate)
    if optimizer_name == Optimizers.ADADELTA.value:
        return tf.train.AdadeltaOptimizer(learning_rate)
    if optimizer_name == Optimizers.RMSPROP.value:
        return tf.train.RMSPropOptimizer(learning_rate)
    if optimizer_name == Optimizers.NADAM.value:
        return NadamOptimizer(learning_rate)
    raise NotImplementedError(optimizer_name + " optimizer not supported")


def train(params, features, labels, num_classes, checkpoint_dir,
          batch_size=1, num_epochs=1,
          save_checkpoint_every_n_epochs=1):
    _set_logger_to_file(checkpoint_dir, 'train')
    num_steps_per_epoch = len(features['train']) // batch_size
    save_checkpoint_steps = save_checkpoint_every_n_epochs * num_steps_per_epoch
    params['num_classes'] = num_classes
    params['log_step_count_steps'] = num_steps_per_epoch
    training_hooks = []
    estimator = tf.estimator.Estimator(model_fn=_train_model_fn,
                                       params=params,
                                       model_dir=checkpoint_dir,
                                       config=tf.estimator.RunConfig(
                                           save_checkpoints_steps=save_checkpoint_steps,
                                           log_step_count_steps=num_steps_per_epoch,
                                           save_summary_steps=num_steps_per_epoch
                                       ))
    if features.get('validation'):
        training_hooks.append(ValidationHook(
            model_fn=_eval_model_fn,
            params=params,
            input_fn=_input_fn(features['validation'],
                               labels['validation'],
                               batch_size,
                               num_epochs=1,
                               shuffle=False),
            checkpoint_dir=checkpoint_dir,
            every_n_steps=save_checkpoint_steps
        ))
    estimator.train(input_fn=_input_fn(features['train'], labels['train'], batch_size),
                    steps=num_epochs * num_steps_per_epoch,
                    hooks=training_hooks)


def test(params, features, labels, checkpoint_dir):
    _set_logger_to_file(checkpoint_dir, 'test')
    params['summary_dir'] = checkpoint_dir + '/test'
    estimator = tf.estimator.Estimator(model_fn=_test_model_fn,
                                       params=params,
                                       model_dir=checkpoint_dir)
    estimator.evaluate(input_fn=_input_fn(features,
                                          labels,
                                          batch_size=params['batch_size'],
                                          num_epochs=1,
                                          shuffle=False))


# see https://stackoverflow.com/a/44296581
def _set_logger_to_file(checkpoint_dir, task):
    log = logging.getLogger('tensorflow')
    log.setLevel(logging.DEBUG)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # create file handler which logs even debug messages
    fh = logging.FileHandler(checkpoint_dir + '/' + task + '.log')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    log.addHandler(fh)


def predict(params, features, checkpoint_dir):
    estimator = tf.estimator.Estimator(model_fn=_predict_model_fn,
                                       params=params,
                                       model_dir=checkpoint_dir)
    predictions = estimator.predict(input_fn=_input_fn(features, num_epochs=1))
    for i, p in enumerate(predictions):
        print(i, p)


def _input_fn(features, labels=None, batch_size=1, num_epochs=None, shuffle=True):
    if labels:
        labels = np.array(labels, dtype=np.int32)
    return tf.estimator.inputs.numpy_input_fn(
        x={'features': np.array(features)},
        y=labels,
        batch_size=batch_size,
        num_epochs=num_epochs,
        shuffle=shuffle
    )


def _add_to_summary(name, value):
    tf.summary.scalar(name, value)


def _create_train_op(loss, learning_rate, optimizer):
    optimizer = _get_optimizer(learning_rate, optimizer)
    optimizer = TowerOptimizer(optimizer)
    return slim.learning.create_train_op(loss, optimizer, global_step=tf.train.get_or_create_global_step())


def _create_model_fn(mode, predictions, loss=None, train_op=None,
                     eval_metric_ops=None, training_hooks=None,
                     evaluation_hooks=None, export_outputs=None):
    return tf.estimator.EstimatorSpec(mode=mode,
                                      predictions=predictions,
                                      loss=loss,
                                      train_op=train_op,
                                      eval_metric_ops=eval_metric_ops,
                                      training_hooks=training_hooks,
                                      evaluation_hooks=evaluation_hooks,
                                      export_outputs=export_outputs)


def _get_output(rnn_outputs, output_layer, num_classes):
    if output_layer == OutputLayers.CTC_DECODER.value:
        ctc_inputs = ctc_ops.convert_to_ctc_dims(rnn_outputs,
                                                 num_classes=num_classes,
                                                 num_steps=rnn_outputs.shape[1],
                                                 num_outputs=rnn_outputs.shape[-1])
        decoded, _ = ctc_ops.ctc_beam_search_decoder(ctc_inputs, get_sequence_lengths(rnn_outputs))
        return _sparse_to_dense(decoded, name="output")
    raise NotImplementedError(output_layer + " not implemented")


def _get_metrics(metrics, y_pred, y_true, num_classes):
    metrics_dict = {}
    for metric in metrics:
        if metric == Metrics.LABEL_ERROR_RATE.value:
            ctc_inputs = ctc_ops.convert_to_ctc_dims(y_pred,
                                                     num_classes=num_classes,
                                                     num_steps=y_pred.shape[1],
                                                     num_outputs=y_pred.shape[-1])
            y_pred, _ = ctc_ops.ctc_beam_search_decoder(ctc_inputs, get_sequence_lengths(y_pred))
            y_true = dense_to_sparse(y_true, token_to_ignore=-1)
            value = metric_functions.label_error_rate(y_pred,
                                                      y_true,
                                                      metric)
        else:
            raise NotImplementedError(metric + " metric not implemented")
        metrics_dict[metric] = value
    return metrics_dict


def create_serving_model(checkpoint_dir, run_params, input_name="features"):
    serving_model_path, input_shape = _export_serving_model(checkpoint_dir, run_params, input_name)
    return serving_model_path, input_shape


def _export_serving_model(checkpoint_dir, model_params, input_name="features"):
    model_params["input_name"] = input_name
    checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
    input_checkpoint = checkpoint.model_checkpoint_path
    with tf.Session(graph=tf.Graph()) as sess:
        saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
        saver.restore(sess, input_checkpoint)
        input_layer = tf.get_default_graph().get_operation_by_name('input_layer').outputs[0]
        input_shape = input_layer.get_shape().as_list()
        input_shape[0] = 1
        estimator = tf.estimator.Estimator(model_fn=_serving_model_fn,
                                           params=model_params,
                                           model_dir=checkpoint_dir)

    def _serving_input_receiver_fn():
        serialized_tf_example = tf.placeholder(dtype=input_layer.dtype,
                                               shape=input_shape,
                                               name=input_name)
        receiver_tensors = {input_name: serialized_tf_example}
        return ServingInputReceiver(receiver_tensors, receiver_tensors)

    serving_model_path = estimator.export_savedmodel(checkpoint_dir, _serving_input_receiver_fn,
                                                     as_text=True)
    return serving_model_path, input_shape


def create_optimized_graph(model_filename,
                           output_nodes="output",
                           output_graph_filename="optimized_graph.pb"):
    freeze_graph.freeze_graph(
        input_graph=None,
        input_saver=None,
        input_binary=False,
        input_checkpoint=None,
        output_node_names=output_nodes,
        restore_op_name=None,
        filename_tensor_name=None,
        output_graph=output_graph_filename,
        clear_devices=True,
        initializer_nodes=None,
        input_saved_model_dir=model_filename
    )


def _write_graph(output_graph_def, output_graph_filename):
    with tf.gfile.GFile(output_graph_filename, "wb") as f:
        f.write(output_graph_def.SerializeToString())


def _serving_model_fn(features, mode, params):
    features = features[params["input_name"]]
    return _predict_model_fn(features, mode, params)


def _predict_model_fn(features, mode, params):
    features = _network_fn(features, mode, params)
    outputs = _get_output(features, params["output_layer"], params["num_classes"])
    predictions = {
        "outputs": outputs
    }

    return _create_model_fn(mode, predictions=predictions,
                            export_outputs={
                                "outputs": tf.estimator.export.PredictOutput(predictions)
                            })


def _test_model_fn(features, labels, mode, params):
    loss, metrics, predictions = _get_evaluation_parameters(features, labels, mode, params)
    _add_to_summary("test_loss", loss)
    for metric_key in metrics:
        _add_to_summary("test_" + metric_key, metrics[metric_key])
    evaluation_hooks = [tf.train.LoggingTensorHook(predictions, every_n_iter=1),
                        tf.train.SummarySaverHook(save_steps=1,
                                                  output_dir=params['summary_dir'],
                                                  summary_op=tf.summary.merge_all())
                        ]
    return _create_model_fn(mode, predictions=predictions, loss=loss,
                            evaluation_hooks=evaluation_hooks)


def _eval_model_fn(features, labels, mode, params):
    loss, metrics, predictions = _get_evaluation_parameters(features, labels, mode, params)

    evaluation_hooks = [tf.train.LoggingTensorHook(predictions, every_n_iter=1)]
    for metric_key in metrics:
        metrics[metric_key] = metric_functions.create_eval_metric(metrics[metric_key])
    return _create_model_fn(mode, predictions=predictions, loss=loss,
                            eval_metric_ops=metrics, evaluation_hooks=evaluation_hooks)


def _train_model_fn(features, labels, mode, params):
    loss, metrics, predictions = _get_evaluation_parameters(features, labels, mode, params)

    train_op = _create_train_op(loss,
                                learning_rate=params["learning_rate"],
                                optimizer=params["optimizer"])

    training_hooks = []
    for metric_key in metrics:
        _add_to_summary(metric_key, metrics[metric_key])
        training_hooks.append(tf.train.LoggingTensorHook(
            {metric_key: metric_key},
            every_n_iter=params["log_step_count_steps"])
        )
    return _create_model_fn(mode,
                            predictions=predictions,
                            loss=loss,
                            train_op=train_op,
                            training_hooks=training_hooks)


def _get_evaluation_parameters(features, labels, mode, params):
    features, predictions = _get_fed_features_and_resulting_predictions(features, mode, params)
    loss = _get_loss(params["loss"], labels=labels,
                     inputs=features, num_classes=params["num_classes"])
    metrics = _get_metrics(params["metrics"],
                           y_pred=features,
                           y_true=labels,
                           num_classes=params["num_classes"])
    return loss, metrics, predictions


def _get_fed_features_and_resulting_predictions(features, mode, params):
    features = features['features']
    features = _network_fn(features, mode, params)
    outputs = _get_output(features, params["output_layer"], params["num_classes"])
    predictions = {
        "outputs": outputs
    }
    return features, predictions


def _network_fn(features, mode, params):
    features = _set_dynamic_batch_size(features)
    for layer in params["network"]:
        features = feed(features, layer, is_training=mode == tf.estimator.ModeKeys.TRAIN)
    return features


def _set_dynamic_batch_size(inputs):
    new_shape = inputs.get_shape().as_list()
    new_shape[0] = -1
    inputs = tf.reshape(inputs, new_shape, name="input_layer")
    return inputs
