import tfutils as network_utils

def model_fn(features, labels, mode, params):
    network = params["network"]
    target_type = params["target_type"]
    metrics = params["metrics"]
    output_layer = params["output_layer"]
    loss = params["loss"]
    learning_rate = params["learning_rate"]
    optimizer = params["optimizer"]

    labels = network_utils.format_labels(labels, target_type)

    for layer in network:
        features = network_utils.feed(features, layer, mode)

    outputs = network_utils.get_output(features, output_layer)
    predictions = {
        "outputs": outputs
    }
    if network_utils.is_predict(mode):
        return network_utils.create_model_fn(mode, predictions=predictions)

    loss = network_utils.get_loss(loss, labels=labels, inputs=features)
    metrics = network_utils.get_metric(metrics, y_pred=features, y_true=labels)

    if network_utils.is_evaluation(mode):
        return network_utils.create_model_fn(mode,
                                             predictions=predictions,
                                             loss=loss,
                                             eval_metric_ops=metrics)

    assert network_utils.is_training(mode)

    train_op = network_utils.create_train_op(loss,
                                             learning_rate=learning_rate,
                                             optimizer=optimizer)
    return network_utils.create_model_fn(mode,
                                         predictions=predictions,
                                         loss=loss,
                                         train_op=train_op)
