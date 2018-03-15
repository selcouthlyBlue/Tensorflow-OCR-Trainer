$(document).ready(function(){
    $('select').material_select();
    $(".button-collapse").sideNav();

    var layer_types = {
        "conv2d": "conv2d",
        "max_pool2d": "max_pool2d",
        "l2_normalize": "l2_normalize",
        "batch_norm": "batch_norm",
        "mdrnn": "mdrnn",
        "birnn": "birnn",
        "dropout": "dropout",
        "collapse_to_rnn_dims": "collapse_to_rnn_dims"
    };

    function populate(selector, values) {
        for (var key in values) {
            $(selector).append($('<option>', {value: key}).text(values[key]))
        }
    }

    function create_selector(key, placeholder_text, values) {
        var placeholder = $('<option>', {"value": ""})
            .prop('disabled', true)
            .prop('selected', true).text(placeholder_text);
        var selector = $('<select>', {
            "name": key
        }).prop('required', true)
            .append(placeholder)
            .each(function () {
                populate(this, values);
            });
        return selector
    }

    function create_int_input_field(input_name) {
        var input_id = Math.floor(Math.random() * 2147483647) + "-" + Math.floor(Math.random() * 2147483647);
        return $('<input>').attr({'type': 'number', 'step': "1", 'min': "1", "name": input_name, "id": input_id}).prop('required', true)
    }

    var paddings = {
        "same": "same",
        "valid": "valid"
    };

    var cell_types = {
        "LSTM": "LSTM",
        "GRU": "GRU",
        "GLSTM": "GLSTM"
    };

    var activation_functions = {
        "tanh": "tanh",
        "relu": "relu",
        "relu6": "relu6"
    };

    function create_network_layer_param(key){
        return "network[][" + key + "]"
    }

    var conv2d_params = {
        "num_filters": create_int_input_field(create_network_layer_param("num_filters"))
            .prop('required', true),
        "kernel_size": create_int_input_field(create_network_layer_param("kernel_size"))
            .prop({'required': true, 'multiple': true}),
        "stride": create_int_input_field(create_network_layer_param("stride"))
            .prop({'required': true, 'multiple': true}),
        "padding": create_selector(create_network_layer_param("padding"), "Select padding", paddings)
    };

    var maxpool2d_params = {
        "pool_size": create_int_input_field(create_network_layer_param("pool_size")).prop('required', true),
        "stride": create_int_input_field(create_network_layer_param("stride")).prop('required', true),
        "padding": create_selector(create_network_layer_param("padding"), "Select padding", paddings)
    };

    var mdrnn_params = {
        "num_hidden": create_int_input_field(create_network_layer_param("num_hidden")),
        "kernel_size": create_int_input_field(create_network_layer_param("kernel_size"))
            .prop({'required': true, 'multiple': true}),
        "cell_type": create_selector(create_network_layer_param("cell_type"),
            "Select cell type", cell_types),
        "activation": create_selector(create_network_layer_param("activation"),
            "Select activation function", activation_functions)
    };

    var birnn_params = {
        "num_hidden": create_int_input_field(create_network_layer_param("num_hidden")),
        "cell_type": create_selector(create_network_layer_param("cell_type"),
            "Select cell type", cell_types),
        "activation": create_selector(create_network_layer_param("activation"),
            "Select activation function", activation_functions)
    };

    var l2_normalize_params = {
        "axis": create_int_input_field(create_network_layer_param("axis"))
            .prop({'required': true, 'multiple': true})
    };

    var dropout_params = {
        "keep_prob": $('<input>').attr({'type': 'number',
            'step': "any", "max": "1", "min": "0.000001",  "name": create_network_layer_param("keep_prob")}).prop('required', true)
            .attr('placeholder', 'Keep Prob')
    };

    var layer_params = {
        "conv2d": conv2d_params,
        "max_pool2d": maxpool2d_params,
        "mdrnn": mdrnn_params,
        "birnn": birnn_params,
        "l2_normalize": l2_normalize_params,
        "dropout": dropout_params
    };

    $('#add-layer').click(function (e) {
        e.preventDefault();
        var layer_type_selector = create_selector(create_network_layer_param("layer_type"), "Select layer type", layer_types);
        $('#layers').append(
            $('<li>', {"class": "collection-item row overflowing"}).append(
                $($('<ul>', {"class": "layer"})).append($('<ul>', {"class": "layer_parameters"}))
                    .prepend(
                    $('<li>', {"class": "input-field col s3"}).append(
                        $(layer_type_selector)
                            .change(
                                function () {
                                    var val = $(this).val();
                                    var params_container = $(this).parent().parent().parent().find('.layer_parameters');
                                    params_container.empty();

                                    function append_layer_params(input_fields){
                                        for(var input_field in input_fields){
                                            var element = input_fields[input_field];
                                            var input_field_label = $('<label>', {'for': element.attr('id')});
                                            input_field_label.text(input_field_label.text() + input_field);
                                            params_container.append(
                                                $('<li>', {"class": "input-field col s2"})
                                                    .append(element)
                                                    .append($(input_field_label))
                                            );
                                        }
                                    }
                                    append_layer_params(layer_params[val]);
                                }
                            )
                        )
                    )
                )
        );
        layer_type_selector.material_select();
        $('select').material_select();
    });
});
