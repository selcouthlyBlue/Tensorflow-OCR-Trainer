var current_layer_index = -1;

$(document).ready(function () {
    $('select').material_select();
});

function create_dynamic_layer_builder(layer_types, padding_types, cell_types, activation_functions){
    function populate(selector, values) {
        $.each(values, function(index, value){
            $(selector).append($('<option>', {"value": value}).text(value))
        });
    }

    function create_selector(key, placeholder_text, values) {
        var placeholder = $('<option>', {"value": ""})
            .prop('disabled', true)
            .prop('selected', true).text(placeholder_text);
        var selector = $('<select>', {"name": key})
            .prop('required', true)
            .append(placeholder);
        populate(selector, values);
        return selector
    }

    function create_int_input_field(input_name) {
        return $('<input>').attr({'type': 'number', 'step': "1", 'min': "1", "name": input_name, "id": input_name}).prop('required', true)
    }


    function create_network_layer_param(key, layer_index){
        return "network[" + layer_index + "][" + key + "]"
    }

    var conv2d_params = function(layer_index) {
        return {"Num Filters": create_int_input_field(create_network_layer_param("num_filters", layer_index)),
                "Kernel Size": create_int_input_field(create_network_layer_param("kernel_size", layer_index)),
                "Stride": create_int_input_field(create_network_layer_param("stride", layer_index)),
                "Padding": create_selector(create_network_layer_param("padding", layer_index), "Select padding", padding_types)}
    };

    var maxpool2d_params = function(layer_index) {
        return {"Pool size": create_int_input_field(create_network_layer_param("pool_size", layer_index)),
                "Stride": create_int_input_field(create_network_layer_param("stride", layer_index)),
                "Padding": create_selector(create_network_layer_param("padding", layer_index), "Select padding", padding_types)}
    };

    var mdrnn_params = function(layer_index) {
        return {"Num Hidden": create_int_input_field(create_network_layer_param("num_hidden", layer_index)),
                "Kernel Size": create_int_input_field(create_network_layer_param("kernel_size", layer_index)).prop("required", false),
                "Cell Type": create_selector(create_network_layer_param("cell_type", layer_index), "Select cell type", cell_types),
                "Activation": create_selector(create_network_layer_param("activation", layer_index), "Select activation", activation_functions)}
    };

    var birnn_params = function(layer_index) {
        return {"Num Hidden": create_int_input_field(create_network_layer_param("num_hidden", layer_index)),
                "Cell Type": create_selector(create_network_layer_param("cell_type", layer_index), "Select cell type", cell_types),
                "Activation": create_selector(create_network_layer_param("activation", layer_index), "Select activation", activation_functions)}
    };

    var dropout_params = function(layer_index) {
        return {"Keep Prob": $('<input>').attr({'type': 'number',
            'step': "any", "max": "1", "min": "0.000001",  "name": create_network_layer_param("keep_prob", layer_index)}).prop('required', true)
            .attr('placeholder', 'Keep Prob')}
    };

    var layer_params = {
        "conv2d": conv2d_params,
        "max_pool2d": maxpool2d_params,
        "mdrnn": mdrnn_params,
        "birnn": birnn_params,
        "l2_normalize": function(layer_index){},
        "dropout": dropout_params,
        "collapse_to_rnn_dims": function(layer_index){},
        "batch_norm": function(layer_index){}
    };

    $('#add-layer').click(function () {
        current_layer_index++;

        var layer_type_selector_behavior = function () {
            var layer_type = $(this).val();
            var layer_index = $(this).attr('name').match(/(\d+)/g);
            var params_container = $(this).parent().parent().parent().children('.layer_parameters');
            params_container.empty();

            function append_layer_params(input_fields){
                $.each(input_fields, function(key, input_field){
                    var input_field_label = $('<label>', {'for': input_field.attr('id')});
                    input_field_label.text(input_field_label.text() + key);
                    params_container.append(
                        $('<li>', {"class": "input-field col s2"})
                            .append(input_field)
                            .append($(input_field_label))
                    );
                    if (input_field.is('select')){
                        input_field.material_select();
                    }
                });
            }

            append_layer_params(layer_params[layer_type](layer_index));
        };

        var layer_type_selector = create_selector(
            create_network_layer_param("layer_type", current_layer_index),
            "Select layer type",
            layer_types).change(layer_type_selector_behavior);

        var remove_button = $(
            $('<a>', {'class': 'waves-effect waves-light btn red circle'})
                .append($('<i>', {'class': 'material-icons'}).text("remove"))
        ).click(function () {
            current_layer_index--;
            $(this).parent().parent().parent().remove();
        });

        $('#layers').append(
            $('<li>', {"class": "layer collection-item row overflowing"}).append(
                $($('<ul>'))
                    .append($('<ul>', {"class": "layer_parameters"}))
                    .prepend($('<li>', {"class": "input-field col s3"}).append(layer_type_selector))
                    .append($('<li>', {"class": "col s1 right"}).append(remove_button))
                )
        );
        layer_type_selector.material_select();
    });
}

$(document).ready(function () {
   $('#my_form').submit(function(){
       if (current_layer_index <= -1){
           alert("Please add some layers.");
           return false;
       }
   })
});