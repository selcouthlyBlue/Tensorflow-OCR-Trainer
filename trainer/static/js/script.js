$(document).ready(function(){
    $('select').material_select();
    $(".button-collapse").sideNav();
    var layer_types = {
        "conv2d": "conv2d",
        "maxpool2d": "maxpool2d",
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

    $('#add-layer').click(function (e) {
        e.preventDefault();
        $('#layers').append(
            $('<li>', {"class": "collection-item row"}).append(
                $($('<div>', {"class": "input-field col s3"})).append(
                    $('<select>', {
                        "name": "network[][layer_type]",
                        "class": "layer"
                    })
                    .prop('required', true)
                    .append($('<option>', {"value": ""})
                        .prop('disabled', true)
                        .prop('selected', true)
                        .text("Select Layer Type")
                    ).each(function(){
                        populate(this, layer_types);
                    })
                )
            ).append($('<div>')
                .append($('<a>')).attr({})
            )
        );
        $('select').material_select();
    });
});
