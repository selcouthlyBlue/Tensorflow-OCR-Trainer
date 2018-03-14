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
    $('#add-layer').click(function (e) {
        e.preventDefault();
        $('#layers').append(
            $('<li>').append(
                $('<div>').append(
                    $('<select>').attr(
                        {
                            "name": "network[][layer_type]",
                            "class": "layer"
                        }
                    )
                    .prop('required', true)
                    .append($('<option>').attr({"value": ""})
                        .prop('disabled', true)
                        .prop('selected', true)
                        .text("Select Layer Type")
                    )
                ).attr({"class": "input-field col s3"})
            ).append($('<div>')
                .append($('<a>')).attr({})
            )
                .attr({
                "class": "collection-item row"
            })
        );
        $('select').material_select();
        $.each(layer_types, function (key, value) {

            }
        );
    });
});
