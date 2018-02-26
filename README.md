**OCR Trainer**

A trainer that uses Tensorflow's Estimator for ease of training models.

The architectures are stored in json files like so:

```json
{
  "network":[
    {"layer_type": "input_layer", "name": "inputs", "shape": [-1, 64, 3200, 1]},
    {"layer_type": "conv2d", "num_filters": 16, "kernel_size": [3, 3]},
    {"layer_type": "max_pool2d", "pool_size": [2, 2]},
    {"layer_type": "mdrnn", "num_hidden": 32, "cell_type": "GLSTM"},
    {"layer_type": "conv2d", "num_filters": 48, "kernel_size": [3, 3]},
    {"layer_type": "max_pool2d", "pool_size": [2, 2]},
    {"layer_type": "dropout", "keep_prob": 0.25},
    {"layer_type": "mdrnn", "num_hidden": 64, "cell_type": "GLSTM"},
    {"layer_type": "dropout", "keep_prob": 0.25},
    {"layer_type": "conv2d", "num_filters": 80, "kernel_size": [3, 3]},
    {"layer_type": "max_pool2d", "pool_size": [2, 2]},
    {"layer_type": "dropout", "keep_prob": 0.25},
    {"layer_type": "mdrnn", "num_hidden": 96, "cell_type": "GLSTM"},
    {"layer_type": "dropout", "keep_prob": 0.25},
    {"layer_type": "conv2d", "num_filters": 112, "kernel_size": [3, 3]},
    {"layer_type": "max_pool2d", "pool_size": [2, 2]},
    {"layer_type": "dropout", "keep_prob": 0.25},
    {"layer_type": "mdrnn", "num_hidden": 128, "cell_type": "GLSTM"},
    {"layer_type": "dropout", "keep_prob": 0.25},
    {"layer_type": "conv2d", "num_filters": 134, "kernel_size": [3, 3]},
    {"layer_type": "max_pool2d", "pool_size": [2, 2]},
    {"layer_type": "dropout", "keep_prob": 0.25},
    {"layer_type": "mdrnn", "num_hidden": 160, "cell_type": "GLSTM"},
    {"layer_type": "dropout", "keep_prob": 0.25},
    {"layer_type": "collapse_to_rnn_dims"},
    {"layer_type": "convert_to_ctc_dims", "num_classes": 80}
  ],
  "output_layer": "ctc_decoder",
  "loss": "ctc",
  "metrics": ["label_error_rate"],
  "num_classes": 80,
  "learning_rate": 0.001,
  "optimizer": "momentum"
}
```

To be able to run training, simply `cd main` and run one of the following training scripts:

-`python train.py`

You can also modify the parameters inside `train.py` to perform your experiments
using different hyperparemters to improve performance.

The `data_dir` should contain the images to be used for training and testing.
The `labels_file` should be of this format:

```
    name_of_the_image_without_extension,(some other things),label 
```

And you should set `labels_delimter` to `,` in this case. By default, it is a space.