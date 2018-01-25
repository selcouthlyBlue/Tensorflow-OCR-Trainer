**OCR Trainer**

A trainer that uses Tensorflow's Estimator for ease of creating and training models.
So far the trainable architectures implemented here are:

1. GridRNNCTCModel
2. CNNMDLSTMCTCModel

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