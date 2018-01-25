**OCR Trainer**

A trainer that uses Tensorflow's Estimator for ease of creating and training models.
So far the trainable architectures implemented here are:

1. GridRNNCTCModel
2. CNNMDLSTMCTCModel

To be able to run training, simply `cd main` and run one of the following training scripts:

-`python train.py`

You can also modify the parameters inside `train.py` to perform your experiments
using different hyperparemters to improve performance.