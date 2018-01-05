To be able to run training, simply cd into main and run one of the following training scripts:

-python train.py
-python train_using_tflearn_trainer.py
-python train_using_tflearn_dnn.py

The other two scripts for training that use tflearn are not yet working.

Running train_using_tflearn_trainer.py throws a RecursionError.
Running train_using_tflearn_dnn.py throws a TypeError requiring a SparseTensor for ctc_loss.