from architecture_enum import Architectures
from optimizer_enum import Optimizers
from train_using_tf_estimator import train

def main():
    train(labels_file='../test/dummy_labels_file.txt',
          data_dir='../test/dummy_data/',
          desired_image_size=(1596, 48),
          architecture=Architectures.CNNMDLSTM,
          num_hidden_units=16,
          optimizer=Optimizers.MOMENTUM,
          learning_rate=0.001,
          batch_size=1,
          test_fraction=0.5,
          validation_steps=5)


if __name__ == '__main__':
    main()
