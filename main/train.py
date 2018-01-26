from architecture_enum import Architectures
from optimizer_enum import Optimizers
from train_using_tf_estimator import train

def main():
    train(labels_file='/home/kapitan/Desktop/Jerome/lines.txt',
          data_dir='/home/kapitan/Desktop/Jerome/IAM_lines/',
          labels_delimiter=' ',
          desired_image_size=(1024, 96),
          architecture=Architectures.CNNMDLSTM,
          num_hidden_units=16,
          optimizer=Optimizers.MOMENTUM,
          learning_rate=0.001,
          test_fraction=0.3,
          num_epochs=160,
          validation_steps=10,
          batch_size=32)


if __name__ == '__main__':
    main()
