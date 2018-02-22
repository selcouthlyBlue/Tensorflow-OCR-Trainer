from train_ocr import train

def main():
    train(model_config_file='../models/Three_layer_CNN_BiRNN_model.json',
          labels_file='/home/kapitan/Desktop/Jerome/words.txt',
          data_dir='/home/kapitan/Desktop/Jerome/words/',
          desired_image_height=48,
          desired_image_width=256,
          test_fraction=0.3,
          num_epochs=160,
          validation_steps=5,
          batch_size=320,
          max_label_length=32)


if __name__ == '__main__':
    main()
