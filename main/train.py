from train_ocr import train

def main():
    train(model_config_file='../models/two_layer_cnn_birnn_ctc_model.json',
          labels_file='/home/kapitan/Desktop/Jerome/words.txt',
          data_dir='/home/kapitan/Desktop/Jerome/words/',
          desired_image_height=128,
          desired_image_width=128,
          test_fraction=0.3,
          num_epochs=160,
          validation_steps=5,
          batch_size=360,
          max_label_length=32)


if __name__ == '__main__':
    main()
