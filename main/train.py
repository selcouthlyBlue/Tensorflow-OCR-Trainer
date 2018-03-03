from train_ocr import train

def main():
    train(model_config_file='../models/three_layer_cmdrnn_ctc_model.json',
          labels_file='/home/kapitan/Desktop/Jerome/words.txt',
          data_dir='/home/kapitan/Desktop/Jerome/words/',
          desired_image_height=168,
          desired_image_width=168,
          test_fraction=0.3,
          num_epochs=160,
          save_checkpoint_epochs=1,
          batch_size=240,
          max_label_length=21)


if __name__ == '__main__':
    main()
