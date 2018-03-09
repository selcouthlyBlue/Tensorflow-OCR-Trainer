from train_ocr import train

def main():
    train(model_config_file='../models/test_model.json',
          labels_file='dummy_labels_file.txt',
          data_dir='dummy_data/',
          desired_image_height=360,
          desired_image_width=360,
          test_fraction=0.3,
          num_epochs=1,
          save_checkpoint_epochs=1,
          batch_size=1,
          max_label_length=37)


if __name__ == '__main__':
    main()
