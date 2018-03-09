from train_ocr import train

def main():
    train(model_config_file='../architectures/test_model.json',
          labels_file='labels.txt',
          data_dir='Test/',
          desired_image_height=360,
          desired_image_width=360,
          test_fraction=0.3,
          num_epochs=1,
          save_checkpoint_epochs=1,
          batch_size=1,
          max_label_length=37)


if __name__ == '__main__':
    main()
