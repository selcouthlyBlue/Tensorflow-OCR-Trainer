from trainer.backend import start_training

def main():
    start_training(model_config_file='architectures/test_model.json',
                   labels_file='dataset/Test/labels.txt',
                   data_dir='dataset/Test/',
                   desired_image_height=38,
                   desired_image_width=38,
                   num_epochs=1,
                   save_checkpoint_epochs=1,
                   batch_size=1,
                   max_label_length=37,
                   charset_file='charsets/chars.txt')


if __name__ == '__main__':
    main()
