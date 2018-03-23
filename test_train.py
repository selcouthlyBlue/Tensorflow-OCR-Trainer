from trainer.backend import train_model

def main():
    train_model(network_config_file='architectures/test_model.json',
                dataset_dir='dataset/Test/',
                desired_image_height=38,
                desired_image_width=38,
                num_epochs=1,
                checkpoint_epochs=1,
                batch_size=1,
                max_label_length=37,
                charset_file='charsets/chars.txt',
                learning_rate=0.0001,
                optimizer="momentum",
                metrics=["label_error_rate"],
                loss="ctc")


if __name__ == '__main__':
    main()
