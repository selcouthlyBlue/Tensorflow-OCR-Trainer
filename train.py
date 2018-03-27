import argparse

from trainer.backend import train_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", help='json file that contains the network architecture')
    parser.add_argument("--dataset_dir", help='folder of the dataset to be used for training')
    parser.add_argument("--desired_image_size", type=int, help='the size to be used by all images')
    parser.add_argument("--num_epochs", type=int, help='number of epochs to run training', default=1)
    parser.add_argument("--checkpoint_epochs", type=int, help='number of epochs before a model is saved', default=1)
    parser.add_argument("--batch_size", type=int, help='number of examples to use per batch of training', default=1)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--optimizer", help='solver type to be used for training')
    parser.add_argument("--loss", help='loss to be used for training')
    parser.add_argument("--metrics", nargs='*', help='metrics to be used for training')
    parser.add_argument("--max_label_length", type=int, default=120,
                        help='the maximum length for each label. Labels will be padded to reach the max length.')
    args = parser.parse_args()
    train_model(architecture_config_file=args.architecture,
                dataset_dir=args.dataset_dir,
                desired_image_size=args.desired_image_size,
                num_epochs=args.num_epochs,
                checkpoint_epochs=args.checkpoint_epochs,
                batch_size=args.batch_size,
                max_label_length=args.max_label_length,
                charset_file='charsets/chars.txt',
                learning_rate=args.learning_rate,
                optimizer=args.optimizer,
                metrics=args.metrics,
                loss=args.loss)


if __name__ == '__main__':
    main()
