import cv2
import numpy as np
from EncoderDecoder import EncoderDecoder

from sklearn.model_selection import train_test_split

def split(features, test_size, labels=None):
    return train_test_split(features, labels, test_size=test_size, random_state=128)

def read_dataset_list(dataset_list_file, delimiter=' '):
    features = []
    labels = []
    with open(dataset_list_file) as f:
        data = f.readlines()
    data = [x.strip() for x in data]
    for example in data:
        example = example.split(delimiter)
        features.append(example[0])
        labels.append(example[-1])
    return features, labels


def read_images(data_dir, image_paths, image_extension='png'):
    images = []
    i = 0
    for image_name in image_paths:
        if i % 100 == 0:
            print('Number of images read: {}/{}'.format(i, len(image_paths)))
        images.append(cv2.imread(data_dir + image_name + '.' + image_extension, 0).astype(np.float32))
        i = i + 1
    print('Done')
    return images

def resize(images, shape):
    resized_images = []
    for image in images:
        resized_images.append(cv2.resize(image, shape))
    return resized_images

def charset():
    return ''.join([line.rstrip('\n') for line in open('chars.txt')])

def encode(labels):
    encoder_decoder = EncoderDecoder()
    encoder_decoder.initialize_encode_and_decode_maps_from(charset())
    encoded_labels = []
    for label in labels:
        encoded_labels.append(encoder_decoder.encode(label))
    return encoded_labels


def convert_to_sparse(labels, dtype=np.int32):
    indices = []
    values = []

    for n, seq in enumerate(labels):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=dtype)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(labels), np.asarray(indices).max(0)[1] + 1], dtype=dtype)

    return indices, values, shape


def get_seq_lens(data):
    return np.asarray([len(s) for s in data], dtype=np.int32)