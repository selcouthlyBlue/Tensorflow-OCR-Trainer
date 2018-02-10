import cv2
import numpy as np
from EncoderDecoder import EncoderDecoder

from sklearn.model_selection import train_test_split

def split(features, test_size, labels=None):
    return train_test_split(features, labels, test_size=test_size)


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
    print('Reading images...')
    images = []
    for image_name in image_paths:
        images.append(cv2.imread(data_dir + image_name + '.' + image_extension))
    print('Done reading images. Number of images read:', len(image_paths))
    return images


def binarize(images):
    print('Binarizing images...')
    binarized_images = []
    for image in images:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binarized_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        binarized_images.append(binarized_image)
    print('Done binarizing images.')
    return binarized_images


def images_as_float32(images):
    float32_images = []
    for image in images:
        float32_images.append(image.astype(np.float32))
    return float32_images


def resize(images, desired_height=None, desired_width=None):
    print("Resizing images...")
    resized_images = []
    for image in images:
        resized_image = _resize(image, desired_height, desired_width)
        resized_images.append(resized_image)
    print("Done resizing images.")
    return resized_images


def _resize(image, desired_height=None, desired_width=None):
    dim = (desired_width, desired_height)
    if (desired_width is None and desired_height is None) or dim is (None, None):
        return image
    if desired_width is None:
        dim = (desired_height, image.shape[1])
    elif desired_height is None:
        dim = (image.shape[0], desired_width)
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)


def charset():
    return ''.join([line.rstrip('\n') for line in open('chars.txt')])


def encode(labels):
    encoder_decoder = EncoderDecoder()
    encoder_decoder.initialize_encode_and_decode_maps_from(charset())
    encoded_labels = []
    for label in labels:
        encoded_labels.append(encoder_decoder.encode(label))
    return encoded_labels


def set_seq_lens(length, number_of_samples):
    return np.full(number_of_samples, length)


def transpose(images):
    transposed_images = []
    for image in images:
        transposed_images.append(image.swapaxes(0,1))
    return transposed_images


def pad(labels, blank_token_index, max_label_length=120):
    padded_labels = []
    for label in labels:
        label = np.append(label, [blank_token_index])
        while len(label) < max_label_length:
            label = np.append(label, [blank_token_index])
        padded_labels.append(label)
    return padded_labels
