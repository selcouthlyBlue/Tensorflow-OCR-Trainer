import cv2
import numpy as np
from EncoderDecoder import EncoderDecoder


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


def invert(images):
    print('Inverting color of images...')
    inverted_images = []
    for image in images:
        inverted_image = cv2.bitwise_not(image)
        inverted_images.append(inverted_image)
    print('Done inverting color of images.')
    return inverted_images


def images_as_float32(images):
    print('Converting pixel values to float...')
    float32_images = []
    for image in images:
        float32_images.append(image.astype(np.float32))
    print('Done converting pixel values to float...')
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
    if dim is (None, None):
        return image
    raw_height, raw_width, num_channels = image.shape
    ratio = float(desired_width)/max(raw_height, raw_width)

    scaled_width = int(raw_width * ratio)
    scaled_height = int(raw_height * ratio)
    resized_image = cv2.resize(image, (scaled_width, scaled_height))

    square_image = _add_padding(desired_height, desired_width, resized_image, scaled_height, scaled_width)
    return np.array(square_image).astype(np.uint8)


def _add_padding(desired_height, desired_width, image, scaled_height, scaled_width):
    delta_width = desired_width - scaled_width
    delta_height = desired_height - scaled_height
    top, bottom = delta_height // 2, delta_height - (delta_height // 2)
    left, right = delta_width // 2, delta_width - (delta_width // 2)
    color = [255, 255, 255]
    square_image = cv2.copyMakeBorder(image,
                                      top,
                                      bottom,
                                      left,
                                      right,
                                      cv2.BORDER_CONSTANT,
                                      value=color)
    return square_image


def get_characters_from(charset_file):
    return ''.join([line.rstrip('\n') for line in open(charset_file)])


def encode(labels, classes):
    encoder_decoder = EncoderDecoder()
    encoder_decoder.initialize_encode_and_decode_maps_from(
        classes
    )
    encoded_labels = []
    for label in labels:
        encoded_labels.append(encoder_decoder.encode(label))
    return encoded_labels


def pad(labels, padding_index, max_label_length=120):
    padded_labels = []
    for label in labels:
        while len(label) < max_label_length:
            label = np.append(label, [padding_index])
        padded_labels.append(label)
    return padded_labels
