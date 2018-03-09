class BaseConfig(object):
    DEBUG = False
    TESTING = False
    ARCHITECTURES_DIRECTORY = "architectures"
    DATASET_DIRECTORY = "dataset"
    CHARSET_DIRECTORY = "charset"
    ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    ALLOWED_LABELS_FILE_EXTENSIONS = {'txt', 'csv'}


class DevelopmentConfig(BaseConfig):
    DEBUG = True
    TESTING = True


class TestingConfig(BaseConfig):
    DEBUG = False
    TESTING = True
