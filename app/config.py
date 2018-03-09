class BaseConfig(object):
    DEBUG = False
    TESTING = False
    MODELS_DIRECTORY = "models"
    DATASET_DIRECTORY = "dataset"
    CHARSET_DIRECTORY = "charset"


class DevelopmentConfig(BaseConfig):
    DEBUG = True
    TESTING = True


class TestingConfig(BaseConfig):
    DEBUG = False
    TESTING = True
