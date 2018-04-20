class BaseConfig(object):
    DEBUG = False
    TESTING = False
    ARCHITECTURES_DIRECTORY = "architectures"
    DATASET_DIRECTORY = "dataset"
    CHARSET_DIRECTORY = "charset"
    ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    ALLOWED_LABELS_FILE_EXTENSIONS = {'txt', 'csv'}
    ALLOWED_ZIP_EXTENSIONS = {'gz', 'rar', 'zip'}
    MODELS_DIRECTORY = "checkpoint"
    VISUALIZATION_HOST = "localhost"
    OUTPUT_GRAPH_FILENAME = "graph.pb"
    MODEL_ZIP_FILENAME = "model_files.gz"
    SERVING_MODEL_CONFIG_FILENAME = "serving_model_config.json"


class DevelopmentConfig(BaseConfig):
    DEBUG = True
    TESTING = True


class TestingConfig(BaseConfig):
    DEBUG = False
    TESTING = True
