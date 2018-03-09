import unittest

import dataset_utils


class DatasetUtilsTest(unittest.TestCase):
    def setUp(self):
        self.data = {'numbers': [1, 2], 'labels':['a', 'c']}

    def test_read_labels_file(self):
        features, labels = dataset_utils.read_dataset_list('labels.txt')
        self.assertTrue(len(features) == len(labels) == 2)
        

if __name__ == '__main__':
    unittest.main()
