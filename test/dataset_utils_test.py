import unittest
import os
import dataset_utils


class DatasetUtilsTest(unittest.TestCase):
    def setUp(self):
        self.data = {'numbers': [1, 2], 'labels':['a', 'c']}

    def test_read_labels_file(self):
        features, labels = dataset_utils.read_dataset_list('dummy_labels_file.txt')
        self.assertTrue(len(features) == len(labels) == 2)

    def test_split_dataset(self):
        test_size = 0.5
        x_train, y_train, x_test, y_test = dataset_utils.split(self.data['numbers'], test_size, labels=self.data['labels'])
        self.assertTrue(len(x_train) == len(y_train) == 1 and len(x_test) == len(y_test) == 1)
        

if __name__ == '__main__':
    unittest.main()
