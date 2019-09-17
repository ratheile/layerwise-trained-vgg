import unittest   # The test framework

from loaders import MnistLoader, SetType

class Test_MinstLoader(unittest.TestCase):

  path = '/home/shafall/datasets/mnist'

  def test_dataset_selection(self):
    loader = MnistLoader(self.path, type = SetType.TEST)
    assert len(loader) == 79
    loader = MnistLoader(self.path, type = SetType.TRAIN)
    assert len(loader) == 469


if __name__ == '__main__':
    unittest.main()