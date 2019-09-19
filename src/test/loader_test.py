import unittest   # The test framework

from loaders import semi_supervised_mnist
from torchvision.datasets import MNIST

class Test_MinstLoader(unittest.TestCase):

  path = '/home/shafall/datasets/mnist'
  mnist_test_len = 10000
  mnist_train_len = 60000

  def test_dataset_selection(self):    
    supervised_loader, unsupvised_loader, test_loader = semi_supervised_mnist(
      self.path, supervised_ratio=0.1, batch_size=1
    )
    assert len(supervised_loader) == 0.1 * self.mnist_train_len 
    assert len(unsupvised_loader) == 0.9 * self.mnist_train_len 
    assert len(test_loader) == self.mnist_test_len

  def test_batch_size(self):
    batch_size = 100
    supervised_loader, unsupvised_loader, test_loader = semi_supervised_mnist(
      self.path, supervised_ratio=0.1, batch_size=batch_size
    )
    assert len(unsupvised_loader) == 0.9 * self.mnist_train_len / batch_size


if __name__ == '__main__':
    unittest.main()