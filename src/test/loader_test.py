import unittest   # The test framework

from loaders import semi_supervised_mnist
from loaders import semi_supervised_cifar10

from torchvision.datasets import MNIST

class Test_Cifar10Loader(unittest.TestCase):

  path = '/home/shafall/datasets/cifar10'
  cifar_test_len = 10000
  cifar_train_len = 50000

  def test_download(self):
    semi_supervised_cifar10(self.path, download=True)


  def test_dataset_selection(self):
    batch_size = 100
    supervised_loader, unsupvised_loader, test_loader = semi_supervised_cifar10(
          self.path, supervised_ratio=0.1, batch_size=batch_size
        )
    self.assertEqual(float(len(supervised_loader)), self.cifar_train_len / batch_size)
    self.assertEqual(float(len(unsupvised_loader)), self.cifar_train_len / batch_size)
    self.assertEqual(float(len(test_loader)), self.cifar_test_len / batch_size)


class Test_MinstLoader(unittest.TestCase):

  path = '/home/shafall/datasets/mnist'
  mnist_test_len = 10000
  mnist_train_len = 60000

  def test_dataset_selection(self):    
    batch_size = 100
    supervised_loader, unsupvised_loader, test_loader = semi_supervised_mnist(
      self.path, supervised_ratio=0.1, batch_size=batch_size
    )
    self.assertEqual(float(len(supervised_loader)), self.mnist_train_len / batch_size)
    self.assertEqual(float(len(unsupvised_loader)), self.mnist_train_len / batch_size)
    self.assertEqual(float(len(test_loader)), self.mnist_test_len / batch_size)


if __name__ == '__main__':
    unittest.main()