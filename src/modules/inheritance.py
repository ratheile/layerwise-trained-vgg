r"""
Experimental class not used in the final product
"""
from torch import nn

class StackableNetwork(object):

  def __init__(self):
    print("Hello from stackable network")
    pass

  def abstract_method(self):
    raise "not implemented"


class Autoencoder(object):

  # define class attributes for the sake of documentation
  decoder: nn.Sequential = None

  def __init__(self, color_channels=3):
    super().__init__()
    print("Hello from autoencoder")
    print("decoder", self.decoder)
    self.decoder = nn.Sequential()


class SupervisedAutoencoder(Autoencoder, StackableNetwork):

  supervision: nn.Sequential = None
  test = 1

  def __init__(self, color_channels):
    # super().__init__(color_channels)
    Autoencoder.__init__(self, color_channels=color_channels)
    StackableNetwork.__init__(self)
    print("Hello from supervised autoencoder")
    print("decoder", self.decoder)
    print("supervision", self.supervision)
    self.supervision = nn.Sequential()
    print("supervision", self.supervision)

  def abstract_method(self):
    print("implementation")


if __name__ == "__main__":
  sae = SupervisedAutoencoder(1)
  sae.test += 1
  sae2 = SupervisedAutoencoder(1)
  print(sae2.test, sae.test)
  sae.abstract_method()