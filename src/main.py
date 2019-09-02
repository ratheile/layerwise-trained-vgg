#%%
from networks import AutoencoderNet
from loaders import MnistLoader

loader = MnistLoader(
  '/home/shafall/datasets/mnist',
  download=False
)
net = AutoencoderNet(loader)
net.train()
net.save()