#%%
from networks import AutoencoderNet

net = AutoencoderNet(
  '/home/shafall/datasets/mnist',
)
net.train_test()
net.save()