#%%
from networks import AutoencoderNet

net = AutoencoderNet(
  '/home/shafall/datasets/cifar10',
)
net.train_test()
net.save()