from modules import Autoencoder, \
  SupervisedAutoencoder, StackableNetwork, NetworkStack, RandomMap

from loaders import semi_supervised_mnist, semi_supervised_cifar10

import torch
from torch.autograd import Variable
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch import nn, Tensor 
from torch import cat as torch_cat
from torch import save as torch_save
from torch import max as torch_max
from torch.optim import Adam, Optimizer

#logging
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

import os
import io
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

from typing import List

@dataclass
class LayerTrainingDefinition:
  num_epochs: int = 100
  model: NetworkStack = None
  optimizer: Optimizer = None


def default_network_factory(
  device: str,
  learning_rate:float,
  layers:int=2,
  weight_decay:float=1e-5
) -> List[LayerTrainingDefinition]:

  model_l1 = SupervisedAutoencoder(color_channels=3).to(device)
  model_l2 = SupervisedAutoencoder(color_channels=3).to(device)

  # TODO: Generate layers programmatically and make it depend on layers parameter
  model_t1 = NetworkStack([
    (model_l1, None),
    ]).to(device)

  model_t2 = NetworkStack([
    (model_l1, RandomMap(in_shape=(24,16,16), out_shape=(3,32,32)).to(device)),
    (model_l2, None),
    ]).to(device)

  optimizer_t1 = Adam(
    model_t1.parameters(), 
    lr=learning_rate, 
    weight_decay=weight_decay
  )

  optimizer_t2 = Adam(
    model_t2.parameters(), 
    lr=learning_rate, 
    weight_decay=weight_decay
  )
  layer_configs = [
    LayerTrainingDefinition(num_epochs=100, model=model_t1, optimizer=optimizer_t1),
    LayerTrainingDefinition(num_epochs=100, model=model_t2, optimizer=optimizer_t2)
  ]

  return layer_configs

class AutoencoderNet():

  supervised_loader: DataLoader = None
  unsupvised_loader: DataLoader = None
  test_loader: DataLoader = None
  writer:SummaryWriter = None

  layer_configs: List[LayerTrainingDefinition] = []

  learning_rate = 1e-3
  num_epochs = 100
  device = 'cpu'
  test_losses = []
  test_accs = []

  def __init__(self, data_path, ):
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #TODO: automatically fill these variables
    color_channels=3
    img_size=32
    self.supervised_loader, self.unsupvised_loader,\
    self.test_loader = semi_supervised_cifar10(
      data_path, supervised_ratio=0.1, batch_size=1000
    )

    assert len(self.supervised_loader) == len(self.unsupvised_loader)
    self.layer_configs = default_network_factory(self.device, self.learning_rate)

    self.writer = SummaryWriter()
    """
    TODO: Implement in factory method to show sizes when network is built
    summary(self.model, input_size=(color_channels,img_size,img_size))
    """

    if not os.path.exists('./dc_img'):
        os.mkdir('./dc_img')
    
    self.decoding_criterion = nn.MSELoss()
    self.pred_criterion = nn.CrossEntropyLoss()


  # TODO: implement this for stacking networks
  # def save(self, model):
  #   torch_save(
  #     self.model.state_dict(),
  #     './conv_autoencoder.pth'
  #   )

  def to_img(self, x):
    x = 0.5 * (x + 1)
    x = np.clip(x, 0, 1)
    return x

  def plot_img(self, real_imgs, dc_imgs, epoch):
    real_imgs = real_imgs[:10,:,:,:]
    dc_imgs = dc_imgs[:10,:,:,:]
    fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(25,4))

    for imgs, row in zip([real_imgs, dc_imgs], axes):
      for img, ax in zip(imgs, row):
        if img.shape[0] == 1:
          image = np.squeeze(self.to_img(img))
        elif img.shape[0] == 3:
          image = self.to_img(img).swapaxes(0,2).swapaxes(0,1)
        else:
          raise "Image dimensions do not match (1/3)"
    
        ax.imshow(image, cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)


    self.writer.add_figure('DecodedImgs', fig, global_step=epoch)

  def train(self, epoch: int, global_epoch:int,  config: LayerTrainingDefinition):
    #TODO: check if still necessary self.model.train()
    for ith_batch in range(len(self.unsupvised_loader)):

      # _s means supervised _us unsupervised
      iter_us = iter(self.unsupvised_loader)
      iter_s = iter(self.supervised_loader)

      img_us, _ = (lambda d: (d[0], d[1]))(next(iter_us))
      img_s, label_s = (lambda d: (d[0], d[1]))(next(iter_s))

      # copy all vars to device and calculate the topmost stack representation
      # TODO: avoid calculating this representation twice (here and in forward())
      dev_img_us = config.model.upwards(img_us.to(self.device))
      dev_img_s = config.model.upwards(img_s.to(self.device))
      dev_label = label_s.to(self.device)
      
      decoding_s, prediction = config.model(dev_img_s)
      decoding_us, _ = config.model(dev_img_us)

      loss_dc_s = self.decoding_criterion(decoding_s, dev_img_s)
      loss_dc_us = self.decoding_criterion(decoding_us, dev_img_us)

      loss_pred = self.pred_criterion(prediction, dev_label)


      combo_loss = loss_dc_s + loss_dc_us + loss_pred

      # ===================backward====================
      config.optimizer.zero_grad()
      combo_loss.backward()
      config.optimizer.step()

    # ===================log========================
    # Calculate Accuracy
    _, predicted = torch_max(prediction.data, 1)
    accuracy = (predicted.cpu().numpy() == label_s.numpy()).sum() / len(label_s)

    print('Epoch [{}/{}]\nTrain Loss: {:.4f}      Train Acc: {:.4f}'
      .format(
          epoch+1,
          self.num_epochs, 
          combo_loss.item(),
          accuracy
        )
      )
    
    self.writer.add_scalar('Train Loss', combo_loss.item(), global_step=global_epoch)
    self.writer.add_scalar('Train Accuracy', accuracy, global_step=global_epoch)
    
  def test(self, epoch: int, global_epoch:int, config: LayerTrainingDefinition):
    # TODO: figure out if necessaryself self.model.eval()
    test_loss = 0
    test_acc = 0

    with torch.no_grad():
      img: Tensor = None
      labels: Tensor = None
      for data in self.test_loader:
        img, label = data[0], data[1]

        # copy all vars to device and calculate the topmost stack representation
        # TODO: avoid calculating this representation twice (here and in forward())
        dev_img = config.model.upwards(img.to(self.device))
        dev_label = label.to(self.device)

        # ===================Forward=====================
        decoding, prediction = config.model(dev_img)
        loss_dc = self.decoding_criterion(decoding, dev_img)
        loss_pred = self.pred_criterion(prediction, dev_label)
        combo_loss = loss_dc + 0.5 * loss_pred
        # Calculate Test Loss
        test_loss += combo_loss.item() / (len(self.test_loader))
        # Calculate Accuracy
        _, predicted = torch_max(prediction.data, 1)
        test_acc += (predicted.cpu().numpy() == label.numpy()).sum() / (len(self.test_loader) * len(label))

    self.test_losses.append(test_loss)
    self.test_accs.append(test_acc)

    print('Test Loss:  {:.4f}      Test Acc:  {:.4f}\n'
      .format(
        test_loss,
        test_acc
      )
    )

    self.writer.add_scalar('Test Loss', test_loss, global_step=global_epoch)
    self.writer.add_scalar('Test Accuracy', test_acc, global_step=global_epoch)

    if epoch % 5 == 0:
      self.plot_img(real_imgs=dev_img.cpu().numpy(),
                    dc_imgs=decoding.cpu().detach().numpy(),
                    epoch=global_epoch)

  def train_test(self):
    total_epochs = 0
    for config in self.layer_configs:
      for epoch in range(config.num_epochs):
        self.train(epoch, config=config, global_epoch=total_epochs)
        self.test(epoch, config=config, global_epoch=total_epochs)
        total_epochs += 1
    