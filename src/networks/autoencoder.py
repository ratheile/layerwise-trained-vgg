from modules import Autoencoder, OriginalAutoencoder, \
  SupervisedAutoencoder, StackableNetwork, NetworkStack

from loaders import semi_supervised_mnist, semi_supervised_cifar10

import torch
from torch.autograd import Variable
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch import nn, Tensor 
from torch import cat as torch_cat
from torch import save as torch_save
from torch import max as torch_max
from torch.optim import Adam

#logging
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

import os
import io

import numpy as np
import matplotlib.pyplot as plt

class AutoencoderNet():

  supervised_loader: DataLoader = None
  unsupvised_loader: DataLoader = None
  test_loader: DataLoader = None
  writer:SummaryWriter = None

  learning_rate = 1e-3
  num_epochs = 100
  device = 'cpu'
  test_losses = []
  test_accs = []

  def __init__(self, mnist_path, ):

    #TODO: automatically fill these variables
    color_channels=3
    img_size=32
    self.supervised_loader, self.unsupvised_loader,\
    self.test_loader = semi_supervised_cifar10(
      mnist_path, supervised_ratio=0.1, batch_size=1000
    )

    assert len(self.supervised_loader) == len(self.unsupvised_loader)

    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    self.model_l1 = SupervisedAutoencoder(color_channels=3).to(self.device)
    self.model_l2 = SupervisedAutoencoder(color_channels=3).to(self.device)
    self.stack = NetworkStack([self.model_l1, self.model_l2]).to(self.device)

    self.writer = SummaryWriter()
    summary(self.model, input_size=(color_channels,img_size,img_size))

    if not os.path.exists('./dc_img'):
        os.mkdir('./dc_img')
    
    self.decoding_criterion = nn.MSELoss()
    self.pred_criterion = nn.CrossEntropyLoss()
    self.optimizer = Adam(
      self.model.parameters(), 
      lr=self.learning_rate, 
      weight_decay=1e-5
    )


  def save(self):
    torch_save(
      self.model.state_dict(),
      './conv_autoencoder.pth'
    )

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

  def train(self, epoch):
    self.model.train()
    for ith_batch in range(len(self.unsupvised_loader)):

      # _s means supervised _us unsupervised
      iter_us = iter(self.unsupvised_loader)
      iter_s = iter(self.supervised_loader)

      img_us, _ = (lambda d: (d[0], d[1]))(next(iter_us))
      img_s, label_s = (lambda d: (d[0], d[1]))(next(iter_s))

      # copy all vars to device
      dev_img_us = img_us.to(self.device)
      dev_img_s = img_s.to(self.device)
      dev_label = label_s.to(self.device)
      
      decoding_s, prediction = self.model(dev_img_s)
      decoding_us, _ = self.model(dev_img_us)

      loss_dc_s = self.decoding_criterion(decoding_s, dev_img_s)
      loss_dc_us = self.decoding_criterion(decoding_us, dev_img_us)

      loss_pred = self.pred_criterion(prediction, dev_label)

      combo_loss = loss_dc_s + loss_dc_us + loss_pred

      # ===================backward====================
      self.optimizer.zero_grad()
      combo_loss.backward()
      self.optimizer.step()

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
    
    self.writer.add_scalar('Train Loss', combo_loss.item(), global_step=epoch)
    self.writer.add_scalar('Train Accuracy', accuracy, global_step=epoch)
    
  def test(self, epoch):
    self.model.eval()
    test_loss = 0
    test_acc = 0

    with torch.no_grad():
      img: Tensor = None
      labels: Tensor = None
      for data in self.test_loader:
        img, label = data[0], data[1]
        dev_img, dev_label = img.to(self.device), label.to(self.device)
        # ===================Forward=====================
        decoding, prediction = self.model(dev_img)
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

    self.writer.add_scalar('Test Loss', test_loss, global_step=epoch)
    self.writer.add_scalar('Test Accuracy', test_acc, global_step=epoch)

    if epoch % 5 == 0:
      self.plot_img(real_imgs=img.numpy(), dc_imgs=decoding.cpu().detach().numpy(), epoch=epoch)

  def train_test(self):
    for epoch in range(self.num_epochs):
      self.train(epoch)
      self.test(epoch)