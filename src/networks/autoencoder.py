from modules import Autoencoder, OriginalAutoencoder, SupervisedAutoencoder
from loaders import semi_supervised_mnist

import torch
from torch.autograd import Variable
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch import nn, Tensor 
from torch import save as torch_save
from torch import max as torch_max
from torch.optim import Adam

from torchsummary import summary

import os

import numpy as np
import matplotlib.pyplot as plt

class AutoencoderNet():

  supervised_loader: DataLoader = None
  unsupvised_loader: DataLoader = None
  test_loader: DataLoader = None

  learning_rate = 1e-3
  num_epochs = 100
  device = 'cpu'
  test_losses = []
  test_accs = []

  def __init__(self, mnist_path):
    self.supervised_loader, self.unsupvised_loader,\
    self.test_loader = semi_supervised_mnist(
      mnist_path, supervised_ratio=0.1, batch_size=1000
    )

    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.model = SupervisedAutoencoder().to(self.device)

    summary(self.model, input_size=(1,28,28))

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
        image = np.squeeze(self.to_img(img))
        ax.imshow(image, cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.savefig("DecodedImgs" + str(epoch))

  def train(self, epoch):
    self.model.train()
    for data in self.supervised_loader:
      img, label = data[0], data[1]
      dev_img, dev_label = img.to(self.device), label.to(self.device)
      # ===================forward=====================
      decoding, prediction = self.model(dev_img)
      loss_dc = self.decoding_criterion(decoding, dev_img)
      loss_pred = self.pred_criterion(prediction, dev_label)
      combo_loss = loss_dc + loss_pred
      # ===================backward====================
      self.optimizer.zero_grad()
      combo_loss.backward()
      self.optimizer.step()
    # ===================log========================
    # Calculate Accuracy
    _, predicted = torch_max(prediction.data, 1)
    accuracy = (predicted.cpu().numpy() == label.numpy()).sum() / len(label)

    print('Epoch [{}/{}]\nTrain Loss: {:.4f}      Train Acc: {:.4f}'
      .format(
          epoch+1,
          self.num_epochs, 
          combo_loss.item(),
          accuracy
        )
      )
    
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
        combo_loss = loss_dc + 0.5*loss_pred
        # Calculate Test Loss
        test_loss += combo_loss.item()
        # Calculate Accuracy
        _, predicted = torch_max(prediction.data, 1)
        test_acc += (predicted.cpu().numpy() == label.numpy()).sum() / len(label)

    test_loss /= 10
    test_acc /= 10
    self.test_losses.append(test_loss)
    self.test_accs.append(test_acc)

    print('Test Loss:  {:.4f}      Test Acc:  {:.4f}\n'
      .format(
        test_loss,
        test_acc
      )
    )

    if epoch % 5 == 0:
      self.plot_img(real_imgs=img.numpy(), dc_imgs=decoding.cpu().detach().numpy(), epoch=epoch)

  def train_test(self):
    for epoch in range(self.num_epochs):
      self.train(epoch)
      self.test(epoch)