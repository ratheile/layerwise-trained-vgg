from modules import Autoencoder, OriginalAutoencoder, SupervisedAutoencoder
from loaders import MnistLoader,SetType 

import torch
from torch.autograd import Variable
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch import nn 
from torch import save as torch_save
from torch import max as torch_max
from torch.optim import Adam

from torchsummary import summary

import os

import numpy as np
import matplotlib.pyplot as plt

class AutoencoderNet():

  test_loader: DataLoader = None
  train_loader: DataLoader = None
  learning_rate = 1e-3
  num_epochs = 100
  device = 'cpu'
  test_losses = []

  def __init__(self, mnist_path):
    self.test_loader = MnistLoader(mnist_path, download=True, type=SetType.TEST)
    self.train_loader = MnistLoader(mnist_path, download=True, type=SetType.TRAIN)

    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x
  
  def plot_tensor(self, img):
    np_img = img.cpu().numpy()[0]
    np_img = np.transpose(np_img, (1,2,0)) 
    plt.imshow(np_img.squeeze())
    plt.show()


  def save_tensor(self, real_imgs, dc_imgs):
    real_imgs = real_imgs[:5,:,:,:]
    dc_imgs = dc_imgs[:5,:,:,:]
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True, figsize=(25,4))

    for imgs, row in zip([real_imgs, dc_imgs], axes):
      for img, ax in zip(imgs, row):
        image = np.squeeze(img)
        ax.imshow(image, cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.savefig("DecodedImgs.png")

  def train(self, epoch):
    for data in self.test_loader:
      img, labels = data
      img = Variable(img)
      # ===================forward=====================
      dev_img = img.to(self.device)
      dev_labels = labels.to(self.device)

      decoding, prediction = self.model(dev_img)
      loss_dc = self.decoding_criterion(decoding, dev_img)
      loss_pred = self.pred_criterion(prediction, dev_labels)
      
      combo_loss = loss_dc + loss_pred
      # ===================backward====================
      self.optimizer.zero_grad()
      combo_loss.backward()
      self.optimizer.step()
    # ===================log========================
    # calculate Accuracy
    _, predicted = torch_max(prediction.data, 1)
    accuracy = (predicted.cpu().numpy() == labels.numpy()) \
      .sum() / len(labels)

    print('epoch [{}/{}], loss:{:.4f}, acc:{:.4f}'
      .format(
          epoch+1,
          self.num_epochs, 
          combo_loss.item(),
          accuracy
        )
      )

    # self.plot_tensor(img)
    # self.plot_tensor(decoding.detach())

    if epoch % 10 == 0:
      pic = self.to_img(decoding.cpu().data)
      save_image(pic, './dc_img/image_{}.png'.format(epoch))
    
  def test(self, epoch):
    self.model.eval()
    test_loss = 0

    with torch.no_grad():
      for data in self.test_loader:
        imgs, labels = data[0].to(self.device), data[1].to(self.device)
        # ===================Forward=====================
        decoding, prediction = self.model(imgs)
        test_loss += self.decoding_criterion(decoding, imgs).item()

      if epoch % 1 == 0:
        self.save_tensor(real_imgs=imgs.cpu().numpy(), dc_imgs=decoding.detach().cpu().numpy())

    test_loss /= 10
    self.test_losses.append(test_loss)

    print('Epoch [{}/{}], Test Loss:{:.4f}'
      .format(
        epoch+1,
        self.num_epochs,
        test_loss
      )
    )

  def train_test(self):
    for epoch in range(self.num_epochs):
      self.train(epoch)
      self.test(epoch)
      