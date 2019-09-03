from modules import Autoencoder, OriginalAutoencoder, SupervisedAutoencoder
from torch.autograd import Variable
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch import nn 
from torch import save as torch_save
from torch import max as torch_max
from torch.optim import Adam
import os

import numpy as np
import matplotlib.pyplot as plt

class AutoencoderNet():

  dataloader: DataLoader = None
  learning_rate = 1e-3
  num_epochs = 100

  def __init__(self, dataloader):
    self.dataloader = dataloader
    self.model = SupervisedAutoencoder()

    if not os.path.exists('./dc_img'):
        os.mkdir('./dc_img')

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

  def train(self):
    decoding_criterion = nn.MSELoss()
    pred_criterion = nn.CrossEntropyLoss()
    optimizer = Adam(
      self.model.parameters(), 
      lr=self.learning_rate, 
      weight_decay=1e-5
    )

    for epoch in range(self.num_epochs):
      for data in self.dataloader:
        img, labels = data
        img = Variable(img)
        # ===================forward=====================
        decoding, prediction = self.model(img)
        loss_dc = decoding_criterion(decoding, img)
        loss_pred = pred_criterion(prediction, labels)
        
        combo_loss = loss_dc + loss_pred
        # ===================backward====================
        optimizer.zero_grad()
        combo_loss.backward()
        optimizer.step()
      # ===================log========================
      # calculate Accuracy
      _, predicted = torch_max(prediction.data, 1)
      accuracy = (predicted.numpy() == labels.numpy()) \
        .sum() / len(labels)

      print('epoch [{}/{}], loss:{:.4f}, acc:{:.4f}'
        .format(
            epoch+1,
            self.num_epochs, 
            combo_loss.item(),
            accuracy
          )
        )

      self.plot_tensor(img)
      self.plot_tensor(decoding.detach())

      if epoch % 10 == 0:
        pic = self.to_img(decoding.cpu().data)
        save_image(pic, './dc_img/image_{}.png'.format(epoch))
      