from modules import StackableNetwork, NetworkStack, SidecarMap

from .layer_training_def import LayerTrainingDefinition
from .cfg_to_network import cfg_to_network

from loaders import semi_supervised_mnist, semi_supervised_cifar10
from loaders import ConfigLoader

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
import logging

import time
import os
import io

import numpy as np
import matplotlib.pyplot as plt

from typing import List, IO

def ensure_dir(path: str):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_layer(layer: nn.Module, path:str):
  logging.info('### Stored layer as {} ###'.format(path))
  ensure_dir(path)
  torch_save(layer.state_dict(), path)

class AutoencoderNet():

  def __init__(self, gcfg: ConfigLoader, rcfg: ConfigLoader):

    if gcfg['device'] == 'cuda':
      self.device = torch.device(
          "cuda" if torch.cuda.is_available() else "cpu"
        )

    self.device        = gcfg['device']
    self.learning_rate = rcfg['learning_rate']
    self.weight_decay  = rcfg['weight_decay']
    self.test_every_n_epochs = rcfg['test_every_n_epochs']
    self.pred_loss_weight = rcfg['pred_loss_weight']

    color_channels = rcfg['color_channels']
    data_path = gcfg['datasets/{}/path'.format(rcfg['dataset'])]

    self.supervised_loader, self.unsupvised_loader,\
    self.test_loader = rcfg.switch('dataset', {
      'cifar10': lambda: semi_supervised_cifar10(
        data_path,
        supervised_ratio=rcfg['supervised_ratio'],
        batch_size=rcfg['batch_size']
      ),
      'mnist': lambda: semi_supervised_mnist(
        data_path,
        supervised_ratio=rcfg['supervised_ratio'],
        batch_size=rcfg['batch_size']
      )})

    self.layer_configs = cfg_to_network(gcfg, rcfg)
    assert len(self.supervised_loader) == len(self.unsupvised_loader)

    self.writer = SummaryWriter()
    """
    TODO: Implement in factory method to show sizes when network is built
    summary(self.model, input_size=(color_channels,img_size,img_size))
    """

    self.decoding_criterion = rcfg.switch('decoding_criterion', {
      'MSELoss': lambda: nn.MSELoss(),
    })

    self.pred_criterion = rcfg.switch('prediction_criterion', {
      'CrossEntropyLoss': lambda: nn.CrossEntropyLoss()
    })

    self.test_losses = []
    self.test_accs =  []

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


    self.writer.add_figure('decoded_imgs', fig, global_step=epoch)

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
      with torch.no_grad():
        t_start = time.process_time_ns()
        dev_img_us = config.stack.upwards(img_us.to(self.device))
        dev_img_s = config.stack.upwards(img_s.to(self.device))
        t_end = time.process_time_ns()
        t_total = t_start - t_end
        logging.info("Upsream calculates in: {t_total}")
      dev_label = label_s.to(self.device)
      
      decoding_s, prediction = config.model(dev_img_s)
      decoding_us, _ = config.model(dev_img_us)

      loss_dc_s = self.decoding_criterion(decoding_s, dev_img_s)
      loss_dc_us = self.decoding_criterion(decoding_us, dev_img_us)

      loss_pred = self.pred_criterion(prediction, dev_label)

      alpha = self.pred_loss_weight
      combo_loss = (loss_dc_s + loss_dc_us) * (1-alpha) + loss_pred * alpha

      # ===================backward====================
      config.optimizer.zero_grad()
      combo_loss.backward()
      config.optimizer.step()

    # ===================log========================
    # Calculate Accuracy
    _, predicted = torch_max(prediction.data, 1)
    accuracy = (predicted.cpu().numpy() == label_s.numpy()).sum() / len(label_s)

    logging.info('Epoch [{}/{}] Train Loss:{:.4f} Train Acc:{:.4f}'
      .format(
          epoch+1,
          config.num_epochs,
          combo_loss.item(),
          accuracy
        )
      )
    
    self.writer.add_scalar('train_loss', combo_loss.item(), global_step=global_epoch)
    self.writer.add_scalar('train_accuracy', accuracy, global_step=global_epoch)
    
  def test(self, 
          epoch: int, 
          global_epoch:int, 
          config: LayerTrainingDefinition,
          plot_every_n_epochs=1):

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
        dev_img = config.stack.upwards(img.to(self.device))
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

    logging.info('Epoch [{}/{}] Test Loss:{:.4f} Test Acc:{:.4f}'
      .format(
        epoch + 1,
        config.num_epochs,
        test_loss,
        test_acc
      )
    )

    self.writer.add_scalar('test_loss', test_loss, global_step=global_epoch)
    self.writer.add_scalar('test_accuracy', test_acc, global_step=global_epoch)

    if epoch % plot_every_n_epochs == 0:
      self.plot_img(real_imgs=dev_img.cpu().numpy(),
                    dc_imgs=decoding.cpu().detach().numpy(),
                    epoch=global_epoch)

  def train_test(self):
    total_epochs = 0
    for id_c, config in enumerate(self.layer_configs): # LayerTrainingDefinition
      if config.pretraining_load is None:
        logging.info('### Training layer {} ###'.format(id_c)) 
        for epoch in range(config.num_epochs):
          self.train(epoch, config=config, global_epoch=total_epochs)
          total_epochs += 1
          if total_epochs % self.test_every_n_epochs == 0:
            self.test(epoch, config=config, global_epoch=total_epochs)
        # end epoch loop


        if config.pretraining_store is not None:
          base = '{}/{}'.format(
            config.pretraining_store, 
            config.layer_name
          )
          fn = f'{base}_stack.pickle'
          save_layer(config.stack, fn)