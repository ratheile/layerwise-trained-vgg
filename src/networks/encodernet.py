from modules import StackableNetwork, NetworkStack, SidecarMap

from modules import SupervisedSidecarAutoencoder, VGG

from loaders import semi_supervised_mnist, semi_supervised_cifar10
from loaders import ConfigLoader

import torch
from torch.autograd import Variable
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch import nn, Tensor 
from torch import cat as torch_cat
from torch import save as torch_save
from torch import load as torch_load
from torch import max as torch_max
from torch.optim import Adam, Optimizer

#logging
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
import logging

import os
import io
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

from typing import List, IO

@dataclass
class LayerTrainingDefinition:
  layer_name: str = None
  #config
  num_epochs: int = 0
  pretraining_store: str = None
  pretraining_load: str = None

  # stack including this layer
  stack: NetworkStack = None

  # this layers elements 
  upstream: nn.Module = None
  model: nn.Module = None

  # other vars
  optimizer: Optimizer = None

def ensure_dir(path: str):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_layer(layer: nn.Module, path:str):
  logging.info('### Stored layer as {} ###'.format(path))
  ensure_dir(path)
  torch_save(layer.state_dict(), path)

def load_layer(layer: nn.Module, path: str):
    return layer.load_state_dict(torch_load(path))

def vgg_sidecar_layer(vgg: VGG, index:int, dropout:float) -> nn.Module:
  vgg_layers, channels, img_size, _ = vgg.get_trainable_modules()[index]
  scae = SupervisedSidecarAutoencoder(vgg_layers, img_size, channels, dropout)
  return scae


def cfg_to_network(gcfg: ConfigLoader, rcfg: ConfigLoader) \
  -> List[LayerTrainingDefinition]:

  num_layers = len(rcfg['layers'])
  device = gcfg['device']
  learning_rate = rcfg['learning_rate']
  weight_decay = rcfg['weight_decay']
  color_channels = rcfg['color_channels']
  vgg_dropout = rcfg['vgg_dropout']
  dataset_name = rcfg['dataset']
  img_size = gcfg[f'datasets/{dataset_name}/img_size']
  num_classes = gcfg[f'datasets/{dataset_name}/num_classes']

  layer_configs = []

  # just initialize VGG, doesnt take much time
  # even when not needed
  vgg = VGG(num_classes=num_classes, dropout=vgg_dropout, img_size=img_size) 

  for id_l, layer in enumerate(rcfg['layers']):

    dropout_rate = layer['dropout_rate']
    model_type = layer['model']
    uprms = layer['upstream_params']

    # Prepare the model
    model = rcfg.switch(f'layers/{id_l}/model', {
      'AE': lambda: SupervisedAutoencoder(
        color_channels=color_channels
      ),
      'VGGn': lambda: vgg_sidecar_layer(vgg, id_l,
        dropout=dropout_rate
      )
    }).to(device)

    # Prepare the upstream for uniform autoencoder networks
    upstream = None
    if id_l < num_layers - 1 and model_type == 'AE':
      upstream = rcfg.switch(f'layers/{id_l}/upstream', {
        'RandomMap': lambda: RandomMap( 
          in_shape=uprms['in_shape'],
          out_shape=uprms['out_shape']
        ),
        'InterpolationMap': lambda: InterpolationMap(),
        'ConvMap': lambda: ConvMap(
          in_shape=uprms['in_shape'],
          out_shape=uprms['out_shape']
        ),
        'DecoderMap': lambda: DecoderMap(model)
      }).to(device)

    # Prepare the upstream for VGG
    elif model_type == 'VGGn':
      _, _, _, upstream_map = vgg.get_trainable_modules()[id_l]
      if upstream_map is not None:
        upstream = SidecarMap([upstream_map])

    prev_stack = [(cfg.model, cfg.upstream) for cfg in layer_configs]
    prev_stack.append((model, upstream))
    stack = NetworkStack(prev_stack).to(device)

    # load stack from pickle if required
    stack_path = layer['pretraining_load']
    if stack_path is not None:
      load_layer(stack, stack_path)

    # some upstream maps require training
    if upstream is not None and upstream.requires_training:
      trainable_params = list(model.parameters()) + list(upstream.parameters())
    else:
      trainable_params = model.parameters()

    optimizer = Adam(
      trainable_params,
      lr=learning_rate,
      weight_decay=weight_decay
    )

    layer_name = f'layer_{id_l}'
        
    layer_configs.append(
      LayerTrainingDefinition(
        layer_name=layer_name,
        num_epochs=layer['num_epoch'], 
        upstream=upstream,
        stack=stack,
        model=model,
        optimizer=optimizer,
        pretraining_store=layer['pretraining_store'],
        pretraining_load=layer['pretraining_load'],
      )
    )
    # end for loop

  return layer_configs


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
        dev_img_us = config.stack.upwards(img_us.to(self.device))
        dev_img_s = config.stack.upwards(img_s.to(self.device))
      dev_label = label_s.to(self.device)
      
      decoding_s, prediction = config.model(dev_img_s)
      decoding_us, _ = config.model(dev_img_us)

      loss_dc_s = self.decoding_criterion(decoding_s, dev_img_s)
      loss_dc_us = self.decoding_criterion(decoding_us, dev_img_us)

      loss_pred = self.pred_criterion(prediction, dev_label)

      alpha = cfg.pred_loss_weight
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