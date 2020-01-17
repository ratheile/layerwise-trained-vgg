r"""
encodernet.py
======================
This module combines all the custom network modules and defines
the schedule how to train and evaluate our networks.

.. autosummary::
  networks.encodernet.AutoencoderNet
"""

from modules import StackableNetwork, NetworkStack, SidecarMap

from .layer_training_def import LayerTrainingDefinition, LayerType
from .cfg_to_network import cfg_to_network

from loaders import semi_supervised_mnist, semi_supervised_cifar10, semi_supervised_cifar100
from loaders import ConfigLoader
from visualizations import CNNLayerVisualization

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
  r"""
  Main Class of this project. Groups all the functions to run a network 
  but is not responsible to assemble the network configuration /  hyperparameters. 
  Most of the functions accept a LayerTrainingDefinition 
  or a ConfigLoader to access the training parameters.
  """

  def __init__(self, gcfg: ConfigLoader, rcfg: ConfigLoader):

    if gcfg['device'] == 'cuda':
      self.device = torch.device(
          "cuda" if torch.cuda.is_available() else "cpu"
        )

    self.device        = gcfg['device']
    self.weight_decay  = rcfg['weight_decay']
    self.test_every_n_epochs = rcfg['test_every_n_epochs']
    self.pred_loss_weight = rcfg['pred_loss_weight']
    self.waves = rcfg['waves']

    color_channels = rcfg['color_channels']

    data_path = gcfg['datasets/{}/path'.format(rcfg['dataset'])]
    dataset_workers = gcfg['dataset_workers']
    dataset_transform = rcfg['dataset_transform']

    self.supervised_loader, self.unsupvised_loader,\
    self.test_loader = rcfg.switch('dataset', {
      'cifar10': lambda: semi_supervised_cifar10(
        data_path,
        dataset_transform,
        supervised_ratio=rcfg['supervised_ratio'],
        batch_size=rcfg['batch_size'],
        augmentation=rcfg['augmentation'],
        num_workers=dataset_workers
      ),
      'cifar100': lambda: semi_supervised_cifar100(
        data_path,
        dataset_transform,
        supervised_ratio=rcfg['supervised_ratio'],
        batch_size=rcfg['batch_size'],
        augmentation=rcfg['augmentation'],
        num_workers=dataset_workers
      ),
      'mnist': lambda: semi_supervised_mnist(
        data_path,
        supervised_ratio=rcfg['supervised_ratio'],
        batch_size=rcfg['batch_size']
      )
      })

    self.layer_configs = cfg_to_network(gcfg, rcfg)
    assert len(self.supervised_loader) == len(self.unsupvised_loader)
    
    model_detail_name = os.path.split(rcfg['model_path'])[-1]
    self.writer = SummaryWriter(comment='_' + model_detail_name)
    self.writer.add_text("run_config", rcfg.to_json())
    self.writer.add_text("environment", gcfg.to_json())

    """
    TODO: Implement in factory method to show sizes when network is built
    summary(self.model, input_size=(color_channels,img_size,img_size))
    """

    self.decoding_criterion = rcfg.switch('decoding_criterion', {
      'MSELoss': lambda: nn.MSELoss(),
      'BCELoss': lambda: nn.BCEWithLogitsLoss()
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
  
  def measure_time(self, t_start):
    current = time.time_ns()
    delta = (current - t_start) / 1e9
    return current, delta

  def train_vgg_classifier(self, epoch: int, global_epoch:int,  config: LayerTrainingDefinition):
    r"""
    The vgg network has its own classification layer with a cost function that differs from the usual
    autoencoder network block. Therefore it has its own training method.
    """

    #TODO: check if still necessary self.model.train()

    tot_t_dataload = 0
    tot_t_loss = 0
    tot_t_optim = 0

    # _s means supervised _us unsupervised
    iter_us = iter(self.unsupvised_loader)
    iter_s = iter(self.supervised_loader)
    n_batches = len(self.unsupvised_loader)

    for ith_batch in range(n_batches):
      t_start = time.time_ns()

      img_s, label_s = (lambda d: (d[0], d[1]))(next(iter_s))

      dev_img_s = img_s.to(self.device)
      dev_label = label_s.to(self.device)

      t_start, t_delta = self.measure_time(t_start)
      tot_t_dataload += t_delta

      # copy all vars to device and calculate the topmost stack representation
      # TODO: avoid calculating this representation twice (here and in forward())

      prediction = config.model(dev_img_s)
      loss_pred = self.pred_criterion(prediction, dev_label)

      t_start, t_delta = self.measure_time(t_start)
      tot_t_loss += t_delta
      # ===================backward====================
      config.optimizer.zero_grad()
      loss_pred.backward()
      config.optimizer.step()

      t_start, t_delta = self.measure_time(t_start)
      tot_t_optim += t_delta
    # ===================log========================
    # Calculate Accuracy
    _, predicted = torch_max(prediction.data, 1)
    accuracy = (predicted.cpu().numpy() == label_s.numpy()).sum() / len(label_s)

    logging.info((
        'Epoch [{}/{}] Train Loss:{:.4f} ' +
        'Train Acc:{:.4f} ' +
        'Time(Loading|Loss|Optim):  {:.2f} {:.2f} {:.2f}'
      ).format(
          epoch+1,
          config.num_epochs,
          loss_pred.item(),
          accuracy,
          tot_t_dataload,
          tot_t_loss,
          tot_t_optim 
        )
      )
    
    self.writer.add_scalar('loss_linear/train', loss_pred.item(), global_step=global_epoch)
    self.writer.add_scalar('accuracy_linear/train', accuracy, global_step=global_epoch)
    
  def test_vgg_classifier(self, 
          epoch: int, 
          global_epoch:int, 
          config: LayerTrainingDefinition,
          plot_every_n_epochs=1):
    r"""
    The vgg network has its own classification layer with a cost function that differs from the usual
    autoencoder network block. Therefore it has its own test method.
    """

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
        dev_img = img.to(self.device)
        dev_label = label.to(self.device)

        # ===================Forward=====================
        prediction = config.model(dev_img)
        loss_pred = self.pred_criterion(prediction, dev_label)
        # Calculate Test Loss
        test_loss += loss_pred.item() / (len(self.test_loader))
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

    self.writer.add_scalar('loss_linear/test', test_loss, global_step=global_epoch)
    self.writer.add_scalar('accuracy_linear/test', test_acc, global_step=global_epoch)

  def train(self, epoch: int, global_epoch:int,  config: LayerTrainingDefinition):
    r"""
    This method is the default train method for a horizontal autoencoder network block.
    It takes care of dataset iteration and loss calculation / optimization.
    It operates on a given config of type LayerTrainingDefinition.
    This code represents what needs to be done in one epoch of training.
    """

    tot_t_dataload = 0
    tot_t_upstream = 0
    tot_t_loss = 0
    tot_t_optim = 0

    # _s means supervised _us unsupervised
    iter_us = iter(self.unsupvised_loader)
    iter_s = iter(self.supervised_loader)
    n_batches = len(self.unsupvised_loader)

    for ith_batch in range(n_batches):
      t_start = time.time_ns()

      img_us, _ = (lambda d: (d[0], d[1]))(next(iter_us))
      img_s, label_s = (lambda d: (d[0], d[1]))(next(iter_s))

      dev_img_us = img_us.to(self.device)
      dev_img_s = img_s.to(self.device)
      dev_label = label_s.to(self.device)

      t_start, t_delta = self.measure_time(t_start)
      tot_t_dataload += t_delta

      # copy all vars to device and calculate the topmost stack representation
      # TODO: avoid calculating this representation twice (here and in forward())

      with torch.no_grad():
        dev_img_us = config.stack.upwards(dev_img_us)
        dev_img_s = config.stack.upwards(dev_img_s)
        t_start, t_delta = self.measure_time(t_start)
        tot_t_upstream += t_delta
      
      decoding_s, prediction = config.model(dev_img_s)
      decoding_us, _ = config.model(dev_img_us)

      loss_dc_s = self.decoding_criterion(decoding_s, dev_img_s)
      loss_dc_us = self.decoding_criterion(decoding_us, dev_img_us)

      loss_pred = self.pred_criterion(prediction, dev_label)

      # tensor names: s = supervised, us = unsupervised, p = prediction(label)
      # a = alpha  r = regulation
      loss_functions = {
        'us*(1-a)+(s+p)*a+r':    lambda s, us, p, a, r: us * (1-a) + (s + p) * a + r,
        '(us+s)*(1-a)+p*a+r':    lambda s, us, p, a, r: (us + s) * (1-a) + p * a + r,
        '(us+10*s)*(1-a)+p*a+r': lambda s, us, p, a, r: (us + 10*s) * (1-a) + p * a + r,
      }

      loss_f = loss_functions[config.ae_loss_function]

      if isinstance(config.tp_alpha, nn.Parameter):
        alpha = 0.5 + (torch.sigmoid(config.tp_alpha) - 0.5)
        reg = - (torch.log(alpha) + torch.log(1-alpha))
      else:
        alpha = config.tp_alpha
        reg = 0

      combo_loss = loss_f(loss_dc_s, loss_dc_us, loss_pred, alpha, reg)

      t_start, t_delta = self.measure_time(t_start)
      tot_t_loss += t_delta
      # ===================backward====================
      config.optimizer.zero_grad()
      combo_loss.backward()
      config.optimizer.step()

      t_start, t_delta = self.measure_time(t_start)
      tot_t_optim += t_delta
    # ===================log========================
    # Calculate Accuracy
    _, predicted = torch_max(prediction.data, 1)
    accuracy = (predicted.cpu().numpy() == label_s.numpy()).sum() / len(label_s)

    logging.info((
        'Epoch [{}/{}] Train Loss:{:.4f} ' +
        'Train Acc:{:.4f} ' +
        'Time(Loading|Upstream|Loss|Optim):  {:.2f} {:.2f} {:.2f} {:.2f}'
      ).format(
          epoch+1,
          config.num_epochs,
          combo_loss.item(),
          accuracy,
          tot_t_dataload,
          tot_t_upstream, 
          tot_t_loss,
          tot_t_optim 
        )
      )
    
    if isinstance(config.tp_alpha, nn.Parameter):
      self.writer.add_scalar('alpha/x', config.tp_alpha.item(), global_step=global_epoch)
      self.writer.add_scalar('alpha/sig_x', alpha.item(), global_step=global_epoch)

    self.writer.add_scalar('loss_total/train', combo_loss.item(), global_step=global_epoch)
    self.writer.add_scalar(f'loss_{config.layer_name}/train', combo_loss.item(), global_step=epoch)
    self.writer.add_scalar('accuracy_total/train', accuracy, global_step=global_epoch)
    self.writer.add_scalar(f'accuracy_{config.layer_name}/train', accuracy, global_step=epoch)
    
  def test(self, 
          epoch: int, 
          global_epoch:int, 
          config: LayerTrainingDefinition,
          plot_every_n_epochs=1):
    r"""
    This method is the default test method for a horizontal autoencoder network block.
    It operates on a given config of type LayerTrainingDefinition.
    The method freezes the current layer and computes the test accuracy.
    """

    test_loss = 0
    test_acc = 0

    # execution times
    t_start = time.time_ns()

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
        # Calculate Test Loss
        test_loss += loss_pred.item() / (len(self.test_loader))
        # Calculate Accuracy
        _, predicted = torch_max(prediction.data, 1)
        test_acc += (predicted.cpu().numpy() == label.numpy()).sum() / (len(self.test_loader) * len(label))

    self.test_losses.append(test_loss)
    self.test_accs.append(test_acc)

    t_start, t_delta = self.measure_time(t_start)
    logging.info('Epoch [{}/{}] Test Loss:{:.4f} Test Acc:{:.4f} Time: {:.2f}'
      .format(
        epoch + 1,
        config.num_epochs,
        test_loss,
        test_acc,
        t_delta
      )
    )

    self.writer.add_scalar('loss_total/test', test_loss, global_step=global_epoch)
    self.writer.add_scalar(f'loss_{config.layer_name}/test', test_loss, global_step=epoch)
    self.writer.add_scalar('accuracy_total/test', test_acc, global_step=global_epoch)
    self.writer.add_scalar(f'accuracy_{config.layer_name}/test', test_acc, global_step=epoch)

    # if epoch % plot_every_n_epochs == 0:
    #   self.plot_img(real_imgs=dev_img.cpu().numpy(),
    #                 dc_imgs=decoding.cpu().detach().numpy(),
    #                 epoch=global_epoch)

  def majority_vote(self, configs: List[LayerTrainingDefinition], global_epoch:int):  
    r"""
    This function implements a voting scheme over all layers.
    It uses a layer-wise softmax prediction. The final decision is taken as argmax
    over the mean of the individual distributions.
    """


    # soft max to compare cross layers, soft max dim 1 == labels
    soft_max = nn.Softmax(dim=1)

    # execution times
    t_start = time.time_ns()
    test_acc = 0
    test_norm = 0

    with torch.no_grad():
      # TODO: Num classes should be a yaml param
      num_classes = 10
      num_batches = len(self.test_loader)
      num_layers = len(configs)

      for data in self.test_loader:
        img, label = data[0], data[1]
        batch_size = len(label) # last batch might be smaller than regular batch size!
        layer_predicted = torch.zeros(batch_size, num_classes).to(self.device)
        for id_c, config in enumerate(configs):
          
          # copy all vars to device and calculate the topmost stack representation
          # TODO: avoid calculating this representation twice (here and in forward())
          dev_img = config.stack.upwards(img.to(self.device))
          dev_label = label.to(self.device)

          # ===================Forward=====================
          decoding, prediction = config.model(dev_img)
          layer_predicted += (soft_max(prediction.data) / num_layers)

        _, predicted = layer_predicted.max(dim=1)

        test_acc += (predicted.cpu().numpy() == label.numpy()).sum()
        test_norm += batch_size

    test_acc = test_acc / test_norm
    t_start, t_delta = self.measure_time(t_start)
    logging.info('Epoch [{}]  Majority Vote Acc:{:.4f} Time: {:.2f}'
      .format(
        global_epoch + 1,
        test_acc,
        t_delta
      )
    )
    self.writer.add_scalar('accuracy_total/maj_vote', test_acc, global_step=global_epoch)


  def wave_train_test(self):
    r"""
    This function defines the overall training schedule over the total number of epochs.
    It trains the layer in a wave-like pattern:

      - Layer 0 is trained for (total_epochs / waves) epochs
      - Layer 1 is trained for (total_epochs / waves) epochs

    after the final layer is trained, we return back to layer 0.

    This method converges much quicker than strict sequential training.
    
    It is called from the main script after class initialization to start the training.
    """
    waves = self.waves
    total_epochs = 0
    layer_epochs = [0]*len(self.layer_configs)
    for id_w in range(waves):
      for id_c, config in enumerate(self.layer_configs): # LayerTrainingDefinition
        if config.pretraining_load is None:
          logging.info('### Training layer {} ###'.format(id_c+1)) 
          wave_epoch_count = int(config.num_epochs / waves)
          epoch = layer_epochs[id_c]
          for wave_epoch in range(wave_epoch_count):
            if config.layer_type == LayerType.Stack:
              self.train(epoch, config=config, global_epoch=total_epochs)
            elif config.layer_type == LayerType.VGGlinear:
              self.train_vgg_classifier(epoch, config=config, global_epoch=total_epochs)
            if total_epochs % self.test_every_n_epochs == 0:
              if config.layer_type == LayerType.Stack:
                self.test(epoch, config=config, global_epoch=total_epochs)
              elif config.layer_type == LayerType.VGGlinear:
                self.test_vgg_classifier(epoch, config=config, global_epoch=total_epochs)
            total_epochs += 1
            epoch += 1
          layer_epochs[id_c] += wave_epoch_count
          # end epoch loop

          # Use this snippet to debug majority vote!
          # valid_layers = list(filter(lambda x: x.layer_type == LayerType.Stack, self.layer_configs))
          # self.majority_vote(valid_layers, global_epoch=total_epochs)

          if config.pretraining_store is True:
            path = '{}/{}_stack.pickle'.format(
              config.model_base_path, 
              config.layer_name
            )
            save_layer(config.stack, path)
        
        else:
          logging.info('### Use pretrained tensors for {} ###'.format(id_c))

      valid_layers = list(filter(lambda x: x.layer_type == LayerType.Stack, self.layer_configs))
      self.majority_vote(valid_layers, global_epoch=total_epochs)

  def train_test(self):
    r"""
    This function defines the overall training schedule over the total number of epochs.
    It trains the layer one by one:
    
    - First layer 0 is trained for n epochs
    - Then layer 1 is trained for n epochs
    - etc...

    It is called from the main script after class initialization to start the training.
    """
    total_epochs = 0
    for id_c, config in enumerate(self.layer_configs): # LayerTrainingDefinition
      if config.pretraining_load is None:
        logging.info('### Training layer {} ###'.format(id_c)) 
        for epoch in range(config.num_epochs):
          if config.layer_type == LayerType.Stack:
            self.train(epoch, config=config, global_epoch=total_epochs)
          elif config.layer_type == LayerType.VGGlinear:
            self.train_vgg_classifier(epoch, config=config, global_epoch=total_epochs)
          total_epochs += 1
          if total_epochs % self.test_every_n_epochs == 0:
            if config.layer_type == LayerType.Stack:
              self.test(epoch, config=config, global_epoch=total_epochs)
            elif config.layer_type == LayerType.VGGlinear:
              self.test_vgg_classifier(epoch, config=config, global_epoch=total_epochs)
        # end epoch loop


        if config.pretraining_store is True:
          path = '{}/{}_stack.pickle'.format(
            config.model_base_path, 
            config.layer_name
          )
          save_layer(config.stack, path)
      
      else:
        logging.info('### Use pretrained tensors for {} ###'.format(id_c))
      
  def visualize(self, epoch = 0): 
    r"""
    Visualize gradients generated with color guided back propagation 
    """
  
    logging.info("## visualize layer ##") 
    
    total_epochs = 0
    lc = len(self.layer_configs)
    img_list = []
    layer_to_hook = 22
    filter_to_hook = 1
    layers = []

    #extract VGG networks from layerstack as sequential 

    for id_c, config in enumerate(self.layer_configs):  
      if id_c == 7: 
        for index in range(0,7):
          layers.append( config.stack.networks_sn[index].upstream_layers[0] )
          layers.append( config.stack.networks_sn[index].upstream_layers[1])
          layers.append( config.stack.networks_sn[index].upstream_layers[2] )

          if not ( config.stack.networks_maps[index] is None):
            # print("Added maxpool") 
            layers.append( config.stack.networks_maps[index].function[0] )
      
    net = nn.Sequential(*layers).cpu() 
    
    #visualize all cnn layers. 
    #write output to tensorboard
    layer_to_vis = [0,3,7,10,14,17,21]
    for f in range (0,10):
      for l_id in layer_to_vis:
      
        layer_vis = CNNLayerVisualization( 
          net , 
          l_id, 
          f)

        imgs_list = layer_vis.visualise_layer_with_hooks(
          epoches = 50, 
          samples = 1) 

        np_data = np.swapaxes(np.squeeze(np.asarray(imgs_list)) ,0,2) 

        self.writer.add_image(
          "conv_layer%d_f%d"%(l_id,f), 
          np_data,
          global_step= f)
  
