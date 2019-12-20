import unittest   # The test framework
import torch
from modules import VGG
from modules import SupervisedSidecarAutoencoder
import pandas as pd

class Test_Vgg(unittest.TestCase):

  def test_init_vgg(self):
    vgg = VGG()
    print(vgg.get_trainable_modules())
    self.assertEqual(vgg.get_trainable_modules(), 10)

  def test_init_sidecar_encoder(self):
    vgg = VGG()
    tmod = vgg.get_trainable_modules()
    mod, chan = tmod[0]
    ssae = SupervisedSidecarAutoencoder(
      main_network_layer=mod,
      channels=chan,
      dropout=0.3
    )