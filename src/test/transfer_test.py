import unittest   # The test framework
import torch
import numpy as np
import pandas as pd
from modules import RandomMap

class Test_Transfer_Function(unittest.TestCase):

  def test_random_map_3d(self):
    N = 5
    M = 3
    tmap = RandomMap((N,N,N),(M, M,M))
    G_t = torch.from_numpy(np.random.randint(
      1,10, size=(N,N,N)).astype(np.float32))
    print(tmap.forward(G_t))


  def test_random_map_3d_batch(self):
    N = 5
    M = 3
    B = 2
    tmap = RandomMap((N,N,N),(M,M,M))
    G_t = torch.from_numpy(np.random.randint(
      1,10, size=(B,N,N,N)).astype(np.float32))
    print(tmap.forward(G_t))

  def test_random_map_2d(self):
    N = 5
    M = 3
    tmap = RandomMap((N,N),(M, M))
    # print(pd.DataFrame(tmap.P_st.numpy()))
    x = np.linspace(0,N-1,N)
    y = np.linspace(0,N-1,N)
    XX, YY = np.meshgrid(x, y)
    G = XX * YY
    G_t = torch.from_numpy(G.astype(np.float32))
    print(G_t)
    print(tmap.forward(G_t))

  def test_forward_view(self):
    N = 5
    M = 3
    B = 2
    N_shape = (N,N,N)
    M_shape = (M,M,M)
    tmap = RandomMap(N_shape,M_shape)
    ones = np.ones(shape=N_shape)
    batch = np.stack([ones * 1, ones * 2, ones * 3])
    batch = batch.astype(np.float32)
    G_t = torch.from_numpy(batch)

    self.assertEqual(G_t[0,:,:,:].sum(), N*N*N)
    self.assertEqual(G_t[1,:,:,:].sum(), N*N*N*2)
    self.assertEqual(G_t[2,:,:,:].sum(), N*N*N*3)