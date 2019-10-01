#%%
import unittest   # The test framework
import torch

class Test_Transfer_Function(unittest.TestCase):
  def test_upstream_decoupling(self):
    i = torch.randn(2, 2)
    i.requires_grad = True
    a = ((i * 3) / (i - 1))
    a.retain_grad()
    b = (a * a).sum()
    b.retain_grad()
    b.backward()
    print(a.grad)
    print(i.grad)
    print(b.grad)

test = Test_Transfer_Function().test_upstream_decoupling()
#%%
