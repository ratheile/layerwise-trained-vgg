#%%
import torch
import numpy as np
bce_loss = torch.nn.BCELoss(reduction='none')
mse_loss = torch.nn.MSELoss(reduction='none')
cre_loss = torch.nn.CrossEntropyLoss()

def m_f(arr):
  return ['{:.2f}'.format(a) for a in arr]
# %%
y = torch.tensor(np.array([.8]*11).astype(np.double))
x = torch.tensor(np.arange(0,1.1,0.1).astype(np.double))


print('x  ', m_f(x))
print('y  ', m_f(y))
print('MSE' , m_f(mse_loss(x, y)))
print('BCE' , m_f(bce_loss(x, y)))


# %%
