from modules import Autoencoder
from torch.autograd import Variable
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch import nn
from torch.optim import Adam

class AutoencoderNet():

  dataloader: DataLoader = None
  learning_rate = 1e-3
  num_epochs = 100

  def __init__(self, dataloader):
    self.dataloader = dataloader
    self.model = Autoencoder()

    if not os.path.exists('./dc_img'):
        os.mkdir('./dc_img')

  def save(self):
    torch.save(
      self.model.state_dict(),
      './conv_autoencoder.pth'
    )

  def to_img(self, x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x
  
  def train(self):
    criterion = nn.MSELoss()
    optimizer = Adam(
      self.model.parameters(), 
      lr=self.learning_rate, 
      weight_decay=1e-5
    )
    for epoch in range(self.num_epochs):
      for data in self.dataloader:
          img, _ = data
          img = Variable(img)
          # ===================forward=====================
          output = self.model(img)
          loss = criterion(output, img)
          # ===================backward====================
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
      # ===================log========================
      print('epoch [{}/{}], loss:{:.4f}'
            .format(epoch+1, self.num_epochs, loss.item()))
      if epoch % 10 == 0:
          pic = self.to_img(output.cpu().data)
          save_image(pic, './dc_img/image_{}.png'.format(epoch))
      