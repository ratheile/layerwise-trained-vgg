from torch import nn

class FCView(nn.Module):
    def __init__(self):
        super().__init__()

    # noinspection PyMethodMayBeStatic
    def forward(self, x):
        n_b = x.data.size(0)
        x = x.view(n_b, -1)
        return x

    def __repr__(self):
        return 'view(nB, -1)'