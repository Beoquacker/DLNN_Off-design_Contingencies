import torch
import torch.nn as nn
from torchdiffeq import odeint


def norm(dim):
    return nn.GroupNorm(dim, dim)


class ConcatConv3d(nn.Module):

    def __init__(self, dim_in, dim_out, ksize=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), dilation=(1, 1, 1),
                 groups=1, bias=True, transpose=False):
        super(ConcatConv3d, self).__init__()
        module = nn.ConvTranspose3d if transpose else nn.Conv3d
        self._layer = module(
            dim_in + 1, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )

    def forward(self, t, x):
        tt = torch.ones_like(x[:, :1, :, :]) * t
        ttx = torch.cat([tt, x], 1)
        return self._layer(ttx)


class ODEfunc(nn.Module):

    def __init__(self, dim):
        super(ODEfunc, self).__init__()
        self.norm1 = norm(dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = ConcatConv3d(dim, dim, (1, 3, 3), (1, 1, 1), (0, 1, 1))
        self.norm2 = norm(dim)
        self.conv2 = ConcatConv3d(dim, dim, (1, 3, 3), (1, 1, 1), (0, 1, 1))
        self.norm3 = norm(dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(t, out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(t, out)
        out = self.norm3(out)
        return out


class ODEBlock(nn.Module):

    def __init__(self, odefunc):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, 1]).float()

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        out = odeint(self.odefunc, x, self.integration_time, rtol=0.001, atol=0.001)
        return out[1]

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value


class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)


Data = torch.load('data.pt')
target = torch.load('target.pt')

if __name__ == '__main__':

    downsampling_layers = [
        nn.Conv3d(1, 200, (1, 5, 5), (1, 3, 3), (0, 2, 2), groups=1),
        nn.GroupNorm(100, 200),
        nn.ReLU(inplace=True),
        nn.Conv3d(200, 200, (1, 8, 8), (1, 1, 1), (0, 1, 1), groups=1),
        nn.GroupNorm(100, 200),
        nn.ReLU(inplace=True),
        nn.Conv3d(200, 200, (1, 6, 6), (1, 1, 1), (0, 0, 0), groups=1),
    ]
    feature_layers = [ODEBlock(ODEfunc(200))]

    fc_layers = [
        nn.GroupNorm(100, 200),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool3d((1, 1, 1)),
        Flatten(),
        nn.Linear(200, 7)
    ]

    model = nn.Sequential(*downsampling_layers, *feature_layers, *fc_layers)
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    for epoch in range(1, 100 + 1):
        print('epoch number: ' + str(epoch))
        diag = 0
        inter = 0
        for ind in range(9):
            optimizer.zero_grad()
            x = Data[ind]
            y = target[ind]
            logits = model(x)
            loss = criterion(logits, y)

            for i in range(9):
                pred = 0
                maxim = -1
                for j in range(7):
                    if maxim < logits[i][j]:
                        pred = j
                        maxim = logits[i][j]
                if pred == y[i]:
                    diag += 1
                if (pred < 4 and y[i] < 4) or (pred >= 4 and y[i] >= 4):
                    inter += 1

            nfe_forward = feature_layers[0].nfe
            feature_layers[0].nfe = 0
            loss.backward()
            optimizer.step()
            nfe_backward = feature_layers[0].nfe
            feature_layers[0].nfe = 0

        print("epoch's " + str(epoch) + " final accuracy:  " + "{:.3f}".format((diag / 81)))
        print("epoch's " + str(epoch) + " final diagnosis: " + "{:.3f}".format((inter / 81)))
    torch.save(model, "ODENet1.pt")
