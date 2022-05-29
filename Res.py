import torch
import torch.nn as nn


def norm(dim):
    return nn.GroupNorm(dim, dim)


def conv1x3x3(in_planes, out_planes, stride=(1, 1, 1)):
    return nn.Conv3d(in_planes, out_planes, kernel_size=(1, 3, 3), stride=stride, padding=(0, 1, 1), bias=False)


def conv1x1x1(in_planes, out_planes, stride=(1, 1, 1)):
    return nn.Conv3d(in_planes, out_planes, kernel_size=(1, 1, 1), stride=stride, bias=False)


class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.norm1 = norm(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.conv1 = conv1x3x3(inplanes, planes, stride)
        self.norm2 = norm(planes)
        self.conv2 = conv1x3x3(planes, planes)

    def forward(self, x):
        shortcut = x

        out = self.relu(self.norm1(x))

        if self.downsample is not None:
            shortcut = self.downsample(out)

        out = self.conv1(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + shortcut


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
        nn.Conv3d(1, 200, (1, 8, 8), (1, 5, 5), (0, 2, 2), groups=1),
        nn.GroupNorm(100, 200),
        nn.ReLU(inplace=True),
        nn.Conv3d(200, 200, (1, 7, 7), (1, 1, 1), (0, 1, 1), groups=1),
        nn.GroupNorm(100, 200),
        nn.ReLU(inplace=True),
        nn.Conv3d(200, 200, (1, 6, 6), (1, 1, 1), (0, 0, 0), groups=1),
    ]
    feature_layers = [ResBlock(200, 200) for _ in range(6)]

    fc_layers = [
        nn.GroupNorm(100, 200),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool3d((1, 1, 1)),
        Flatten(),
        nn.Linear(200, 7),
    ]

    model = nn.Sequential(*downsampling_layers, *feature_layers, *fc_layers)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    best_acc = 0
    for epoch in range(1, 100 + 1):
        print('epoch number: ' + str(epoch))
        cnt = 0
        rezat = 0
        for ind in range(9):
            x = Data[ind]
            y = target[ind]
            logits = model(x)
            optimizer.zero_grad()
            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()

            for i in range(9):
                pred = 0
                maxim = -1
                for j in range(7):
                    if maxim < logits[i][j]:
                        pred = j
                        maxim = logits[i][j]
                if pred == y[i]:
                    cnt += 1
                if (pred < 4 and y[i] < 4) or (pred >= 4 and y[i] >= 4):
                    rezat += 1
            if pred >= 4:
                print(str(pred) + ", YES!")

        print("epoch's " + str(epoch) + " final accuracy:  " + "{:.3f}".format((cnt / 81)))
        print("epoch's " + str(epoch) + " final diagnosis: " + "{:.3f}".format((rezat / 81)))
    torch.save(model, "ResNet.pt")
