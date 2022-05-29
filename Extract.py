import numpy as np
import torch

array_data = np.load('small_data.npy').reshape(81, 10, 512, 512)
Train_data = torch.zeros([9, 9, 1, 8, 60, 60])
for i in range(9):
    for b in range(9):
        for j in range(8):
            for k in range(80):
                for l in range(80):
                    Train_data[i][b][0][j][k][l] = array_data[i][j + 1][196 + k][206 + l]
torch.save(Train_data, 'data.pt')
del array_data

target = torch.tensor(
    [[1, 6, 2, 2, 1, 2, 1, 1, 0],
     [2, 1, 0, 1, 3, 2, 3, 2, 3],
     [4, 1, 2, 4, 4, 1, 1, 1, 1],
     [4, 2, 4, 2, 1, 2, 1, 5, 1],
     [1, 3, 6, 2, 1, 3, 4, 1, 1],
     [1, 2, 1, 1, 1, 1, 1, 2, 5],
     [1, 3, 2, 3, 4, 6, 1, 1, 1],
     [2, 3, 1, 2, 1, 1, 2, 2, 2],
     [2, 2, 2, 6, 2, 2, 2, 2, 3]])
torch.save(target, 'target.pt')
