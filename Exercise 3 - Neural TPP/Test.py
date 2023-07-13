import numpy as np
import torch

a = torch.tensor([[0.325, 0., 0., 0., -3., 0., 0., 0., 0., 0.],
        [0.8, 5., 0., 0., 0., 0., 9., 0., 0., 0.],
        [0., 0., 1.77, 0., 0., 0., 0., 0., 0., 0.6],
        [0., -4.3, 0., 0., 0., 4., 0., 0., 7., 0.],
        [0., 1.2, 0.8, 0., 0., 0., 0., 0., 0., 0.0]])
print(a)
#batch = torch.tensor(a)
print(a.size(0))

b = torch.tensor([True])

b = b.expand(a.size(0), a.size(1))

zero = torch.zeros(1)
mask1 = (a != zero).bool()
mask2 = (a != 0).bool()
print(mask1, mask2)

print(a[0, 0])
for i in range(a.size(0)):
    for j in range(a.size(1)):
        if a[i, j] == torch.tensor(0.8):
            print(i,j)

print(b)

grwhw = torch.tensor([-10000])
print(grwhw)
'''
mask = torch.tensor([True])

    mask = mask.expand(batch.size(0), batch.size(1))

    for i in range(batch.size(0)):
        for j in range(batch.size(1)):
            if batch[i][j] == 0:
                mask[i][j] = False

'''