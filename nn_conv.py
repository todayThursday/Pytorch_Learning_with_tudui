import torch
import torch.nn.functional as F

in_put=torch.tensor([[1,2,0,3,1],
                [0,1,2,3,1],
                [1,2,1,0,0],
                [5,2,3,1,1],
                [2,1,0,1,1]])

kernal=torch.tensor([[1,2,1],
                     [0,1,0],
                     [2,1,0]])

in_put=torch.reshape(in_put,(1,1,5,5))
kernal=torch.reshape(kernal,(1,1,3,3))
print(in_put.shape)
print(kernal.shape)
output=F.conv2d(in_put,kernal,stride=1)
output2=F.conv2d(in_put,kernal,stride=2)
output3=F.conv2d(in_put,kernal,stride=1,padding=5)
print(output3)