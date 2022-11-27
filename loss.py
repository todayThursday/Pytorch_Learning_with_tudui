import torch
from torch import nn
from torch.nn import L1Loss, MSELoss

inputs=torch.tensor([1,2,3],dtype=float)
targets=torch.tensor([2,4,6],dtype=float)

inputs=torch.reshape(inputs,(1,1,1,3))
targets=torch.reshape(targets,(1,1,1,3))

loss=L1Loss(reduction='sum')
loss_mse=MSELoss()
result=loss(inputs,targets)
result2=loss_mse(inputs,targets)

print(result)
print(result2)

x=torch.tensor([0.1,0.2,0.3])
y=torch.tensor([1])
x=torch.reshape(x,(1,3))
loss_cross=nn.CrossEntropyLoss()
resultss=loss_cross(x,y)
print(resultss)