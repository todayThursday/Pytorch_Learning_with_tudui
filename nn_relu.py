import torch
import torchvision
from torch import nn
from torch.nn import ReLU, Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

class ben(nn.Module):
    def __init__(self):
        super(ben, self).__init__()
        self.relu1=ReLU(inplace=False)
        self.sigmoid1=Sigmoid()

    def forward(self,input):
        output=self.sigmoid1(input)
        return output

data=torchvision.datasets.CIFAR10('./data',train=False,transform=torchvision.transforms.ToTensor(),download=True)
dataloader=DataLoader(data,batch_size=4)


input=torch.tensor([[1,-0.5],
                    [-1,3]])

output=torch.reshape(input,(-1,1,2,2))
print(output.shape)
ben=ben()
output=ben(input)
print(output)

step=0
writer=SummaryWriter('logs_sigmoid')
for data in dataloader:
    imgs,targets=data
    writer.add_images('input',imgs,step)
    output=ben(imgs)
    writer.add_images('output',output,step)
    step=step+1

writer.close()


