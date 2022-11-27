import torch
import torchvision
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

data=torchvision.datasets.CIFAR10('./data',train=False,download=True,transform=torchvision.transforms.ToTensor())

dataloader=DataLoader(data,batch_size=64)



class HE(nn.Module):
    def __init__(self):
        super(HE, self).__init__()
        self.maxpool=MaxPool2d(kernel_size=3,ceil_mode=False)

    def forward(self,input):
        output=self.maxpool(input)
        return output

helo=HE()
step=0
writer=SummaryWriter('logs_maxpool')
for da in dataloader:
    imgs,targets=da
    writer.add_images('input',imgs,step)
    out=helo(imgs)
    writer.add_images('out',out,step)
    step=step+1

writer.close()
