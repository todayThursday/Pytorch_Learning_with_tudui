import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class BEN(nn.Module):
    def __init__(self):
        super(BEN, self).__init__()
        self.conv1=Conv2d(3,32,5,stride=1,padding=2)
        self.maxpool1=MaxPool2d(2)
        self.conv2=Conv2d(32,32,5,stride=1,padding=2)
        self.maxpool2=MaxPool2d(2)
        self.conv3=Conv2d(32,64,5,stride=1,padding=2)
        self.maxpool3=MaxPool2d(2)
        self.flatten=Flatten()
        self.linear1=Linear(1024,64)
        self.linear2=Linear(64,10)

        self.mymoudle=Sequential(
            Conv2d(3, 32, 5, stride=1, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, stride=1, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, stride=1, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self,x):
        x=self.mymoudle(x)
        # x=self.conv1(x)
        # x=self.maxpool1(x)
        # x=self.conv2(x)
        # x=self.maxpool2(x)
        # x=self.conv3(x)
        # x=self.maxpool3(x)
        # x=self.flatten(x)
        # x=self.linear1(x)
        # x=self.linear2(x)

        return x

dataset=torchvision.datasets.CIFAR10('./data',train=False,transform=torchvision.transforms.ToTensor(),
                                     download=True)
data=DataLoader(dataset,batch_size=1)

# writer=SummaryWriter('./loss')
ben=BEN()
# step=0
loss=nn.CrossEntropyLoss()
for da in data:
    imgs,targets=da
    # writer.add_image('in',imgs,step)
    out=ben(imgs)
    # writer.add_images('out',out,step)
    lossh=loss(out,targets)
    lossh.backward()
    print('ok')
# writer.close()