import torch
import torchvision.models
from torch import nn

# 1
# mo=torch.load('vgg16_1.pth')
# print(mo)
# 2
vgg16=torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load('vgg16_2.pth'))
# print(vgg16)
#陷阱

class Ben(nn.Module):
    def __init__(self):
        super(Ben, self).__init__()
        self.conv1=Conv2d(3,64,kernel_size=3)

    def forward(self,x):
        x=self.conv1(x)
        return x

mo=torch.load('vgg16_3.pth')
print(mo)