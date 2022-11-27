import torch
import torchvision.models
from torch import nn
from torch.nn import Conv2d

vgg16 = torchvision.models.vgg16(pretrained=False)
# 1模型结构+模型参数
torch.save(vgg16,'vgg16_1.pth')
# 2m模型参数（官方推荐）
torch.save(vgg16.state_dict(),'vgg16_2.pth')
#陷阱
class Ben(nn.Module):
    def __init__(self):
        super(Ben, self).__init__()
        self.conv1=Conv2d(3,64,kernel_size=3)

    def forward(self,x):
        x=self.conv1(x)
        return x

ben=Ben()
torch.save(ben,'vgg16_3.pth')
