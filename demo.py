# 利用训练好的模型，给他提供输入
import torch
import torchvision.transforms
from PIL import Image
from torch import nn
from torchvision import transforms

image_path = './imgs/dog.png'
image = Image.open(image_path)
print(image)

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 24)),
                                            torchvision.transforms.ToTensor()])
image = transform(image)
print(image.shape)


class BEN(nn.Module):
    def __init__(self):
        super(BEN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)

        return x


model = torch.load('ben_0.pth',map_location=torch.device('cpu'))
print(model)
image = torch.reshape(image, (1, 3, 32, 32))
model.eval()
with torch.no_grad():
    out = model(image)
print(out)

print(out.argmax(1))
