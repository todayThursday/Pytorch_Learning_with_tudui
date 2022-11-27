import torch
from torch import nn


class Hello(nn.Module):
    def __init__(self):
        super(Hello, self).__init__()

    def forward(self, input):
        output = input + 1
        return output


hello = Hello()
x = torch.tensor(1.0)
output = hello(x)
print(output)
