import torch.optim
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time
# 网络模型 数据（输入，标注） 损失函数 .cuda()
# from mymo import *

# 准备数据集
train_data = torchvision.datasets.CIFAR10(root='./data', train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root='./data', train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)

# 长度
train_data_size = len(train_data)
test_data_size = len(test_data)

print('训练数据集长度：{}'.format(train_data_size))
print('测试数据集长度：{}'.format(test_data_size))

# 利用dataloader来加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 搭建神经网络
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
# 创建网络模型
ben = BEN()
ben=ben.cuda()

# 创建损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn=loss_fn.cuda()

# 优化器
# learning_rate=0.01
# 1e-2=1x(10)^-2
learning_rate = 1e-2
optimizer = torch.optim.SGD(ben.parameters(), lr=learning_rate)

# 设置训练网络参数


# 训练次数
total_train_step = 0
# 测试次数
total_test_step = 0
# 训练次数
epoch = 10
# tensorboard
writer = SummaryWriter('firstlogs')
start_time=time.time()
for i in range(epoch):
    print("第{}轮训练开始".format(i + 1))

    # 训练
    # ben.train()
    for data in train_dataloader:
        imgs, targets = data
        imgs=imgs.cuda()
        targets=targets.cuda()
        output = ben(imgs)
        loss = loss_fn(output, targets)

        # 优化器
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            end_time=time.time()
            print(end_time-start_time)
            print('训练次数：{},Loss={}'.format(total_train_step, loss.item()))
            writer.add_scalar('train_loss', loss.item(), total_train_step)

    # 测试
    # ben.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            imgs=imgs.cuda()
            targets=targets.cuda()
            output = ben(imgs)
            test_loss = loss_fn(output, targets)
            total_test_loss = total_test_loss + test_loss.item()
            accuracy = (output.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy

    print('整体测试集loss:{}'.format(total_test_loss))
    print('整体测试集上的正确率：{}'.format(total_accuracy / test_data_size))
    writer.add_scalar('test_loss', total_test_loss, total_test_step)
    writer.add_scalar('test_accuracy', total_accuracy / test_data_size, total_test_step)
    total_test_step = total_test_step + 1

    torch.save(ben, './ben_{}.pth'.format(i))
    # torch.save(ben.state_dict()
    print('模型已保存')

writer.close()
