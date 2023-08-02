import os
import platform
import time

from torch.utils.tensorboard import SummaryWriter
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import matplotlib.pyplot as plt
import torchvision
from arch_code import arch_code
from DGNAS.nas_util import toMat
from DGNAS.nasnetwork import SuperNet

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import cv2
import numpy as np
import torch #直接包含整个包
import torch.nn as nn #
import torch.optim as optim
import torch.nn.functional as F#激活函数
from sklearn import decomposition
from torchvision import transforms
from torchvision import datasets#数据集
from torch.utils.data import DataLoader#数据集加载
#1.以上是加载库

BATCH_SZ=32#由于数据量很大，要分批读入内存，代表一批有多少数据
cudaId=0
EPOCHS_CNT=10 #训练的轮次
print(torch.cuda.is_available())
# writer=SummaryWriter(log_dir="../logs", flush_secs=60)#指定存储目录和刷新间隔

class RESNET(nn.Module):#必须继承nn.Module
    def __init__(self):#其中，numblocks是一个列表代表每层网络的块数
        super().__init__()
        self.net=torchvision.models.resnet18(pretrained=False)

        print(self.net)
        # for param in self.net.parameters():
        #     param.requires_grad = False
        self.net.fc = nn.Linear(512, 10)

    def forward(self,x):#重写前向传播算法，x是张量
        #3*224*224
        x=self.net(x)
        # x=x.view(x.size(0),-1)
        # print(x.shape)
        # x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x
    def fill0(self):
        self.net.conv1.weight.data.zero_()
        self.net.layer4[0].conv1.weight.data.zero_()
        self.net.fc.weight.data.zero_()
def adjustOpt(optimizer,epoch):
    p = 0.9
    if epoch%4==0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= p

def train_model(model,train_loader,optimizer,epoch):#model是模型，device是设备，train_loader是数据集，optimizer是梯度，epoch当前第几轮
    model.train()
    correct=0#正确率
    avgLoss=0
    # adjustOpt(optimizer, epoch)
    for batch_index, (d,t) in enumerate(train_loader):
        data,target=d.cuda(cudaId),t.cuda(cudaId)#部署到device
        optimizer.zero_grad()#初始化梯度为0
        output=model(data)#训练后结果
        loss=F.cross_entropy(output,target)#计算损失,用交叉熵,默认是累计的
        loss.backward()#反向传播
        avgLoss +=loss.item()
        optimizer.step()#参数优化
        pred = output.max(1, keepdim=True)[1]  # 找到概率值最大的下标,这里调用了python的函数，也可以自己写
        correct += pred.eq(target.view_as(pred)).sum().item()
    avgLoss /=len(train_loader.dataset)  # 计算平均数
    Accuracy=100 * correct / len(train_loader.dataset)
    print("第{}，正确率{:.6f}  loss：{:.6f}".format(epoch,Accuracy,avgLoss))
    # writer.add_scalar('Accuracy', Accuracy, epoch)
    # writer.add_scalar('avgLoss', avgLoss, epoch)

# 6.以上是定义训练方法
def test_model(model,test_loader):##model是模型，device是设备，test_loader是测试数据集
    model.eval()#模型验证
    correct=0#正确率
    sumLoss=0#测试损失
    with torch.no_grad():#，不进行训练时
        for batch_index, (data, target) in enumerate(test_loader):
            data,target=data.cuda(cudaId),target.cuda(cudaId)#部署到device
            output=model(data)#训练后结果
            pred = output.max(1, keepdim=True)[1]  # 找到概率值最大的下标,这里调用了python的函数，也可以自己写
            correct+=pred.eq(target.view_as(pred)).sum().item()
        print("正确率：{:.6f}".format(100*correct/len(test_loader.dataset)))            #输出


pipeline =transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
# print(torch.cuda.is_available())
train_set=datasets.CIFAR10("picsData/cifar10",train=True,download=False,transform=pipeline)
test_set=datasets.CIFAR10("picsData/cifar10",train=False,download=False,transform=pipeline)
train_loader = DataLoader(train_set,batch_size=BATCH_SZ,shuffle=True)#加载数据集，shuffle=True是打乱数据顺序
test_loader = DataLoader(test_set,batch_size=BATCH_SZ,shuffle=False)
# model=SuperNet(arch_code).cuda(cudaId)
# model=RESNET().cuda(cudaId)

# start_time = time.time()
# for epoch in range(0, 1):
#     test_model(model,DEVICE,test_loader)
# end_time = time.time()
# t1=end_time-start_time

# model.fill0()
# for epoch in range(0,100):
#     train_model(model,train_loader,optimizer,epoch)
# torch.save(model, "nasData/model_supernet.pt")#保存模型pt || pth
model = torch.load("nasData/model_supernet.pt")#加载神经网络
optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum=0.9)
# print(model)
start_time = time.time()
for epoch in range(0, 1):
    test_model(model,test_loader)
end_time = time.time()
t1=end_time-start_time

# for i in range(5):
#     for j in range(len(arch_code[i])):
#         if j!=9 and j!=10:
#             arch_code[i][j]=0
for i in range(5):
    # for j in range(len(arch_code[i])):
    arch_code[i][10]=0
model.setArch(arch_code)
model.cuda()
# print(model)
for epoch in range(0, 1):
    test_model(model,test_loader)
for epoch in range(0,10):
    train_model(model,train_loader,optimizer,epoch)
for epoch in range(0, 5):
    test_model(model,test_loader)

# for epoch in range(0,100):
#     train_model(model,train_loader,optimizer,epoch)
# torch.save(model, "nasData/model_supernet.pt")#保存模型pt || pth

start_time = time.time()
for epoch in range(0, 1):
    test_model(model,test_loader)
end_time = time.time()
t2=end_time-start_time
print(t1,t2)
