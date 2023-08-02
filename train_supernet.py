import collections
import math
import os
import platform
import time
import argparse
from torch.utils.tensorboard import SummaryWriter
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import matplotlib.pyplot as plt
import torchvision
from arch_code import getSuperCode
from DGNAS.nas_util import toMat, getParameterCnt, getsubdataloader
from DGNAS.nasnetwork import SuperNet, TorchResNet, netDir

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

LAYER_NUM=-2147483647
# writer=SummaryWriter(log_dir="../logs", flush_secs=60)#指定存储目录和刷新间隔

def adjustOpt(optimizer,epoch):
    p = 0.9
    if epoch%4==0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= p
def distillation_loss(x,y):
    return nn.KLDivLoss(reduction="batchmean")(x,y)
def distill_model(model,teacher_model,train_loader,optimizer,epoch):
    model.train()
    teacher_model.eval()
    correct=0#正确率
    avgLoss=0
    # adjustOpt(optimizer, epoch)
    for batch_index, (d,t) in enumerate(train_loader):
        data,target=d.cuda(args.cudaId),t.cuda(args.cudaId)#部署到device
        optimizer.zero_grad()#初始化梯度为0
        optimizer.zero_grad()#初始化梯度为0
        output,norm=model.forward_norm(data)#训练后结果
        # norm=torch.Tensor(norm)
        # norm0=norm.norm(p=2).sqrt()
        # std = norm.std()
        output2=teacher_model.forward(data).detach()
        output=F.log_softmax(output, dim=1)
        output2 = F.softmax(output2, dim=1)
        loss=distillation_loss(output, output2)
        # print(loss,std)
        loss.backward()#反向传播
        avgLoss +=loss.item()
        optimizer.step()#参数优化
        pred = output.max(1, keepdim=True)[1]  # 找到概率值最大的下标,这里调用了python的函数，也可以自己写
        correct += pred.eq(target.view_as(pred)).sum().item()
    avgLoss /=len(train_loader.dataset)  # 计算平均数
    Accuracy=100 * correct / len(train_loader.dataset)
    print("蒸馏第{}，正确率{:.2f}  loss：{:.12f}".format(epoch,Accuracy,avgLoss))
    return Accuracy,avgLoss
def train_model(model,train_loader,optimizer,epoch):#model是模型，device是设备，train_loader是数据集，optimizer是梯度，epoch当前第几轮
    model.train()
    correct=0#正确率
    avgLoss=0
    # adjustOpt(optimizer, epoch)
    for batch_index, (d,t) in enumerate(train_loader):
        data,target=d.cuda(args.cudaId),t.cuda(args.cudaId)#部署到device
        optimizer.zero_grad()#初始化梯度为0
        output=model.forward(data)#训练后结果
        # print(output)
        loss=F.cross_entropy(output,target)#计算损失,用交叉熵,默认是累计的
        loss.backward()#反向传播
        avgLoss +=loss.item()
        optimizer.step()#参数优化
        pred = output.max(1, keepdim=True)[1]  # 找到概率值最大的下标,这里调用了python的函数，也可以自己写
        correct += pred.eq(target.view_as(pred)).sum().item()
    avgLoss /=len(train_loader.dataset)  # 计算平均数
    Accuracy=100 * correct / len(train_loader.dataset)
    print("第{}，正确率{:.2f}  loss：{:.12f}".format(epoch,Accuracy,avgLoss))
    return Accuracy,avgLoss
    # writer.add_scalar('Accuracy', Accuracy, epoch)
    # writer.add_scalar('avgLoss', avgLoss, epoch)
def test_model(model,test_loader):##model是模型，device是设备，test_loader是测试数据集
    model.eval()#模型验证
    correct=0#正确率
    sumLoss=0#测试损失
    with torch.no_grad():#，不进行训练时
        t0=time.time()
        for batch_index, (data, target) in enumerate(test_loader):
            data,target=data.cuda(args.cudaId),target.cuda(args.cudaId)#部署到device
            output=model(data)#训练后结果
            pred = output.max(1, keepdim=True)[1]  # 找到概率值最大的下标,这里调用了python的函数，也可以自己写
            correct+=pred.eq(target.view_as(pred)).sum().item()
        print("正确率：{:.2f}".format(100*correct/len(test_loader.dataset)),"推理时间："+str(time.time()-t0))            #输出
        return 100*correct/len(test_loader.dataset)

def train_supernet(args):
    if not os.path.exists(args.savePath):
        os.mkdir(args.savePath)
    LAYER_NUM=args.LAYER_NUM
    pipeline =transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))])
    datasets.ImageNet
    train_set=datasets.CIFAR10("picsData/cifar10",train=True,download=False,transform=pipeline)
    test_set=datasets.CIFAR10("picsData/cifar10",train=False,download=False,transform=pipeline)
    train_loader = DataLoader(train_set,batch_size=args.BATCH_SZ,shuffle=True)#加载数据集，shuffle=True是打乱数据顺序
    test_loader = DataLoader(test_set,batch_size=args.BATCH_SZ,shuffle=False)
    train_loader=getsubdataloader(train_loader,args.traindata_scale)
    savePath=args.savePath
    model_name = args.model_name + ".pt"
    if not torch.cuda.is_available():
        print("cuda erro")
        return
    if args.model_name in netDir:
        Net=netDir[args.model_name]
    else:
        print("初始化模型失败:"+model_name)
        return
    print(args.cudaId)
    if args.model_name[0:10]=="mysupernet":
        arch_code=getSuperCode(LAYER_NUM)
        model=Net(arch_code,out_fea=10,layer_num=LAYER_NUM,args=args).cuda(args.cudaId)
    else:
        model=Net(out_fea=10, args=args).cuda(args.cudaId)
    if args.isResume:
        if os.path.exists(savePath+model_name):
            model = torch.load(savePath+model_name)#加载神经网络
        else:
            print("不存在已有模型")
            return
    teachNet=None
    if "distillation" in args and args.distillation in netDir:
        teachNet=netDir[args.distillation](out_fea=10, args=args).cuda(args.cudaId)
        # teachPath = teachNet.path
        # if os.path.exists(teachPath):
        #     par=torch.load(teachPath)
        #     if isinstance(par, collections.OrderedDict):
        #         teachNet.load_state_dict(par)
        #     else:
        #         teachNet = par # 加载神经网络
        teachNet.eval()
        test_model(teachNet, test_loader)
        print("蒸馏：teacher:{} to student:{}".format(args.distillation,model_name))
    print(args.model_name+" 参数量:{:2f} million".format(getParameterCnt(model=model)))
    writer = SummaryWriter("logs_fina")  # 存放log文件的目录
    optimizer = optim.Adam(model.parameters(), lr = args.lr)
    maxTestAcc = test_model(model, test_loader)
    for epoch in range(0,args.epoch_cnt):
        if teachNet==None:
            Accuracy,avgLoss=train_model(model,train_loader,optimizer,epoch)
        else:
            Accuracy, avgLoss = distill_model(model,teachNet, train_loader, optimizer, epoch)
        if epoch%30==0 and epoch>0:
            acc=test_model(model, test_loader)
            writer.add_scalar('testAcc', acc, epoch)
            if maxTestAcc<acc:
                maxTestAcc = acc
                torch.save(model, savePath+model_name)  # 保存模型pt || pth
                print("保存到"+savePath+model_name)

        writer.add_scalar('trainAcc', Accuracy, epoch)
        writer.add_scalar('avgLoss', avgLoss, epoch)
        # if avgLoss<0.00000001:
        #     break
    torch.save(model, savePath+model_name)#保存模型pt || pth
    start_time = time.time()
    for epoch in range(0, 10):
        test_model(model,test_loader)
    avgtime=(time.time()-start_time)/10
    print("平均推理延时："+str(avgtime))

parser = argparse.ArgumentParser()
parser.add_argument('--savePath', type=str, default="nasData/train_dist/")#layer_dist_maxstd
parser.add_argument('--cudaId', type=int, default=0)
parser.add_argument('--BATCH_SZ', type=int, default=128)
parser.add_argument('--traindata_scale', type=float, default=1)#学习率
parser.add_argument('--model_name', type=str, default="mysupernet10")#mysupernet,vgg,mysupernet20
parser.add_argument('--isResume', type=bool, default=True)#1模型初始参数
parser.add_argument('--epoch_cnt', type=int, default=511)#1模型初始参数args
parser.add_argument('--is_fineTuning', type=bool, default=False)#1模型初始参数args
parser.add_argument('--distillation', type=str, default="densenet121")#蒸馏的网络,densenet121
parser.add_argument('--LAYER_NUM', type=int, default=10)#蒸馏的网络
parser.add_argument('--lr', type=float, default=0.0001)#学习率
parser.add_argument('--kernel_norm_sc', type=float, default=0.05)#核范数缩放比例
args = parser.parse_args()
train_supernet(args)
# net=torch.load("nasData/train/imagenet21k+imagenet2012_ViT-L_32.pth")
# print(net)
# model=netDir['vit'](10,args)
# x=torch.ones([4,3,224,224])
# x=model.forward(x)
# print(x.shape)
# conv=nn.Conv2d(3, 6, kernel_size=5,stride=1)
# print(conv.weight.shape)
# Pretrained model
#下一个
# parser = argparse.ArgumentParser()
# parser.add_argument('--savePath', type=str, default="nasData/layer10_other/")#
# parser.add_argument('--cudaId', type=int, default=0)
# parser.add_argument('--BATCH_SZ', type=int, default=128)
# parser.add_argument('--model_name', type=str, default="resnet")#mysupernet,vgg
# parser.add_argument('--isResume', type=bool, default=True)#1模型初始参数
# parser.add_argument('--epoch_cnt', type=int, default=0)#1模型初始参数args
# parser.add_argument('--is_fineTuning', type=bool, default=True)#1模型初始参数args
# parser.add_argument('--lr', type=float, default=0.01)#学习率
# args = parser.parse_args()
# train_supernet(args)