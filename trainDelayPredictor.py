import copy
import os
import platform
import random
import shutil
import time
import argparse

import MNN
from torch.utils.tensorboard import SummaryWriter
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import matplotlib.pyplot as plt
import torchvision
from arch_code import arch_code
from DGNAS.nas_util import toMat, archCodeEncoder, archCodeAdList, cutcodeFromFSP, compareArchCode, getKernelList, \
    cutcodeFromDeep, estimate_cell_weight, cutcell_argmax, getsubdataloader, getParameterCnt, getFSP, \
    torchToMnn, send_model, test_model_norm, test_model, test_model_time, estimateNetTime, torchToOnnx, onnxExampletime, \
    example, getAllMnnKernelTime
from DGNAS.nasnetwork import SuperNet, RepairNet, netDir

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
from torch.utils.data import DataLoader, random_split  # 数据集加载
BATCH_SZ=128#由于数据量很大，要分批读入内存，代表一批有多少数据
cudaId=0
EPOCHS_CNT=10 #训练的轮次
LAYER_NUM=10
zheng_alpha=0.5
# savePath="nasData/layer10_other/"
savePath="nasData/train_dist/"
print(torch.cuda.is_available())
def getMnnKernelTime(model_path):
    """ inference mobilenet_v1 using a specific picture """
    if not os.path.exists(model_path):
        print("模型不存在")
        return
    interpreter = MNN.Interpreter(model_path)
    session = interpreter.createSession()
    input_tensor = interpreter.getSessionInput(session)

    def callback(op, before, after):
        op_name = op.type  # 获取算子类型
        elapsed_time = (after - before) * 1000.0  # 将时间转换为毫秒
        print(f"Operator {op_name}: {elapsed_time:.2f} ms")

    # interpreter.runSessionWithCallBack(callback)

    image = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
    #cv2 read as bgr format
    image = image[..., ::-1]
    #change to rgb format
    image = cv2.resize(image, (32, 32))
    #resize to mobile_net tensor size
    image = image - (103.94, 116.78, 123.68)
    image = image * (0.017, 0.017, 0.017)
    #preprocess it
    image = image.transpose((2, 0, 1))
    #change numpy data type as np.float32 to match tensor's format
    image = image.astype(np.float32)
    # image=np.tile(image, (10, 1, 1, 1))
    # print(image.shape)
    #cv2 read shape is NHWC, Tensor's need is NCHW,transpose it
    tmp_input = MNN.Tensor((1, 3, 32, 32), MNN.Halide_Type_Float,\
                    image, MNN.Tensor_DimensionType_Caffe)
    # print("ssaddsasda")
    input_tensor.copyFrom(tmp_input)
    # print("ssaddsasda")
    t0=time.time()
    interpreter.runSession(session)
    # print("ssaddsasda")
    ans=interpreter.getSessionOutputAll(session)
    realTime = time.time() - t0
    # print(interpreter.getCallBackInfo())
    for k in ans.keys():
        tensor_shape = ans[k].getShape()
        ans[k]=np.array(ans[k].getData(), copy=False).reshape(tensor_shape)
    return ans['output'],ans['t0'],ans['t1'],realTime
def predictorTime(model_path_0,model_path_1):
    acc0,time0,time0_add,realTime0=getMnnKernelTime(model_path_0)
    acc1,time1,time1_add,realTime1=getMnnKernelTime(model_path_1)
    s0=time0.sum()+time0_add.sum()
    s1=time1.sum() + time1_add.sum()
    print(time0.sum(axis=1))
    print(time1.sum(axis=1))
    # print(time0.sum(),time0_add.sum(),s0,realTime0)
    # print(time1.sum(), time1_add.sum(), s1, realTime1)
    print(s1/s0,realTime1/realTime0)
pipeline = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))])
# print(torch.cuda.is_available())
train_set=datasets.CIFAR10("picsData/cifar10",train=True,download=False,transform=pipeline)
test_set=datasets.CIFAR10("picsData/cifar10",train=False,download=False,transform=pipeline)
train_loader = DataLoader(train_set,batch_size=BATCH_SZ,shuffle=True)#加载数据集，shuffle=True是打乱数据顺序
test_loader = DataLoader(test_set,batch_size=BATCH_SZ,shuffle=False)
model_name="mysupernet10"
model = torch.load(savePath+model_name+".pt")  # 加载神经网络
optimizer = optim.Adam(model.parameters(), lr = 0.001)

def tomnn(li):
    for a in li:
        model_1 = torch.load(savePath + a+".pt")  # 加载神经网络
        model_1.forward=model_1.forward_time
        torchToMnn(model_1,savePath+a+".mnn",True)
def testmnntime1():
    nowname = "submodel/model_supernet_60"
    li=[savePath+nowname+".mnn",savePath+"mysupernet10.mnn"]
    for i in range(14*LAYER_NUM):
        nowname = "submodel/model_test_{}".format(i)
        li.append(savePath + nowname + ".mnn")
    ans=getAllMnnKernelTime(li,1000)
    print(ans)
    np.save(savePath+"time.npy",ans)
def testtime1():
    li=[]
    for i in range(LAYER_NUM):
        for j in range(14):
            li.append((i, j))
    timeList=np.load(savePath+"time.npy")
    superacc=timeList[0]
    for i in range(1,len(timeList)):
        superacc-=timeList[0]-timeList[i]

    print(timeList,superacc)
def testtime(li):
    re=np.zeros((4))
    cnt=1000
    for _ in range(cnt):
        for i,a in enumerate(li):
            acc0,time0,time0_add,real=getMnnKernelTime(savePath+a+".mnn")
            re[i]+=real
    re/=cnt
    s0,s1=re[0]-re[1],re[0]-re[2]
    print(re,s0,s1,re[0]-s0-s1)
testmnntime1()
testtime1()
# model0=torch.load(savePath+"model_supernet_60.pt")
# nowname="submodel/model_supernet_60"
# torchToMnn(model0,savePath+nowname+".mnn")
# tomnn(["mysupernet10","model_test0_0","model_test0_1","model_test1_0"])
# testtime(["mysupernet10","model_test0_0","model_test0_1","model_test1_0"])
# onnxExampletime(savePath+"model_supernet_40.onnx")
# model_1=torch.load(savePath+"model_supernet_40.pt")  # 加载神经网络
# model.forward=model.forward_time
# torchToMnn(model,savePath+"mysupernet10.mnn",True)
# model_1.forward=model_1.forward_time
# torchToMnn(model_1,savePath+"model_supernet_40.mnn",True)
# predictorTime(savePath+"mysupernet10.mnn",savePath+"model_supernet_40.mnn")
# example(savePath+"model_supernet10.mnn")
# example(savePath+"model_supernet_40.mnn")