import copy
import math
import os
import platform
# from attention import BasicAttention
import time
from collections import deque
from functools import cmp_to_key

# from torch_geometric.nn import GATConv
# from torchvision.models import VisionTransformer, vit_l_32

from basic_operations import *
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import torchvision

from DGNAS.nas_util import Kahn, toMat
# from vit_pytorch.vit import ViT
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch #直接包含整个包
import torch.nn as nn #
import torch.optim as optim
import torch.nn.functional as F#激活函数

from torchvision import transforms
from torch.utils.data import DataLoader#数据集加载
operation_canditates = {
    0: lambda C, stride: Zero(stride),
    1: lambda C, stride: Identity() if stride == 1 else FactorizedReduce(C, C),
    2: lambda C, stride: SepConv(C, C, 3, stride, 1),
    3: lambda C, stride: ResSepConv(C, C, 3, stride, 1),
}
def archToId(op):
    if isinstance(op, Zero):
        return 0
    elif isinstance(op, Identity):
        return 1
    elif isinstance(op, SepConv):
        return 2
    elif isinstance(op, ResSepConv):
        return 3
class Cell(nn.Module):#一个cell里包含14条边，也就是14个模块

    def __init__(self, steps,multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev, genotype):
        super(Cell, self).__init__()
        self.reduction = reduction
        self.steps = steps
        self.multiplier=multiplier
        # For search stage, the affine of BN should be set to False, in order to avoid conflict with architecture params
        self.affine = False
        self.waste=deque()
        if reduction_prev:#把size减少一半
            self.preprocess0 = FactorizedReduce(C_prev_prev, C)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)

        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)

        self._ops = nn.ModuleList()
        self._complie(C, reduction, genotype)

    def _complie(self, C, reduction, genotype):
        offset = 0
        for i in range(self.steps):#steps默认是4，代表分支特征向量个数
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                geno=genotype[offset + j]
                op = operation_canditates[geno](C, stride)
                self._ops.append(op)
            offset += 2 + i
    def setArch(self,archCode,C,reduction):
        offset = 0
        chCnt=0
        if not hasattr(self, "waste"):
            self.waste = deque()
        for i in range(self.steps):  # steps默认是4，代表分支特征向量个数
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                if archCode[offset + j]<=1:
                    if archToId(self._ops[offset + j])!=archCode[offset + j]:
                        self.waste.append((offset + j,self._ops[offset+j]))
                        self._ops[offset + j] = operation_canditates[archCode[offset + j]](C, stride)
                        chCnt+=1
            offset += 2 + i
        return chCnt

    def setBack(self):
        for a in self.waste:
            i,op=a[0],a[1]
            self._ops[i]=op
    def clearWaste(self):
        self.waste = deque()
    def calcFSP(self,x, y):#计算FSP矩阵
        shx = x.shape
        shy = y.shape
        if shx == shy:
            return (x * y).mean(axis=[1, 2,3])
        shmin = [slice(0, min(shx[i], shy[i])) for i in range(len(shx))]  # 动态切片
        return (x[shmin] * y[shmin]).mean(axis=[1, 2,3])
    def forward(self, s0, s1):#
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        states = [s0, s1]#两个输入
        offset = 0
        for i in range(self.steps):#分别计算第：0,1;   2,3,4;   5,6,7,8;  9,10,11,12,13;
            li=[]
            for j, h in enumerate(states):
                if archToId(self._ops[offset + j])!=0:
                    x=self._ops[offset + j](h)
                    # fan=self.calcFSP(x,h)
                    # fan = torch.norm(fan, 1)
                    # if fan<10:
                    #     x*=0
                    # print(fan)
                    li.append(x)
            # s = sum(self._ops[offset + j](h) for j, h in enumerate(states))
            offset += len(states)
            if len(li)!=0:
                s=sum(li)
                states.append(s)
            else:
                states.append(self._ops[offset-1](states[-1]).detach())
        ans=torch.cat(states[-self.multiplier:], dim=1)
        # FSP_loss=self.calcFSP(s0,ans)
        # FSP_loss=torch.zeros((states[0].shape[0],2,4))计算FSP_loss
        # for i in range(2):#计算FSP矩阵
        #     for j in range(4):
        #         # print(self.calcFSP(states[i], states[j + 2]).shape)
        #         FSP_loss[:,i,j]=self.calcFSP(states[i],states[j+2])
        # print(FSP_loss.shape)
        return ans
    def forward_fsp(self, s0, s1):#
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        states = [s0, s1]#两个输入
        offset = 0
        fsp=[]
        for i in range(self.steps):#分别计算第：0,1;   2,3,4;   5,6,7,8;  9,10,11,12,13;
            li=[]
            for j, h in enumerate(states):
                x=self._ops[offset + j](h)
                li.append(x)
                fan=self.calcFSP(x,h)
                fan = torch.norm(fan, 1)
                # fan=torch.var(x)
                fsp.append(fan.item())
                # print(fan)

            # s = sum(self._ops[offset + j](h) for j, h in enumerate(states))
            s=sum(li)
            offset += len(states)
            states.append(s)
        ans=torch.cat(states[-self.multiplier:], dim=1)
        return ans,fsp
    def forward_time(self, s0, s1):#
        forward_t0 = time.time()
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        states = [s0, s1]#两个输入
        offset = 0
        timeList = []
        # timeList_add=[]
        for i in range(self.steps):#分别计算第：0,1;   2,3,4;   5,6,7,8;  9,10,11,12,13;
            li=[]
            for j, h in enumerate(states):
                t0=time.time()
                x=self._ops[offset + j](h)
                li.append(x)
                t1 = time.time()
                timeList.append(t1-t0)
                # print(fan)

            # s = sum(self._ops[offset + j](h) for j, h in enumerate(states))
            # t0 = time.time()
            s=sum(li)
            offset += len(states)
            states.append(s)
            # t1 = time.time()
            # timeList_add.append(t1-t0)
        ans=torch.cat(states[-self.multiplier:], dim=1)
        return ans,timeList,time.time()-forward_t0-sum(timeList)

    def forward_norm(self, s0, s1,p=2):  #
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        states = [s0, s1]  # 两个输入
        offset = 0
        feaList = []
        for i in range(self.steps):  # 分别计算第：0,1;   2,3,4;   5,6,7,8;  9,10,11,12,13;
            li = []
            for j, h in enumerate(states):
                x = self._ops[offset + j](h)
                li.append(x)
                l2_loss = 0
                for param in self._ops[offset + j].parameters():
                    if param.requires_grad and len(param.size()) > 1:  # 只对权重矩阵进行计算
                        l2_loss += torch.norm(param, p=p)
                # print(l2_loss)
                feaList.append(l2_loss)
            s = sum(li)
            offset += len(states)
            states.append(s)
        ans = torch.cat(states[-self.multiplier:], dim=1)
        return ans, feaList
    def getParameterCnt(self):#得到参数量
        total_params = sum(p.numel() for p in self.preprocess0.parameters())
        total_params += sum(p.numel() for p in self.preprocess1.parameters())
        total_params += sum(p.numel() for p in self._ops.parameters())
        return total_params
class TorchResNet(torch.nn.Module):
    def __init__(self, out_fea=10):
        super(TorchResNet, self).__init__()
        self.net=torchvision.models.resnet18(pretrained=False)
        # for param in self.net.parameters():
        #     param.requires_grad = False
        print(self.net)
        # self.net.conv1=nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # self.net.avgpool=nn.Sequential(
        #     nn.AdaptiveAvgPool2d(output_size=(1, 1),
        #
        # )
        self.net.fc = nn.Linear(512, out_fea)
        self.name = "resnet18.pt"
    def forward(self, x):#
        x = self.net(x)
        # x=x.view(x.size(0),-1)
        # print(x.shape)
        # x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x
class TorchDenNet(torch.nn.Module):
    def __init__(self, out_fea=10):
        super(TorchDenNet, self).__init__()
        self.net=torchvision.models.densenet121(pretrained=False)
        print(self.net)
        # self.net.features.conv0=nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.net.fc = nn.Linear(1024, out_fea)
        self.name = "pytorchdennet"
    def forward(self, x):#
        x = self.net(x)
        # x=x.view(x.size(0),-1)
        # print(x.shape)
        # x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x
class VGG(torch.nn.Module):
    def __init__(self, out_fea,args):
        super(VGG, self).__init__()
        self.net=torchvision.models.vgg16(pretrained=False)
        print(self.net)
        self.net.classifier[6] = nn.Linear(4096, out_fea,bias=True)
    def forward(self, x):#
        x = self.net(x)
        return x
class ResNet(torch.nn.Module):
    def __init__(self, out_fea,args):
        super(ResNet, self).__init__()
        self.net=torchvision.models.resnet50(pretrained=True)
        if args.is_fineTuning:#如果是微调
            for param in self.net.parameters():
                param.requires_grad = False
        print(self.net)
        self.net.fc = nn.Linear(2048, out_fea,bias=True)
    def forward(self, x):#
        x = self.net(x)
        return x
class DenseNet(torch.nn.Module):
    def __init__(self, out_fea,args):
        super(DenseNet, self).__init__()
        self.net=torchvision.models.densenet201(pretrained=True)
        if args.is_fineTuning:  # 如果是微调
            for param in self.net.parameters():
                param.requires_grad = False
        print(self.net)
        # self.net.fc = nn.Linear(1024, out_fea)
    def forward(self, x):#
        x = self.net(x)
        return x
class MyVit(torch.nn.Module):
    def __init__(self, out_fea,args):
        super(MyVit, self).__init__()
        # self.net=ViT(image_size = 32,
        #     patch_size = 16,
        #     num_classes = out_fea,
        #     dim = 1024,
        #     depth = 10,
        #     heads = 16,
        #     mlp_dim = 2048,
        #     dropout = 0.1,
        #     emb_dropout = 0.1
        # )
        # self.net=torch.load("nasData/train/imagenet21k+imagenet2012_ViT-L_32.pth")
        self.net= vit_l_32(pretrained=True)
        self.net.heads=nn.Linear(1024,10)
        print(self.net)
        # self.net.fc = nn.Linear(1024, out_fea)
    def forward(self, x):#
        x = torch.nn.functional.pad(x, (96, 96, 96, 96), mode='constant', value=0)
        # print(x.shape)
        x = self.net(x)
        return x
class SuperNet(torch.nn.Module):
    def __init__(self,arch_code,layer_num=5,out_fea=10,stem_multiplier=3,multiplier=4,feaNodeCnt=4,args=None):
        #其中arch_code是结构编码，layer_num是cell层数，out_fea是输出维度，stem_multiplier是stem的输入维度，multiplier是阶段个数
        # self.tran = nn.Transformer(d_model=lstmInfea, nhead=3, num_encoder_layers=2, num_decoder_layers=2,
        #                            dim_feedforward=lstmInfea, batch_first=True, dropout=0.1)
        super(SuperNet, self).__init__()
        C = 32
        self.C=C
        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )
        # self.otherConv=nn.Sequential(
        #     nn.Conv2d(1, 6, kernel_size=5,stride=2),
        #     nn.ReLU(),
        #     nn.Conv2d(6, 16, kernel_size=5,stride=2),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(),
        #     nn.Conv2d(16, 32, kernel_size=3, stride=1),
        #     nn.ReLU(),
        # )
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        reduction_prev = False
        self.cells = nn.ModuleList()
        for i in range(layer_num):
            if i in [layer_num // 3, 2 * layer_num // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(feaNodeCnt, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev,
                        genotype=arch_code[i])
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr
            # print(C_prev)
        # print(self.cells)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, out_fea, bias=True)#+288
        # self.contribute_mlp=nn.Sequential(
        #     nn.Linear(C_prev, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, layer_num * 14)
        # )

    def setArch(self, arch_code,layer_num=5,out_fea=10,C=16,stem_multiplier=3,multiplier=4,feaNodeCnt=4):
        # C = self.C
        C_curr = stem_multiplier * C
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        reduction_prev = False
        for i in range(layer_num):
            if i in [layer_num // 3, 2 * layer_num // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            self.cells[i].setArch(arch_code[i],C_curr,reduction)
            reduction_prev = reduction
            C_prev_prev, C_prev = C_prev, multiplier * C_curr
    def setBack(self,layer_num):
        for i in range(layer_num):
            self.cells[i].setBack()

    def clearWaste(self, layer_num):
        for i in range(layer_num):
            self.cells[i].clearWaste()

    def setClassifier(self,layers,num_classes,cudaID):
        x=torch.ones((1,3,224,224)).cuda(cudaID)
        s0 = s1 = self.stem(x)
        # batch_sz=input.shape[0]
        # FSP_loss=torch.zeros((batch_sz,self._layers,2,4))
        for i, cell in enumerate(self.cells):
            if i == layers:
                break
            s0, s1 = s1, cell(s0, s1)
        out = self.global_pooling(s1)  # 池化
        sh=out.view(out.size(0),-1).shape[1]
        # print(sh)
        self.classifier = nn.Linear(sh, num_classes,bias=True).cuda(cudaID)
    def forward_norm(self,x):
        s0 = s1 = self.stem(x)
        normMat=[]
        for i, cell in enumerate(self.cells):
            tmp,cell_norm=cell.forward_norm(s0, s1)
            normMat.append(cell_norm)
            s0, s1 = s1, tmp
        out = self.global_pooling(s1)  # 池化
        out=out.view(out.size(0),-1)
        x=out
        x = self.classifier(x)  # 分类
        return x,normMat
    def getParameterCnt(self,layer_num):#得到参数量
        total_params = sum(p.numel() for p in self.stem.parameters())+sum(p.numel() for p in self.classifier.parameters())
        for i in range(layer_num):
            total_params+= self.cells[i].getParameterCnt()
        return total_params
    def calcFSP(self,x, y):#计算FSP矩阵
        shx = x.shape
        shy = y.shape
        if shx == shy:
            return (x * y).mean(axis=[1, 2,3])
        shmin = [slice(0, min(shx[i], shy[i])) for i in range(len(shx))]  # 动态切片
        return (x[shmin] * y[shmin]).mean(axis=[1, 2,3])
    def calcFSP_1(self,x, y):#计算FSP矩阵
        shx = x.shape
        shy = y.shape
        if shx == shy:
            return (x - y)
        shmin = [slice(0, min(shx[i], shy[i])) for i in range(len(shx))]  # 动态切片
        return (x[shmin] - y[shmin])
    def forward(self, x,layer_num=-1):
        s0 = s1 = self.stem(x)
        self.cell_output = [s0]
        # otherx = self.otherConv(x)
        # otherx = otherx.view(otherx.size(0), -1)
        for i, cell in enumerate(self.cells):
            s_cell=cell(s0, s1)
            self.cell_output.append(s_cell)
            s0, s1 = s1, s_cell
            if layer_num>=0 and layer_num==i:
                return s1
        out = self.global_pooling(s1)  # 池化
        out = out.view(out.size(0), -1)
        x=out
        # x = torch.cat((out, otherx), dim=1)
        x = self.classifier(x)  # 分类
        # x = F.softmax(x, dim=1)

        return x
    def freezeConv(self,isFreeze=False):
        for param in self.cells.parameters():
            param.requires_grad = isFreeze
    def forward_fsp(self, x):
        s0 = s1 = self.stem(x)
        # otherx = self.otherConv(x)
        # otherx = otherx.view(otherx.size(0), -1)
        fsp=[]
        # print(otherx.shape)
        for i, cell in enumerate(self.cells):
            s_cell,fspi=cell.forward_fsp(s0, s1)
            fsp.append(fspi)
            s0, s1 = s1, s_cell
        out = self.global_pooling(s1)  # 池化
        out = out.view(out.size(0), -1)
        x=out
        # x = torch.cat((out, otherx), dim=1)
        x = self.classifier(x)  # 分类
        return x,fsp
    def forward_time(self, x):
        t0 = time.time()
        s0 = s1 = self.stem(x)
        timeLi=[]
        timeLi_add=[time.time()-t0]
        for i, cell in enumerate(self.cells):
            s_cell,timeListi,timeListi_add=cell.forward_time(s0, s1)
            timeLi.append(timeListi)
            timeLi_add.append(timeListi_add)
            s0, s1 = s1, s_cell
        t0=time.time()
        out = self.global_pooling(s1)  # 池化
        out = out.view(out.size(0), -1)
        x=out
        # x = torch.cat((out, otherx), dim=1)
        x = self.classifier(x)  # 分类
        timeLi_add.append(time.time()-t0)
        return x,torch.Tensor(timeLi),torch.Tensor(timeLi_add)
    def examfun(self, x,layer_num=-1):
        s0 = s1 = self.stem(x)
        self.cell_output = [s0]
        # otherx = self.otherConv(x)
        # otherx = otherx.view(otherx.size(0), -1)
        for i, cell in enumerate(self.cells):
            s_cell = cell(s0, s1)
            self.cell_output.append(s_cell)
            s0, s1 = s1, s_cell
            if layer_num >= 0 and layer_num == i:
                return s1
        out = self.global_pooling(s1)  # 池化
        out = out.view(out.size(0), -1)
        x = out
        # x = torch.cat((out, otherx), dim=1)
        x = self.classifier(x)  # 分类
        return x, torch.ones(2,7), torch.zeros(2,7)+0.5
class RepairNet(torch.nn.Module):#修补网络
    def __init__(self,node_nums,out_fea=10):
        # 其中arch_code是结构编码，layer_num是cell层数，out_fea是输出维度，stem_multiplier是stem的输入维度，multiplier是阶段个数
        super(RepairNet, self).__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # 原始的模型使用的是 平均池化
        )
        self.gonv = nn.Sequential(
            GATConv(1, 6),
            nn.ReLU(),
            GATConv(6, 16),
            nn.ReLU(),
        )
        self.fc=nn.Sequential(
            nn.Linear(960+400,512),
            nn.ReLU(),
            nn.Linear(512, out_fea),
            nn.Tanh(),
        )
    def batchnormGnn(self,x):
        # print(x.shape)
        sz=x.shape[0]*x.shape[1]
        avg=x.sum().item()/sz
        std=(x-avg).mul(x-avg)
        std=std.sum().item()/sz
        std=math.sqrt(std)
        return (x-avg).div(std)
    def forward(self,node_fea,edge_index,edge_fea,pic_fea):
        batch_size=pic_fea.shape[0]
        pic_fea= self.conv(pic_fea)
        pic_fea = pic_fea.view(pic_fea.size(0), -1)
        x = self.gonv[0](node_fea, edge_index, edge_fea)
        x = self.batchnormGnn(x)
        x = self.gonv[1](x)
        x = self.gonv[2](x, edge_index, edge_fea)
        x = self.batchnormGnn(x)
        x = self.gonv[3](x)
        x = x.view(-1)
        # print(x.shape,pic_fea.shape)
        x= x.repeat(batch_size,1)
        # print(x.shape, pic_fea.shape)
        x=torch.cat((x,pic_fea),dim=1)
        x = self.fc(x)  # 分类
        return x

netDir={"mysupernet5":SuperNet,"mysupernet":SuperNet,"vgg":VGG,"resnet":ResNet,"densenet":DenseNet,"vit":MyVit}
for i in range(50):
    netDir["mysupernet"+str(i)]=SuperNet

from cifar10_models.densenet import Densenet121, Densenet161, Densenet169
netDir["densenet121"]=Densenet121
netDir["densenet161"]=Densenet161
netDir["densenet169"]=Densenet169

# mat=toMat(li,6)
# print(mat)
# net=Cell(128,mat)
# x=torch.ones((7,128,56,56))
# net.forward_time(x)
# x=torch.Tensor(([[1,1],[1,2]]))
# x = F.softmax(x, dim=1)
# print(x)