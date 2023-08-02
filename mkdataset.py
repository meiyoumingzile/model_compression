import copy
import os
import platform
import random
import shutil
import time
import argparse
from torch.utils.tensorboard import SummaryWriter
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import matplotlib.pyplot as plt
import torchvision

from DGNAS.arch_code import getSuperCode
from DGNAS.nas_util import toMat, archCodeEncoder, archCodeAdList, cutcodeFromFSP, compareArchCode, getKernelList, \
    cutcodeFromDeep, estimate_cell_weight, cutcell_argmax, getsubdataloader, getParameterCnt, getFSP, \
    torchToMnn, send_model, test_model_norm, test_model, test_model_time, estimateNetTime
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
#1.以上是加载库

BATCH_SZ=128#由于数据量很大，要分批读入内存，代表一批有多少数据
cudaId=0
EPOCHS_CNT=10 #训练的轮次
LAYER_NUM=10
zheng_alpha=0.5
# savePath="nasData/layer10_other/"
savePath="nasData/train_dist/"
print(torch.cuda.is_available())
# writer=SummaryWriter(log_dir="../logs", flush_secs=60)#指定存储目录和刷新间隔

def adjustOpt(optimizer,epoch):
    p = 0.9
    if epoch%4==0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= p

def train_model(model,train_loader,optimizer,epoch,fp=1):#model是模型，device是设备，train_loader是数据集，optimizer是梯度，epoch当前第几轮
    model.train()
    correct=0#正确率
    avgLoss=0
    # adjustOpt(optimizer, epoch)
    for batch_index, (d,t) in enumerate(train_loader):
        data,target=d.cuda(cudaId),t.cuda(cudaId)#部署到device
        optimizer.zero_grad()#初始化梯度为0
        output=model.forward(data)#训练后结果
        loss=F.cross_entropy(output,target)#计算损失,用交叉熵,默认是累计的
        loss.backward()#反向传播
        avgLoss +=loss.item()
        optimizer.step()#参数优化
        pred = output.max(1, keepdim=True)[1]  # 找到概率值最大的下标,这里调用了python的函数，也可以自己写
        correct += pred.eq(target.view_as(pred)).sum().item()
    avgLoss /=len(train_loader.dataset)  # 计算平均数
    Accuracy=100 * correct / len(train_loader.dataset)
    print("第{}，正确率{:.6f}  loss：{:.6f}".format(epoch,Accuracy,avgLoss))
    return Accuracy,avgLoss
    # writer.add_scalar('Accuracy', Accuracy, epoch)
    # writer.add_scalar('avgLoss', avgLoss, epoch)
def _train_repair_model(archCode,superNet,nowNet,repairNet,optimizer,train_loader,epoch):#model是模型，device是设备，train_loader是数据集，optimizer是梯度，epoch当前第几轮
    correct = 0  # 正确率
    avgLoss = 0
    node_fea, edge_index, edge_fea=archCodeEncoder(archCode,LAYER_NUM)
    repairNet.train()
    t0=time.time()
    for batch_index, (d, t) in enumerate(train_loader):
        data, target = d.cuda(cudaId), t.cuda(cudaId)  # 部署到device
        optimizer.zero_grad()  # 初始化梯度为0
        output_su = superNet.forward(data)  # 训练后结果
        output_now = nowNet.forward(data)  # 训练后结果
        output_re = repairNet.forward(node_fea.cuda(cudaId), edge_index.cuda(cudaId), edge_fea.cuda(cudaId), data)
        output_re = output_re + output_now.detach()
        loss = F.mse_loss(output_re, output_su.detach())  # 计算损失,用交叉熵,默认是累计的
        loss.backward()  # 反向传播
        avgLoss += loss.item()*1000
        optimizer.step()  # 参数优化
        pred = output_re.max(1, keepdim=True)[1]  # 找到概率值最大的下标,这里调用了python的函数，也可以自己写
        correct += pred.eq(target.view_as(pred)).sum().item()
    avgLoss /= len(train_loader.dataset)  # 计算平均数
    Accuracy = 100 * correct / len(train_loader.dataset)
    print("训练：第{}，正确率{:.6f}, loss：{:.6f}, 推理延时：{:.6f}".format(epoch, Accuracy, avgLoss,time.time()-t0))

def _test_repair_model(archCode, superNet, nowNet, repairNet, test_loader, epoch):  # model是模型，device是设备，train_loader是数据集，optimizer是梯度，epoch当前第几轮
    correct = 0  # 正确率
    avgLoss = 0
    node_fea, edge_index, edge_fea = archCodeEncoder(archCode, LAYER_NUM)
    repairNet.eval()
    t0 = time.time()
    for batch_index, (d, t) in enumerate(test_loader):
        data, target = d.cuda(cudaId), t.cuda(cudaId)  # 部署到device
        output_su = superNet.forward(data)  # 训练后结果
        output_now = nowNet.forward(data)  # 训练后结果
        output_re = repairNet.forward(node_fea.cuda(cudaId), edge_index.cuda(cudaId), edge_fea.cuda(cudaId), data)
        output_re = output_re + output_now.detach()
        loss = F.mse_loss(output_re, output_su.detach())  # 计算损失,用交叉熵,默认是累计的
        avgLoss += loss.item()*1000
        pred = output_re.max(1, keepdim=True)[1]  # 找到概率值最大的下标,这里调用了python的函数，也可以自己写
        correct += pred.eq(target.view_as(pred)).sum().item()
    Accuracy = 100 * correct / len(test_loader.dataset)
    print("推理： 第{}，正确率{:.6f}, loss：{:.6f}, 推理延时：{:.6f}".format(epoch, Accuracy, avgLoss, time.time() - t0))
    return Accuracy
    # writer.add_scalar('Accuracy', Accuracy, epoch)
    # writer.add_scalar('avgLoss', avgLoss, epoch)

main_li_0=[0,1,2,3,5,6,9,10]
main_li_1=[4,7,8,11,12,13]
def getEdgeList(codeArch):
    li=[]
    for i in range(LAYER_NUM):
        for j in main_li_1:
            if codeArch[i][j]!=0:
                li.append((i,j))
        cnt=0
        for j in main_li_0:
            cnt+=codeArch[i][j]!=0
        if cnt>1:
            for j in main_li_0:
                if codeArch[i][j] != 0:
                    li.append((i, j))
    print("before cuting, count of edge: " + str(len(li)))
    cutMain=False
    if len(li)==0:#剪主干时，变成直连
        # print("before cuting, count of main edge: " + str(len(li)))
        cutMain=True
        for i in range(LAYER_NUM):
            for j in main_li_0:
                if codeArch[i][j] >1:
                    li.append((i, j))
    return li,cutMain
def selectCut(codeArch,edge,cut0):#随机剪裁，保证网络一定是联通的
    x, y = edge[0], edge[1]
    if cut0:
        codeArch[x][y] = 1
    else:
        codeArch[x][y] = 0

def randCut(codeArch):#随机剪裁，保证网络一定是联通的
    if codeArch==None:
        return None
    codeArch=copy.deepcopy(codeArch)
    li,cut0=getEdgeList(codeArch)
    if len(li)==0:
        return None
    edge=li[random.randint(0,len(li)-1)]
    selectCut(codeArch,edge,cut0)
    return codeArch
def randCutArch(arch_code_now,model,T=5):
    for _ in range(T):
        arch_code_now = randCut(arch_code_now)
    model.setArch(arch_code_now, layer_num=LAYER_NUM)
    model.cuda()
    return arch_code_now
def init_distillation_model(model):
    optimizer= [optim.Adam(model.stem.parameters(), lr=0.01)]+[optim.Adam(cell.parameters(), lr=0.01) for cell in model.cells]
    return optimizer+[optim.Adam(model.classifier.parameters(), lr=0.001)]
def __distillation_cell(model,model2,data,optimizer,layer_num):
    optimizer[layer_num].zero_grad()  # 初始化梯度为0
    output = model.forward(data,layer_num)  # 训练后结果
    output2 = model2.forward(data,layer_num).detach()  # 训练后结果
    output = F.softmax(output, dim=1)
    output2 = F.softmax(output2, dim=1)
    loss = distillation_loss(output, output2)
    loss.backward()  # 反向传播
    optimizer[layer_num].step()  # 参数优化
def distillation_cell(model,model2,data,optimizer):
    # print(optimizer)
    s0 = s1 = model.stem(data)
    s0_2 = s1_2 = model2.stem(data).detach()
    # optimizer[0].zero_grad()  # 初始化梯度为0
    # output = F.log_softmax(s0, dim=1)
    # output2 = F.softmax(s0_2, dim=1)
    # loss = distillation_loss(output, output2)
    # loss.backward()  # 反向传播
    # optimizer[0].step()  # 参数优化

    s0, s1 = s0.detach(), s1.detach()
    s0_2, s1_2 = s0_2.detach(), s1_2.detach()
    for i, cell in enumerate(model.cells):
        lossitem=0
        for _ in range(10):
            s_cell = model.cells[i].forward(s0, s1)
            s_cell_2 = model2.cells[i].forward(s0_2 ,s1_2).detach()
            optimizer[i+1].zero_grad()  # 初始化梯度为0
            output,output2=s_cell,s_cell_2
            output = F.log_softmax(output, dim=1)
            output2 = F.softmax(output2 , dim=1)
            loss = distillation_loss(output, output2)
            lossitem+=loss.item()
            # print(loss.item())
            loss.backward()  # 反向传播
            optimizer[i+1].step()  # 参数优化
        # print("loss{}: {:6f} ".format(i,lossitem))
        s0, s1 = s1.detach(), s_cell.detach()
        s0_2, s1_2 = s1_2.detach(), s_cell_2.detach()
    for _ in range(10):
        output,output2 = model.global_pooling(s1),model2.global_pooling(s1_2)
        output,output2 = output.view(output.size(0), -1),output2.view(output2.size(0), -1)
        output,output2 = model.classifier(output), model2.classifier(output2).detach()
        optimizer[-1].zero_grad()  # 初始化梯度为0
        output = F.log_softmax(output, dim=1)
        output2 = F.softmax(output2, dim=1)
        loss = distillation_loss(output, output2)
        loss.backward()  # 反向传播
        optimizer[-1].step()  # 参数优化

def distillation_loss(x,y):
    return nn.KLDivLoss(reduction="batchmean")(x,y)
def distillation_model(model,model2,train_loader,optimizer,optimizer_cell,epoch):#model是模型，device是设备，train_loader是数据集，optimizer是梯度，epoch当前第几轮
    model.train()
    correct=0#正确率
    avgLoss=0
    # adjustOpt(optimizer, epoch)

    for batch_index, (d,t) in enumerate(train_loader):
        data,target=d.cuda(cudaId),t.cuda(cudaId)#部署到device
        # distillation_cell(model,model2,data,optimizer_cell)
        optimizer.zero_grad()#初始化梯度为0
        output=model.forward(data)#训练后结果
        output2=model2.forward(data)#训练后结果
        output=F.log_softmax(output, dim=1)
        output2 = F.softmax(output2, dim=1)
        loss=distillation_loss(output, output2.detach())
        # loss=(1-zheng_alpha)*F.cross_entropy(output, target)+zheng_alpha*F.cross_entropy(output, output2.detach())
        # print(target)
        # loss =F.cross_entropy(output, output2)
        loss.backward()#反向传播
        avgLoss +=loss.item()
        optimizer.step()#参数优化
        pred = output.max(1, keepdim=True)[1]  # 找到概率值最大的下标,这里调用了python的函数，也可以自己写
        correct += pred.eq(target.view_as(pred)).sum().item()
    avgLoss /=len(train_loader.dataset)  # 计算平均数
    Accuracy=100 * correct / len(train_loader.dataset)
    print("第{}，正确率{:.6f}  loss：{:.6f}".format(epoch, Accuracy, avgLoss))
    return Accuracy, avgLoss
def purning_submodel(arch_code,train_loader,test_loader,model_name):
    li=[]
    modelpt_name = model_name + ".pt"
    for i in range(LAYER_NUM):
        for j in range(14):
            li.append([(i,j)])
    for i in range(len(li)):
        model0 = torch.load(savePath + modelpt_name).cuda()  # 加载神经网络
        model0.eval()
        arch_code0=copy.deepcopy(arch_code)
        for j in range(len(li[i])):
            arch_code0[li[i][j][0]][li[i][j][1]]=0
        model0.setArch(arch_code0, layer_num=len(model.cells), out_fea=10, C=32)
        model0.cuda()
        nowname="submodel/model_test_{}".format(i)
        torch.save(model0, savePath + nowname+".pt")  # 保存模型pt || pth
        torchToMnn(model0,savePath+nowname+".mnn")
        os.remove(savePath+nowname+".onnx")
def purning_submodel_rand(arch_code,train_loader,test_loader,model_name):
    modelpt_name = model_name + ".pt"
    for i in range(140):
        model0 = torch.load(savePath + modelpt_name).cuda()  # 加载神经网络
        arch_code0 = copy.deepcopy(arch_code)
        for _ in range(3):
            arch_code0 = randCut(arch_code0)
        if arch_code0==None:
            print("剪枝完毕")
            break
        model0.setArch(arch_code0, layer_num=len(model.cells), out_fea=10, C=32)
        model0.cuda()
        nowname = "submodel/model_test_{}".format(i)
        torch.save(model0, savePath + nowname + ".pt")  # 保存模型pt || pth
        torchToMnn(model0, savePath + nowname + ".mnn")
        os.remove(savePath + nowname + ".onnx")
def purning_rand(arch_code,model,optimizer,train_loader,test_loader,model_name,isSave=False):
    begin = -1
    modelpt_name = model_name + ".pt"
    print(modelpt_name)
    Net = netDir["mysupernet10"]
    # model = Net(arch_code, out_fea=10, layer_num=LAYER_NUM, args=None)
    if begin < 0:
        model = torch.load(savePath + modelpt_name).cuda()  # 加载神经网络
        # print(savePath + modelpt_name)
    else:
        arch_code = np.load(savePath + 'model_arch_code_{}.npy'.format(begin))
        model = torch.load(savePath + model_name + "_{}.pt".format(begin)).cuda()  # 加载神经网络
    model_supernet = torch.load(savePath + modelpt_name).cuda()  # 加载神经网络
    model_supernet.eval()
    train_loader_sub = getsubdataloader(train_loader, 0.1)
    # print(arch_code)
    # test_model_norm(model, test_loader)
    print(" 参数量:{:2f} million".format(getParameterCnt(model=model)))

    # superAcc,timeLi,timeLi_0,timeLi_add,timeLi_add_0=getKernelList(model, test_loader,layer_num=LAYER_NUM)
    # print(superAcc,timeLi_add-timeLi_add_0)
    for i in range(0,1000):
        for _ in range(2):
            arch_code=randCut(arch_code)
        if arch_code==None:
            print("剪枝完毕")
            break
        model.setArch(arch_code,layer_num=len(model.cells),out_fea=10,C=32)
        model.cuda()
        acc=test_model(model, train_loader_sub)
        # estimateNetTime(arch_code, model, test_loader, timeLi0, timeLi0_add)
        # if superAcc-acc<1:
        #     continue
        estimateNetTime(arch_code,model,test_loader,timeLi,timeLi_add,layer_num=LAYER_NUM)
        # model2=model2
        # model2.load_state_dict(model.state_dict())
        for epoch in range(0, 0):
            Accuracy, avgLoss=distillation_model(model,model_supernet, train_loader, optimizer,optimizer, epoch)
            if epoch%10==0:
                acc = test_model(model, test_loader)
            if avgLoss<0.001:
                break
        if isSave or i%20==0:
            torch.save(model, savePath+"model_supernet_{}.pt".format(i))  # 保存模型pt || pth
            np.save(savePath+'model_arch_code_{}.npy'.format(i), np.array(arch_code))
        avg_acc=0
        # for epoch in range(0, 5):
        #     avg_acc+=test_model(model, test_loader)
        # print("轮次：{}，精度：{:.6f}".format(i,avg_acc/5))
def mkNasDataset_greedy(arch_code,model,optimizer,train_loader,test_loader,model_name):
    begin=-1
    modelpt_name=model_name+".pt"
    print(modelpt_name)
    Net=netDir["mysupernet20"]
    model = Net(arch_code, out_fea=10, layer_num=LAYER_NUM, args=None)
    if begin<0:
        model = torch.load(savePath + modelpt_name).cuda()  # 加载神经网络
    else:
        arch_code = np.load(savePath+'model_arch_code_{}.npy'.format(begin))
        model = torch.load(savePath + model_name+"_{}.pt".format(begin)).cuda()  # 加载神经网络
    model_supernet = torch.load(savePath + modelpt_name).cuda()  # 加载神经网络
    model_supernet.eval()
    train_loader_sub = getsubdataloader(train_loader, 0.1)
    print(arch_code)
    test_model_norm(model, test_loader)
    test_model(model, test_loader)
    # arch_code[3]=[0]*14
    # arch_code[2] = [0] * 14
    # print(len(model.cells))
    for i in range(max(begin+1,0),200):
        arch_code_now=copy.deepcopy(arch_code)
        Accuracy=100
        while(Accuracy>80):
            # fsp=getFSP(model,train_loader,LAYER_NUM)
            # print("fsp:",fsp)
            accsub = estimate_cell_weight(arch_code_now, model, train_loader_sub, LAYER_NUM)
            # arch_code_now,isCut = cutcodeFromFSP(model, arch_code_now, train_loader,LAYER_NUM)
            # arch_code_now,isCut = cutcodeFromDeep(model, arch_code_now, train_loader,LAYER_NUM)
            # arch_code_now=randCut(arch_code_now)
            x,y=cutcell_argmax(arch_code_now,accsub,LAYER_NUM)
            print("accsub:",accsub)
            print(x,y,accsub[x][y])
            if x==-1 or accsub[x][y]<-100:
                break
            if accsub[x][y]<0:
                arch_code_now=randCut(arch_code_now)
            else:
                arch_code_now[x][y] = 0
            model.setArch(arch_code_now, layer_num=len(model.cells))
            model.cuda()
            Accuracy=test_model(model, train_loader)
        # arch_code_now=randCutArch(arch_code_now,model,T=20)
        # while (Accuracy>70):
        #     accsub = estimate_cell_weight(arch_code_now, model, train_loader_sub, LAYER_NUM,val=1)
        #     print(1,accsub)
        #     x, y = cutcell_argmax(arch_code_now, accsub, LAYER_NUM)
        #     if x == -1 or accsub[x][y] < -10:
        #         break
        #     arch_code_now[x][y] = 1
        #     model.setArch(arch_code_now, layer_num=len(model.cells))
        #     model.cuda()
        #     Accuracy = test_model(model, train_loader)
        # arch_code_now = cutcodeFromFSP(model, arch_code_now, data, label)
        # print(arch_code,arch_code_now)
        isSame=compareArchCode(arch_code,arch_code_now,LAYER_NUM)
        # li,cutMain=getEdgeList(arch_code_now)
        print(arch_code_now)
        if isSame:
            break
        # randCut(arch_code_now)
        # randCut(arch_code_now)
        # # selectCut(arch_code_now, edge, cut0)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        opt_cell=init_distillation_model(model)
        # model.freezeConv()
        # model2=model2
        # model2.load_state_dict(model.state_dict())
        for epoch in range(0, 50):
            Accuracy,avgLoss=train_model(model, train_loader, optimizer, epoch)
            # Accuracy, avgLoss = distillation_model(model,model_supernet,train_loader,optimizer,opt_cell,epoch)
            if epoch % 20 == 0 and epoch>0:
                test_model(model, test_loader)
                torch.save(model, savePath + model_name+"_{}.pt".format(i))  # 保存模型pt || pth
                np.save(savePath + 'model_arch_code_{}.npy'.format(i), np.array(arch_code_now))
            if avgLoss < 0.0001 and epoch>5:
                break
        torch.save(model, savePath + model_name+"_{}.pt".format(i))  # 保存模型pt || pth
        np.save(savePath + 'model_arch_code_{}.npy'.format(i), np.array(arch_code_now))
        avg_acc = 0
        for epoch in range(0, 1):
            avg_acc += test_model(model, test_loader)
        avg_acc /= 1
        print("轮次：{}，精度：{:.6f}".format(i, avg_acc))
        arch_code=arch_code_now
        break
def mkNasDataset_flight(arch_code,model,train_loader,test_loader):#内伐网络
    for i in range(30,62):
        model = torch.load(savePath + "model_supernet_{}.pt".format(i)).cuda()
        optimizer = optim.Adam(list(model.classifier.parameters())+list(model.otherConv.parameters()), lr=0.01)
        optimizer_cells = optim.Adam(list(model.classifier.parameters())+list(model.stem.parameters()) + list(model.cells.parameters()), lr=0.01)
        optimizer_fc = optim.Adam(list(model.classifier.parameters()), lr=0.01)
        # Accuracy, avgLoss=train_model(model, train_loader, optimizer, 0)
        for epoch in range(0, 5):
            for _ in range(5):
                Accuracy, avgLoss = train_model(model, train_loader, optimizer_cells, epoch,-1)
            for _ in range(20):
                Accuracy, avgLoss = train_model(model, train_loader, optimizer, epoch)
        for epoch in range(0, 20):
            Accuracy, avgLoss = train_model(model, train_loader, optimizer_fc, epoch)
        torch.save(model, savePath+"model_flightnet_{}.pt".format(i))  # 保存模型pt || pth
        # np.save(savePath+'model_arch_code_{}.npy'.format(i), np.array(arch_code))
        avg_acc=0
        for epoch in range(0, 5):
            avg_acc+=test_model(model, test_loader)
        print("轮次：{}，精度：{:.6f}".format(i,avg_acc/5))
        break
def mkNasDataset_improve(train_loader,test_loader,begin=30):
    super_model = torch.load(savePath + "model_supernet.pt")  # 加载神经网络
    print(getParameterCnt())
    print("model_supernet.pt 参数量:"+str(super_model.getParameterCnt(LAYER_NUM)))
    for i in range(7,8):
        model_subi = torch.load(savePath + "model_supernet_{}.pt".format(i))
        archCodei = np.load(savePath + "model_arch_code_{}.npy".format(i)).tolist()
        print("model_supernet_{}.pt 参数量:{}".format(i,model_subi.getParameterCnt(LAYER_NUM)))
        # model_subi = SuperNet(archCodei, layer_num=1).cuda()
        optimizer = optim.Adam(model_subi.parameters(), lr=0.01)
        print(archCodei)
        for epoch in range(0, 1000):
            # Accuracy,avgLoss=train_model(model_subi, train_loader, optimizer, epoch)
            Accuracy,avgLoss=train_model_mkNasDataset(model_subi, super_model, train_loader, optimizer, epoch)
            if epoch%20==0:
                test_model(model_subi, test_loader)
            if avgLoss<0.00005:
                break
            # if epoch>0 and epoch%100==0:
                # torch.save(model_subi, savePath + "model_supernet_{}.pt".format(i))  # 保存模型pt || pth
                # np.save(savePath + 'model_arch_code_{}.npy'.format(i), np.array(archCodei))
        # torch.save(model_subi, savePath+"model_supernet_{}.pt".format(i))  # 保存模型pt || pth
        # np.save(savePath+'model_arch_code_{}.npy'.format(i), np.array(archCodei))
        avg_acc=0
        for epoch in range(0, 5):
            avg_acc+=test_model(model_subi, test_loader)
        print("轮次：{}，精度：{:.6f}".format(i,avg_acc/5))
def testNasDataset(arch_code,model,optimizer,train_loader,test_loader):
    arch_code=arch_code.copy()
    for i in range(40,1000):
        if not (os.path.exists(savePath+"model_supernet_{}.pt".format(i)) and savePath+'model_arch_code_{}.npy'.format(i)):
            break
        model_subi=torch.load(savePath+"model_supernet_{}.pt".format(i))
        archCodei = np.load(savePath+"model_arch_code_{}.npy".format(i))
        _,convEdgeCnt=archCodeAdList(archCodei,layer_num=LAYER_NUM)
        avg_acc=0
        t0=time.time()
        for epoch in range(0,5):
            avg_acc+=test_model(model_subi, test_loader)
        print("轮次：{}，卷积数量：{}，精度：{:.6f}".format(i,convEdgeCnt,avg_acc/5),"推理时间："+str(time.time()-t0))
def train_repair_model(arch_code,train_loader,test_loader,read=True):
    if read:
        repairNet =torch.load(savePath+"repairNet_0.pt").cuda(cudaId)
    else:
        repairNet=RepairNet(node_nums=6*LAYER_NUM,out_fea=10).cuda(cudaId)
    test_set=test_loader.dataset
    test_set1, test_set2 = random_split(test_set, [len(test_set) // 2, len(test_set) - len(test_set) // 2])
    # 分别创建两个数据加载器
    train_loader = DataLoader(test_set1, batch_size=BATCH_SZ, shuffle=False)
    test_loader = DataLoader(test_set2, batch_size=BATCH_SZ, shuffle=False)

    optimizer = optim.Adam(repairNet.parameters(), lr=0.01)
    superNet = torch.load(savePath+"model_supernet.pt").cuda(cudaId)  # 加载神经网络
    superNet.eval()
    for i in range(1,1000):
        if not (os.path.exists(savePath+"model_supernet_{}.pt".format(i)) and savePath+'model_arch_code_{}.npy'.format(i)):
            break
        nowNet=torch.load(savePath+"model_supernet_{}.pt".format(i)).cuda(cudaId)
        archCodei = np.load(savePath+"model_arch_code_{}.npy".format(i))
        # nowNet.setArch(archCodei,layer_num=len(nowNet.cells))
        nowNet.eval()
        for epoch in range(20):
            _train_repair_model(archCodei,superNet,nowNet,repairNet,optimizer,train_loader,epoch)
        if i%10==0:
            avg_acc=0
            torch.save(repairNet,savePath + "repairNet_{}.pt".format(i))
            for epoch in range(5):
                avg_acc+=_test_repair_model(archCodei, superNet, nowNet, repairNet, test_loader, epoch)
            print(avg_acc/5)


pipeline = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))])
# print(torch.cuda.is_available())
train_set=datasets.CIFAR10("picsData/cifar10",train=True,download=False,transform=pipeline)
test_set=datasets.CIFAR10("picsData/cifar10",train=False,download=False,transform=pipeline)
train_loader = DataLoader(train_set,batch_size=BATCH_SZ,shuffle=True)#加载数据集，shuffle=True是打乱数据顺序
test_loader = DataLoader(test_set,batch_size=BATCH_SZ,shuffle=False)
# model = torch.load(savePath+"model_supernet.pt")  # 加载神经网络
model_name="mysupernet10"
model = torch.load(savePath+model_name+".pt")  # 加载神经网络
# arch_code = np.load(savePath+"model_arch_code.npy")#加载神经网络 = torch.load("nasData/model_supernet_19.pt")#加载神经网络
optimizer = optim.Adam(model.parameters(), lr = 0.001)
# train_repair_model(arch_code,train_loader,test_loader)
# mkNasDataset_flight(arch_code,model,train_loader,test_loader)
arch_code=getSuperCode(LAYER_NUM)
purning_submodel_rand(arch_code,train_loader,test_loader,model_name)
# purning_rand(arch_code,model,optimizer,train_loader,test_loader,model_name)
# model_1=torch.load(savePath+"model_supernet_40.pt")  # 加载神经网络
# # model_1.forward=model_1.forward_fsp
# torchToMnn(model_1,savePath+"model_supernet_40.mnn",True)
# # model.forward=model.forward_fsp
# torchToMnn(model,savePath+"mysupernet10.mnn",True)
# for i in range(10):
#     getKernelList(model,train_loader,LAYER_NUM)
# torchToMnn(model,"/home/data/hw/Z_bing333/project/DGNAS/aaa.mnn")
# send_model("/home/data/hw/Z_bing333/project/DGNAS/aaa.mnn",host="10.12.11.248")
# mkNasDataset_greedy(arch_code,model,optimizer,train_loader,test_loader,model_name)
# model=SuperNet(arch_code,layer_num=5,out_fea=1000).cuda()
# model.setClassifier(5,1000,0)
exit()
model=torchvision.models.vgg19(pretrained=True)
print("model_supernet.pt 参数量:{:2f}".format(getParameterCnt(model = model)))
time0=time.time()
x=torch.ones([100,3,224,224])
x=model.forward(x)
x = x.max(1, keepdim=True)[1]  # 找到概率值最大的下标,这里调用了python的函数，也可以自己写
print(x.shape)
print(time.time()-time0)
# mkNasDataset_improve(train_loader,test_loader,30)
# testNasDataset(arch_code,model,optimizer,train_loader,test_loader)

