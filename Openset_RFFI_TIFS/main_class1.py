import copy
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F#激活函数
from torch import optim
import pickle
from arch_code import arch_code
from nasnetwork import SuperNet, TransNet, TorchResNet, TorchDenNet
from sklearn.metrics import roc_curve, auc , confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

from dataset_preparation import awgn, LoadDataset, ChannelIndSpectrogram
from deep_learning_models import TripletNet, identity_loss

import os
BATCH_SZ=128
LAYER_NUM=5
SAVEPATH="savedata/"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"  # 将 0 替换为您想要使用的 GPU 编号
#%%
def data_transform1(data, label,
    dev_range = np.arange(0,4, dtype = int),
    pkt_range = np.arange(0,60, dtype = int)
                        ):
    train_data = np.load(data)
    train_label = np.load(label)[:,np.newaxis].astype(int)
    label_start = 0
    label_end = 3
    num_dev = label_end - label_start + 1
    num_pkt = len(train_label)
    num_pkt_per_dev = int(num_pkt / num_dev)

    print('Dataset information: Dev ' + str(label_start) + ' to Dev ' +
          str(label_end) + ', ' + str(num_pkt_per_dev) + ' packets per device.')

    sample_index_list = []
    for dev_idx in dev_range:
        sample_index_dev = np.where(train_label == dev_idx)[0][pkt_range].tolist()
        sample_index_list.extend(sample_index_dev)
    data_list = []
    label_list = []
    for num in range(len(sample_index_list)):
        train_data1 = train_data[sample_index_list[num]][0] + train_data[sample_index_list[num]][1]*1j
        data_list.extend(train_data1)
        label_list.extend(train_label[sample_index_list[num]])
    data = np.array(data_list).astype(np.complex128).reshape(-1,320)
    label = np.array(label_list).astype(int).reshape(-1,1)

    return data, label

def data_transform_excel(data_path,
    dev_range = np.arange(0,7, dtype = int),
    pkt_range = np.arange(0,600, dtype = int)
        ):
    df = pd.read_excel(data_path, header=None)
    file_data = df.values
    np.random.shuffle(file_data)
    label = file_data[:,640].astype(int)
    label = np.transpose(label)
    label = label - 1

    label_start = int(np.min(label)) + 1
    label_end = int(np.max(label)) + 1
    num_dev = label_end - label_start + 1
    num_pkt = len(label)
    num_pkt_per_dev = int(num_pkt / num_dev)

    print('Dataset information: Dev ' + str(label_start) + ' to Dev ' +
          str(label_end) + ', ' + str(num_pkt_per_dev) + ' packets per device.')

    sample_index_list = []

    for dev_idx in dev_range:
        sample_index_dev = np.where(label == dev_idx)[0][pkt_range].tolist()
        sample_index_list.extend(sample_index_dev)
    #均值方差归一化
    # train_data_i = []
    # train_data_q = []
    # for num in range(len(sample_index_list)):
    #     train_data2 = file_data[sample_index_list[num], :-1]
    #     iq_i_1 = train_data2[::2]
    #     iq_q_1 = train_data2[1::2]
    #     train_data_i.extend(iq_i_1)
    #     train_data_q.extend(iq_q_1)
    # i_mean = np.mean(np.array(train_data_i).reshape(1,-1))
    # i_std = np.std(np.array(train_data_i).reshape(1,-1))
    # q_mean = np.mean(np.array(train_data_q).reshape(1,-1))
    # q_std = np.std(np.array(train_data_q).reshape(1,-1))

    #最大最小归一化
    train_data_i = []
    train_data_q = []
    for num in range(len(sample_index_list)):
        train_data2 = file_data[sample_index_list[num], :-1]
        iq_i_1 = train_data2[::2]
        iq_q_1 = train_data2[1::2]
        train_data_i.extend(iq_i_1)
        train_data_q.extend(iq_q_1)
    i_min = np.min(np.array(train_data_i).reshape(1,-1))
    i_max = np.max(np.array(train_data_i).reshape(1,-1))
    q_min = np.min(np.array(train_data_q).reshape(1,-1))
    q_max = np.max(np.array(train_data_q).reshape(1,-1))
    data_list = []
    for num in range(len(sample_index_list)):
        train_data1 = file_data[sample_index_list[num], :-1]
        # train_data1 = train_data1[,:-1]
        iq_i = train_data1[::2]
        iq_q = train_data1[1::2]
        iq_i_normalization = (iq_i - i_min)/(i_max-i_min)
        iq_q_normalization = (iq_q - q_min)/(q_max-q_min)
        train_data = iq_i_normalization +iq_q_normalization*1j
        data_list.extend(train_data)

        # label_list.extend(label[sample_index_list[num]])
    label_list = label[sample_index_list]
    data = np.array(data_list).astype(np.float32).reshape(-1, 320)
    label = np.array(label_list).astype(int).reshape(-1, 1)

    return data, label
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
        cutMain=True
        for i in range(LAYER_NUM):
            for j in main_li_0:
                if codeArch[i][j] >1:
                    li.append((i, j))
        print("before cuting, count of main edge: " + str(len(li)))
    return li,cutMain
def selectCut(codeArch,edge,cut0):#随机剪裁，保证网络一定是联通的
    x, y = edge[0], edge[1]
    if cut0:
        codeArch[x][y] = 1
    else:
        codeArch[x][y] = 0

def randCut(codeArch):#随机剪裁，保证网络一定是联通的
    codeArch=copy.deepcopy(codeArch)
    li,cut0=getEdgeList(codeArch)
    edge=li[random.randint(0,len(li)-1)]
    selectCut(codeArch,edge,cut0)
    return codeArch
def compareArchCode(codeArch1,codeArch2):#随机剪裁，保证网络一定是联通的
    isSame=True
    for i in range(LAYER_NUM):
       for j in range(len(codeArch1[i])):
           if codeArch1[i][j]!=codeArch2[i][j]:
               isSame=False
               print("edge({} {}): {} to {}".format(i,j,codeArch1[i][j],codeArch2[i][j]))
    if isSame:
        print("相同")
    return isSame
def toSeqData(data,i,next_i,n,lens=16):
    li=[]
    for i in range(i,next_i):
        if i+lens>n:
            ze=torch.zeros((i+lens-n,320))
            li.append(torch.cat((data[i:n],ze),dim=0))
        else:
            li.append(data[i:i+lens])
    return torch.stack(li,dim=0)
def train_model(model,data,label,optimizer,epoch):#model是模型，device是设备，train_loader是数据集，optimizer是梯度，epoch当前第几轮
    model.train()
    correct=0#正确率
    avgLoss=0
    # adjustOpt(optimizer, epoch)
    n=label.shape[0]
    for i in range(0,n,BATCH_SZ):
        next_i=min(i+BATCH_SZ,n)
        train_data=data[i:next_i].cuda()
        # train_data=toSeqData(data,i,next_i,n).cuda()
        # print(train_data.shape)
        target=label[i:next_i].to(torch.long).cuda()
        # print(target)
        optimizer.zero_grad()#初始化梯度为0
        output=model(train_data)#训练后结果
        # output=F.softmax(output,dim=0)
        loss=F.cross_entropy(output,target)#计算损失,用交叉熵,默认是累计的
        loss.backward()#反向传播
        avgLoss +=loss.item()
        optimizer.step()#参数优化
        pred = output.max(1, keepdim=True)[1]  # 找到概率值最大的下标,这里调用了python的函数，也可以自己写
        correct += pred.eq(target.view_as(pred)).sum().item()
    avgLoss /=n  # 计算平均数
    Accuracy=100 * correct / n
    print("第{}，正确率{:.6f}  loss：{:.6f}".format(epoch,Accuracy,avgLoss))
    return Accuracy,avgLoss
def test_model(model,data,label,epoch):#model是模型，device是设备，train_loader是数据集，optimizer是梯度，epoch当前第几轮
    model.eval()
    correct=0#正确率
    n=label.shape[0]
    t0=time.time()
    for i in range(0,n,BATCH_SZ):
        next_i = min(i + BATCH_SZ, n)
        train_data = data[i:next_i].cuda()
        # train_data = toSeqData(data, i, next_i, n).cuda()
        target = label[i:next_i].to(torch.long).cuda()
        output = model(train_data)  # 训练后结果
        # output = F.softmax(output, dim=0)
        pred = output.max(1, keepdim=True)[1]  # 找到概率值最大的下标,这里调用了python的函数，也可以自己写
        correct += pred.eq(target.view_as(pred)).sum().item()
    Accuracy=100 * correct / n
    print("第{}，正确率{:.6f} 延时：{:.6f}".format(epoch,Accuracy,time.time()-t0))
    return Accuracy
def cutcodeFromFSP(model,arch_code,data,label):#model是模型，device是设备，train_loader是数据集，optimizer是梯度，epoch当前第几轮
    model.eval()
    correct = 0  # 正确率
    n = label.shape[0]
    fsp=np.zeros((LAYER_NUM,14))
    for i in range(0, n, BATCH_SZ):
        next_i = min(i + BATCH_SZ, n)
        train_data = data[i:next_i].cuda()
        # train_data = toSeqData(data, i, next_i, n).cuda()
        target = label[i:next_i].to(torch.long).cuda()
        output,fspi = model.forward_fsp(train_data)  # 训练后结果
        fsp+=np.array(fspi)
        # output = F.softmax(output, dim=0)
        pred = output.max(1, keepdim=True)[1]  # 找到概率值最大的下标,这里调用了python的函数，也可以自己写
        correct += pred.eq(target.view_as(pred)).sum().item()
    # for j, h in enumerate(fsp):
    #     if h
    # print(fsp)
    arch_code=copy.deepcopy(arch_code)
    arch_code_mask = copy.deepcopy(arch_code)#mask==-1代表保护该边
    minfspij=10001000000
    nx,ny=-1,-1
    for i in range(LAYER_NUM):
        cnt=0
        mask_i,mask_j=0,0
        for j in main_li_0:
            if arch_code[i][j] !=0:
                mask_i, mask_j=i,j
                cnt+=1
        if cnt==1 and arch_code_mask[mask_i][mask_j]==1:
            arch_code_mask[mask_i][mask_j]=-1
        for j,h in enumerate(arch_code[i]):
            if arch_code[i][j]!=0 and minfspij>fsp[i][j] and arch_code_mask[i][j]!=-1:
                minfspij = fsp[i][j]
                nx, ny=i,j
    if nx==-1:
        return arch_code,False
    if ny in main_li_0:
        cnt = 0
        for j in main_li_0:
            if arch_code[nx][j] != 0:
                cnt += 1
        if cnt>1:
            arch_code[nx][ny]=0
        else:
            arch_code[nx][ny] = 1
    else:
        arch_code[nx][ny] = 0
    return arch_code,True

def trainSuperNet(data,label,data_0, label_0,data_test,label_test,data_test_0, label_test_0,MyNet=SuperNet):#torch.Size([2400, 1, 25, 8]),torch.Size([2400, 1])
    if MyNet!=SuperNet:
        net = MyNet(out_fea=7).cuda()
    else:
        net = MyNet(arch_code, layer_num=LAYER_NUM, out_fea=7).cuda()
        net.name="model_supernet"
    print(data_0.shape,label_0.shape)
    # data,label,data_test,label_test=data_0, label_0,data_test_0, label_test_0
    # net = TransNet(320, out_fea=4).cuda()
    # net = torch.load(SAVEPATH + "model_supernet.pt")
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    for epoch in range(200):
        Accuracy,avgLoss=train_model(net,data,label,optimizer,epoch)
        if epoch%10==0:
            test_model(net, data_test, label_test, epoch)
        if avgLoss < 0.0000005:
            break
    torch.save(net, SAVEPATH + net.name+".pt")  # 保存模型pt || pth
    net=torch.load(SAVEPATH + net.name+".pt")
    for epoch in range(10):
        test_model(net,data_test,label_test,epoch)
def mkNasDataset(arch_code,data,label,data_test,label_test):
    arch_code=arch_code.copy()
    model = torch.load(SAVEPATH + "model_supernet.pt").cuda()  # 加载神经网络
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for i in range(0,62):
        arch_code_now = arch_code
        randCut(arch_code_now)
        randCut(arch_code_now)
        # selectCut(arch_code_now, edge, cut0)
        model.setArch(arch_code_now, layer_num=len(model.cells))
        model.cuda()
        # model2=model2
        # model2.load_state_dict(model.state_dict())
        for epoch in range(0, 100):
            Accuracy, avgLoss = train_model(model, data, label, optimizer, epoch)
            if avgLoss < 0.000001:
                break
        torch.save(model, SAVEPATH + "model_supernet_{}.pt".format(i))  # 保存模型pt || pth
        np.save(SAVEPATH+'model_arch_code_{}.npy'.format(i), np.array(arch_code_now))
        avg_acc = 0
        for epoch in range(0, 5):
            avg_acc += test_model(model, data_test, label_test, epoch)
        avg_acc /= 5
        print("轮次：{}，精度：{:.6f}".format(i, avg_acc))
def mkNasDataset_test(arch_code,data,label,data_test,label_test):
    for i in range(5,6):
        # model=torch.load(SAVEPATH + "model_supernet_{}.pt".format(i))  # 保存模型pt || pth
        # np.save(SAVEPATH+'model_arch_code_{}.npy'.format(i), np.array(arch_code_now))
        arch_code_now=np.load(SAVEPATH+'model_arch_code_{}.npy'.format(i))
        print(arch_code_now)
        # model = torch.load(SAVEPATH + "model_supernet.pt")  # 保存模型pt || pth
        model=SuperNet(arch_code=arch_code_now,layer_num=7)
        model.setArch(arch_code_now, layer_num=len(model.cells))
        model.cuda()

        # model2=model2
        # model2.load_state_dict(model.state_dict())
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        for epoch in range(0, 1000):
            Accuracy, avgLoss = train_model(model, data, label, optimizer, epoch)
            if epoch % 10 == 0:
                test_model(model, data_test, label_test, epoch)
            if avgLoss < 0.000001 and epoch > 5:
                break
        # torch.save(model, SAVEPATH + "model_supernet_{}.pt".format(i))  # 保存模型pt || pth
        # np.save(SAVEPATH+'model_arch_code_{}.npy'.format(i), np.array(arch_code_now))
        avg_acc = 0
        for epoch in range(0, 5):
            avg_acc += test_model(model, data_test, label_test, epoch)
        avg_acc /= 5
        print("轮次：{}，精度：{:.6f}".format(i, avg_acc))
def mkNasDataset_greedy(arch_code,data,label,data_test,label_test):
    arch_code=arch_code.copy()
    model = torch.load(SAVEPATH + "model_supernet.pt").cuda()  # 加载神经网络
    # arch_code = np.load(SAVEPATH+'model_arch_code_{}.npy'.format(60))
    # model = torch.load(SAVEPATH + "model_supernet_{}.pt".format(60)).cuda()  # 加载神经网络
    test_model(model, data_test, label_test, 0)
    # arch_code[3]=[0]*14
    # arch_code[2] = [0] * 14

    for i in range(0,200):
        arch_code_now=arch_code
        Accuracy=100
        isCut=True
        while(Accuracy>100/7 and isCut):
            arch_code_now,isCut = cutcodeFromFSP(model, arch_code_now, data, label)
            # arch_code_now=randCut(arch_code_now)
            model.setArch(arch_code_now, layer_num=len(model.cells))
            model.cuda()
            Accuracy=test_model(model, data, label, 0)
        # arch_code_now = cutcodeFromFSP(model, arch_code_now, data, label)
        # print(arch_code,arch_code_now)
        isSame=compareArchCode(arch_code,arch_code_now)
        li,cutMain=getEdgeList(arch_code_now)
        print(arch_code_now)
        if isSame:
            break
        # randCut(arch_code_now)
        # randCut(arch_code_now)
        # # selectCut(arch_code_now, edge, cut0)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        # model2=model2
        # model2.load_state_dict(model.state_dict())
        for epoch in range(0, 200):
            Accuracy, avgLoss = train_model(model, data, label, optimizer, epoch)
            if epoch % 10 == 0:
                test_model(model, data_test, label_test, epoch)
            if avgLoss < 0.000001 and epoch>5:
                break
        torch.save(model, SAVEPATH + "model_supernet_{}.pt".format(i))  # 保存模型pt || pth
        np.save(SAVEPATH + 'model_arch_code_{}.npy'.format(i), np.array(arch_code_now))
        avg_acc = 0
        for epoch in range(0, 1):
            avg_acc += test_model(model, data_test, label_test, epoch)
        avg_acc /= 1
        print("轮次：{}，精度：{:.6f}".format(i, avg_acc))
        arch_code=arch_code_now


def shuffleData(data,label,data_0, label_0,T=1):
    up=data.shape[0]
    ind = torch.randperm(up)
    data = data[ind]
    label = label[ind]
    data_0 = data_0[ind]
    label_0 = label_0[ind]
    return data,label,data_0,label_0

    # while(T>0):
    #     for i in range(up-1, 0, -1):
    #         rnd = random.randint(0, i)  # 每次随机出0-i-1之间的下标
    #         # rnd1 = random.randint(0, up)  # 每次随机出0-i-1之间的下标
    #         # rnd2 = random.randint(0, up)  # 每次随机出0-i-1之间的下标
    #         # print(rnd)
    #         print(label[i], label[rnd])
    #         data[i], data[rnd] = data[rnd], data[i]
    #         label[i], label[rnd] = label[rnd], label[i]
    #         print(label[i], label[rnd])
    #     T-=1
    #     # self.deck[rnd1], self.deck[rnd2] = self.deck[rnd2], self.deck[rnd1]
def getTensorData(data_path,dev_range_enrol,pkt_range_enrol,snr_range ,isNor=True):
    # LoadDatasetObj = LoadDataset()

    # Load the enrollment dataset. (IQ samples and labels)
    # data_enrol, label_enrol = LoadDatasetObj.load_iq_samples(file_path_enrol,
    #                                                          dev_range_enrol,
    #                                                          pkt_range_enrol)
    # data_enrol, label_enrol = data_transform1(data=data_path, label=label_path, dev_range=dev_range_enrol, pkt_range =pkt_range_enrol)
    data_0, label_0 = data_transform_excel(data_path=data_path, dev_range=dev_range_enrol,
                                                   pkt_range=pkt_range_enrol)
    print('训练选取' + str(len(dev_range_enrol)) + '个设备,每个设备' + str(len(pkt_range_enrol)) + '个包被使用.')
    data=data_0
    label=label_0
    # print(label)
    if isNor==True:
        data = awgn(data, snr_range)
    ChannelIndSpectrogramObj = ChannelIndSpectrogram()
    # Convert IQ samples to channel independent spectrograms. (enrollment data)
    data = ChannelIndSpectrogramObj.channel_ind_spectrogram(data)
    data = torch.Tensor(data)
    data = data.view(data.shape[0], 1, 25, 8)
    label = torch.Tensor(label.tolist()).squeeze()
    target_size = (32, 32)
    # 计算在每个维度上的填充量
    pad_values = [12, 12, 3, 4]  # 初始化填充值为0
    # 进行填充操作
    data = F.pad(data, pad_values)

    data_0=torch.Tensor(data_0)
    label_0=torch.Tensor(label_0).squeeze()
    data, label,data_0, label_0 = shuffleData(data, label,data_0, label_0)
    return data, label,data_0, label_0
def addGuss(data):
    mean = 0.0
    stddev = 0.1
    noise = torch.randn(data.size()) * stddev + mean
    data = data + noise

    # 确保像素值范围在0到1之间
    data = torch.clamp(data, 0, 1)
    return data

def train_feature_extractor(data_path=None):
    if data_path!="" and data_path!=None:
        dev_range = np.arange(0, 7, dtype=int)
        pkt_range = np.arange(0, 600, dtype=int)
        snr_range = np.arange(0, 600, dtype=int)
        pkt_range_test = np.arange(600, 700, dtype=int)
        snr_range_test = np.arange(600, 700, dtype=int)
        data,label,data_0, label_0 =getTensorData(data_path,dev_range,pkt_range,snr_range,isNor=False)
        data_test, label_test,data_test_0, label_test_0 = getTensorData(data_path, dev_range, pkt_range_test, snr_range_test,isNor=False)
    else:
        xdir=readData()
        data, label, data_test, label_test=xdir['data'],xdir['label'],xdir['data_test'],xdir['label_test']
        data_0, label_0, data_test_0, label_test_0 = xdir['data_0'], xdir['label_0'], xdir['data_test_0'], xdir['label_test_0']
    # print(label_test)
    # data=addGuss(data)
    # data_test=addGuss(data_test)
    for i in range(5):
        trainSuperNet(data,label,data_0, label_0,data_test,label_test,data_test_0, label_test_0)
    # mkNasDataset_greedy(arch_code,data,label,data_test,label_test)
    # mkNasDataset(arch_code,data,label,data_test,label_test)
    mkNasDataset_test(arch_code,data,label,data_test,label_test)#直接训练结果不到80
    return None

def mkdataSet(data_path):
    dev_range = np.arange(0, 7, dtype=int)
    pkt_range = np.arange(0, 600, dtype=int)
    snr_range = np.arange(0, 600, dtype=int)
    pkt_range_test = np.arange(600, 700, dtype=int)
    snr_range_test = np.arange(0, 600, dtype=int)
    data,label,data_0, label_0 =getTensorData(data_path,dev_range,pkt_range,snr_range,isNor=False)
    data_test, label_test,data_test_0, label_test_0 = getTensorData(data_path, dev_range, pkt_range_test, snr_range_test,isNor=False)
    li=[data,label,data_0, label_0,data_test, label_test,data_test_0, label_test_0]
    print(data.shape,data_test.shape)
    xdir = {
        'data': data,
        'label': label,
        'data_0': data_0,
        'label_0': label_0,
        'data_test': data_test,
        'label_test': label_test,
        'data_test_0': data_test_0,
        'label_test_0': label_test_0,
    }

    # 指定保存文件的路径
    file_path = 'dataset/data.pkl'

    # 使用 pickle 序列化保存数据
    with open(file_path, 'wb') as f:
        pickle.dump(xdir, f)
    print("数据集保存成功")
def readData():
    file_path = 'dataset/data.pkl'
    with open(file_path, 'rb') as f:
        loaded_data = pickle.load(f)
    print(loaded_data['data'].shape)
    print("数据集读取成功")
    return loaded_data
if __name__ == '__main__':

    # Specifies what task the program runs for.
    # 'Train'/'Classification'/'Rogue Device Detection'
    run_for = 'Train'
    data_path = 'test.xlsx'
    # mkdataSet(data_path)
    if run_for == 'Train':
        feature_extractor = train_feature_extractor()



