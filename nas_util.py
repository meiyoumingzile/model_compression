
import copy
import math
import os
import socket
import subprocess
import time
from collections import deque
import random

import MNN
import cv2
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, Subset, random_split
from ftplib import FTP


main_li_0=[0,1,2,3,5,6,9,10]
main_li_1=[4,7,8,11,12,13]
def Kahn(mat):#邻接矩阵拓扑排序
    n=len(mat)
    du=[0]*n
    order = [0] * n
    st=deque()
    for i in range(n):
        for j in range(n):
            if mat[i][j]!=0:
                du[j]+=1
    for i in range(n):
        if du[i]==0:
            st.append(i)
    order_val=0
    while(len(st)!=0):
        now=st.pop()
        order[now]=order_val
        order_val+=1
        for i in range(n):
            if mat[now][i]!=0:
                du[i]-=1
                if du[i]==0:
                    st.append(i)

    for i in range(n):#检测是不是无环图
        if du[i]!=0:
            print("不是DAG")
            return None
    return order


def toMat(li,n):#编集合数组，转换邻接矩阵
    mat0=[0 for i in range(n)]
    mat=[mat0.copy() for i in range(n)]
    for a in li:
        mat[a[0]][a[1]]=a[2]
    return mat
def archCodeAdList(archCode,layer_num):
    layerNodeCnt=6
    adList = [[] for i in range(layer_num * layerNodeCnt)]
    edgeCnt=0
    for i in range(layer_num):
        s=0
        for k in range(2, 6):
            for j in range(k):
                e = (i* layerNodeCnt + j, i * layerNodeCnt + k, archCode[i][s+j])
                adList[e[0]].append(e)
                if e[2]>1:
                    edgeCnt+=1
            s+=k
        if i>0:
            for j in range(2,6):
                e=((i-1)*layerNodeCnt+j,i*layerNodeCnt+1,1)
                adList[e[0]].append(e)
        if i>1:
            for j in range(2,6):
                e=((i-2)*layerNodeCnt+j,i*layerNodeCnt,1)
                adList[e[0]].append(e)
    return adList,edgeCnt
def archCodeEncoder(archCode,layer_num):
    edge_index, edge_fea=[[],[]],[]
    layerNodeCnt=6
    n=layer_num * layerNodeCnt
    adList,_=archCodeAdList(archCode,layer_num)
    node_fea=torch.ones((n,1))
    for i in range(n):
        for edge in adList[i]:
            edge_index[0].append(edge[0])
            edge_index[1].append(edge[1])
            ze=torch.zeros((4))
            ze[edge[2]]=1
            edge_fea.append(ze)
    edge_fea=torch.stack(edge_fea,dim=0)
    # print(edge_fea.shape)
    return node_fea, torch.tensor(edge_index,dtype=torch.long), edge_fea
class Edge():
    def __init__(self,next,val):
        self.next=next
        self.val = val
def getFSP(model,train_loader,LAYER_NUM):
    model.eval()
    correct = 0  # 正确率
    fsp=np.zeros((LAYER_NUM,14))
    for batch_index, (d,t) in enumerate(train_loader):
        train_data, target = d.cuda(), t.cuda()  # 部署到device
        output,fspi = model.forward_fsp(train_data)  # 训练后结果
        fsp+=np.array(fspi)
        # output = F.softmax(output, dim=0)
        pred = output.max(1, keepdim=True)[1]  # 找到概率值最大的下标,这里调用了python的函数，也可以自己写
        correct += pred.eq(target.view_as(pred)).sum().item()
    return fsp
def cutcodeFromFSP(model,arch_code,train_loader,LAYER_NUM):#model是模型，device是设备，train_loader是数据集，optimizer是梯度，epoch当前第几轮
    fsp=getFSP(model,train_loader,LAYER_NUM)
    # for j, h in enumerate(fsp):
    #     if h
    # print(timeLi,arch_code)
    # exit()
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
def cutcodeFromDeep(model,arch_code,train_loader,LAYER_NUM):#model是模型，device是设备，train_loader是数据集，optimizer是梯度，epoch当前第几轮
    arch_code = copy.deepcopy(arch_code)
    for i in reversed(range(LAYER_NUM)):
        li=[]
        for j in main_li_1:
            if arch_code[i][j]!=0:
                li.append((i,j))
        if len(li)>0:
            nx,ny=li[random.randint(0,len(li)-1)]
            arch_code[nx][ny] = 0
            return arch_code,True

        for j in main_li_0:
            if arch_code[i][j] != 0:
                li.append((i, j))
        if len(li) > 1:
            nx, ny = li[random.randint(0, len(li) - 1)]
            arch_code[nx][ny] = 0
            return arch_code, True
        elif len(li) == 1 and arch_code[li[0]][li[1]]!=1:
            nx, ny = li[0]
            arch_code[nx][ny] = 1
            return arch_code, True

    return arch_code,False
def compareArchCode(codeArch1,codeArch2,LAYER_NUM=5):#随机剪裁，保证网络一定是联通的
    isSame=True
    for i in range(LAYER_NUM):
       for j in range(len(codeArch1[i])):
           if codeArch1[i][j]!=codeArch2[i][j]:
               isSame=False
               print("edge({} {}): {} to {}".format(i,j,codeArch1[i][j],codeArch2[i][j]))
    if isSame:
        print("相同")
    return isSame

def calcForwardTime(timeListi,timeListi_add):
    sumTime=0
    for t in timeListi_add:
        for a in t:
            sumTime += a
    for t in timeListi:
        for a in t:
            sumTime+=a

    return sumTime
def getParameterCnt(model=None):#得到参数量
    if model==None:
        model = torchvision.models.vgg16(pretrained=False)
    total_params = sum(p.numel() for p in model.parameters())
    return total_params/1000000
def getKernelList(model,test_loader,layer_num=20,cudaId=0):
    model.cuda()
    acc,timeLi,timeLi_add=test_model_time(model,test_loader,layer_num,cudaId,modelName="model")
    arch_code = np.zeros((layer_num,14)).tolist()
    model0= copy.deepcopy(model)
    model0.setArch(arch_code, layer_num=layer_num)
    model0.cuda()
    _,timeLi_0,timeLi_add_0=test_model_time(model0,test_loader,layer_num,cudaId,modelName="model_0")
    # print(timeLi_0,timeLi)
    return acc,timeLi,timeLi_0,timeLi_add,timeLi_add_0
    # for j, h in enumerate(fsp):
def getsubdataloader(dataloader,rate=0.5):
    if rate>=1:
        return dataloader
    dataset=dataloader.dataset
    dataset_length = len(dataset)
    new_dataset_length = math.ceil(dataset_length*rate)

    # 创建索引列表
    indices = list(range(dataset_length))

    # 随机划分索引为原始数据集和剩余数据集
    subset_indices, _ = random_split(indices, [new_dataset_length, dataset_length - new_dataset_length])

    # 创建子集
    subset = Subset(dataset, subset_indices)

    # 创建 DataLoader
    # print(len(subset),len(dataset))
    dataloader = DataLoader(subset, batch_size=dataloader.batch_size, shuffle=True)
    return dataloader
def test_model(model,test_loader,cudaId=0):##model是模型，device是设备，test_loader是测试数据集
    model.eval()#模型验证
    correct=0#正确率
    with torch.no_grad():#，不进行训练时
        t0=time.time()
        for batch_index, (data, target) in enumerate(test_loader):
            # print(data.shape)
            data,target=data.cuda(cudaId),target.cuda(cudaId)#部署到device
            output=model.forward(data)#训练后结果
            pred = output.max(1, keepdim=True)[1]  # 找到概率值最大的下标,这里调用了python的函数，也可以自己写
            correct+=pred.eq(target.view_as(pred)).sum().item()
        return 100*correct/len(test_loader.dataset)
def test_model_norm(model,test_loader,cudaId=0):##model是模型，device是设备，test_loader是测试数据集
    model.eval()#模型验证
    correct=0#正确率
    with torch.no_grad():#，不进行训练时
        t0=time.time()
        for batch_index, (data, target) in enumerate(test_loader):
            # print(data.shape)
            data,target=data.cuda(cudaId),target.cuda(cudaId)#部署到device
            output,norm_mat=model.forward_norm(data)#训练后结果
            print(norm_mat)
            pred = output.max(1, keepdim=True)[1]  # 找到概率值最大的下标,这里调用了python的函数，也可以自己写
            correct+=pred.eq(target.view_as(pred)).sum().item()
        return 100*correct/len(test_loader.dataset)
def test_model_time(model,test_loader,layer_num=20,cudaId=0,modelName=""):##model是模型，device是设备，test_loader是测试数据集
    model.eval()#模型验证
    correct=0#正确率
    with torch.no_grad():#，不进行训练时
        simple=torch.zeros((1,3,32,32)).cuda(cudaId)
        model.forward_time(simple)  # 训练后结果
        timeLi0=np.zeros((layer_num,14))
        timeLi0_add = np.zeros((layer_num+3))
        t0=time.time()
        generTime=0
        for batch_index, (data, target) in enumerate(test_loader):
            # print(data.shape)
            batch_time_0 = time.time()
            data,target=data.cuda(cudaId),target.cuda(cudaId)#部署到device
            output,timeLi,timeLi_add=model.forward_time(data)#训练后结果
            generTime_0 = time.time()
            pred = output.max(1, keepdim=True)[1]  # 找到概率值最大的下标,这里调用了python的函数，也可以自己写
            correct+=pred.eq(target.view_as(pred)).sum().item()
            timeLi_add=torch.nn.functional.pad(timeLi_add, (0, 1),value=time.time() - generTime_0)
            timeLi = timeLi.numpy()
            timeLi_add = timeLi_add.numpy()
            timeLi0 += timeLi
            timeLi0_add += timeLi_add
            # print(timeLi.sum() , timeLi_add.sum(), time.time() - batch_time_0,data.shape)
        print("测试模型："+modelName)
        print("算子总时间:{:.12f}, 相加总时间:{:.12f}, 合计:{:.12f}, 真实时间:{:.12f}".format(timeLi0.sum(),timeLi0_add.sum(),timeLi0.sum()+timeLi0_add.sum(),time.time()-t0))

        n=len(test_loader.dataset)
        return 100*correct/n,timeLi0/n,timeLi0_add/n
def estimateNetTime(arch_code,model,test_loader,timeLi_super,timeLi0_add_super,layer_num=20,cudaId=0):##model是模型，device是设备，test_loader是测试数据集
    model.eval()#模型验证
    correct=0#正确率
    predSumTime=0
    datasetlen=len(test_loader.dataset)
    for i in range(layer_num):
        for j in range(14):
            if arch_code[i][j]!=0:
                predSumTime+=timeLi_super[i][j]
    with torch.no_grad():#，不进行训练时
        simple=torch.zeros((1,3,32,32)).cuda(cudaId)
        model.forward_time(simple)  # 训练后结果
        timeLi0=np.zeros((layer_num,14))
        timeLi0_add = np.zeros((layer_num+3))
        t0=time.time()
        generTime=0
        for batch_index, (data, target) in enumerate(test_loader):
            # print(data.shape)
            batch_time_0 = time.time()
            data,target=data.cuda(cudaId),target.cuda(cudaId)#部署到device
            output,timeLi,timeLi_add=model.forward_time(data)#训练后结果
            generTime_0 = time.time()
            pred = output.max(1, keepdim=True)[1]  # 找到概率值最大的下标,这里调用了python的函数，也可以自己写
            correct+=pred.eq(target.view_as(pred)).sum().item()
            timeLi_add = torch.nn.functional.pad(timeLi_add, (0, 1), value=time.time() - generTime_0)
            timeLi = timeLi.numpy()
            timeLi_add = timeLi_add.numpy()
            timeLi0 += timeLi
            timeLi0_add += timeLi_add

            # print(timeLi.sum() , timeLi_add.sum(), time.time() - batch_time_0,data.shape)
        print("预测值：",(predSumTime+timeLi0_add_super.sum())*datasetlen,timeLi0.sum()+timeLi0_add.sum(),time.time()-t0)
        return 100*correct/datasetlen,timeLi0/datasetlen,timeLi0_add/datasetlen
def estimate_cell_weight(arch_code,model,train_loader,LAYER_NUM,val=0):
    model.eval()
    n,m=LAYER_NUM,14
    ans=torch.zeros((LAYER_NUM,14))
    acc_super = test_model(model, train_loader)
    for x in range(LAYER_NUM):
        for y in range(len(arch_code[x])):
            if arch_code[x][y]!=0 and arch_code[x][y]!=val:
                code_now = arch_code[x][y]
                arch_code[x][y] = val
                model.clearWaste(LAYER_NUM)
                model.setArch(arch_code, layer_num=len(model.cells),out_fea=10,C=16)
                model.cuda()
                acc=test_model(model,train_loader)
                acc_sub=acc-acc_super
                ans[x][y]=acc_sub
                arch_code[x][y]=code_now
                model.setBack(LAYER_NUM)
                # print(x,y,acc_sub,acc)
    return ans
def cutcell_argmax(arch_code,loss_acc,LAYER_NUM,val=0):
    max_acc=-1000
    ans_x,ans_y=-1,-1
    for x in range(LAYER_NUM):
        for y in range(len(arch_code[x])):
            if arch_code[x][y]!=0 and max_acc<loss_acc[x][y] and arch_code[x][y]!=val:
                max_acc=loss_acc[x][y]
                ans_x,ans_y=x,y
    return ans_x,ans_y

def torchToOnnx(model,model_path="res.onnx",output_names=['output'],dyn=None):
    model.cpu().eval()
    dummy_input = torch.randn([1, 3, 32, 32]).cpu() # batch,channel,height,width
    input_names=["input"]
    torch.onnx.export(model, dummy_input, model_path, opset_version=11, input_names=['input'],
                      output_names=output_names,
                      dynamic_axes=dyn)
def onnxExampletime(model_path):
    import onnxruntime

    ort_session = onnxruntime.InferenceSession(model_path)
    input_name = ort_session.get_inputs()[0].name
    input_data=np.random.rand(1,3,32,32).astype(np.float32)
    # 运行推理，传入输入数据
    output = ort_session.run(None, {input_name: input_data})
    print(output)

def onnxToMnn(onnx_name,mnnname,path="/home/data/hw/Z_bing333/package/MNN/build/"):
    cmd_dtr = "./MNNConvert -f ONNX --modelFile "+onnx_name+" --MNNModel "+mnnname+" --bizCode biz"
    print(path+cmd_dtr)
    # os.system("conda activate nas_py37cu114")
    res_str=subprocess.run(path+cmd_dtr, shell=True, capture_output=True)
    # print(res_str)
    # res_str = res.stdout.decode()
    # print(res)
    # if "Converted Done" in res_str:
    #     print("Converted ok")
def torchToMnn(model,model_path,putTime=False):
    directory, mnnname = model_path.rsplit('/', 1)
    name, extension  = mnnname.rsplit('.', 1)
    directory+="/"
    onnxname=name+".onnx"
    if putTime:
        output_names=['output',"t0","t1"]
        dyn={'output': {0: 'batch_size'}, output_names[0]: {0: 'n'},
                                        output_names[1]: {0: 'n'}}
        # output_names=['output']
        # dyn={'output': {0: 'batch_size'}}
        torchToOnnx(model,directory+onnxname,output_names,dyn)
    else:
        torchToOnnx(model, directory + onnxname)
    onnxToMnn(directory+onnxname,directory+mnnname)
def send_model(model_path,host='10.12.11.236',port=8888,user='hong', passwd='c214216',BUFFER_SIZE=4096):#发送文件
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((host, port))
    file_name=os.path.basename(model_path)
    print(f"等待发送：{model_path}")
    client_socket.sendall( (file_name).encode('utf-8')+ b'EOF')

    # client_socket.sendall(("").encode('utf-8'))
    with open(model_path, 'rb') as file:
        while True:
            data = file.read(BUFFER_SIZE)
            if not data:
                client_socket.sendall(b'EOF')
                break
            client_socket.sendall(data)
        print(f"发送完毕")
    data = client_socket.recv(BUFFER_SIZE).decode('utf-8')
    print(f"文件发送完成：{model_path}", data)
def mnnforward(mnnmodel,pic):#pic是一个np数组
    session = mnnmodel.createSession()
    image = pic.astype(np.float32)
    input_tensor = mnnmodel.getSessionInput(session)
    tmp_input = MNN.Tensor((1, 3, 32, 32), MNN.Halide_Type_Float,\
                    image, MNN.Tensor_DimensionType_Caffe)
    input_tensor.copyFrom(tmp_input)
    mnnmodel.runSession(session)
    output_tensor = mnnmodel.getSessionOutput(session)
    #constuct a tmp tensor and copy/convert in case output_tensor is nc4hw4
    tmp_output = MNN.Tensor((1, 1001), MNN.Halide_Type_Float, np.ones([1, 1001]).astype(np.float32), MNN.Tensor_DimensionType_Caffe)
    output_tensor.copyToHostTensor(tmp_output)
    print(output_tensor)
    print("expect 983")
    print("output belong to class: {}".format(np.argmax(tmp_output.getData())))
def example(model_path):
    """ inference mobilenet_v1 using a specific picture """
    if not os.path.exists(model_path):
        print("模型不存在")
        return
    interpreter = MNN.Interpreter(model_path)
    session = interpreter.createSession()
    input_tensor = interpreter.getSessionInput(session)
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
    #cv2 read shape is NHWC, Tensor's need is NCHW,transpose it
    tmp_input = MNN.Tensor((1, 3, 32, 32), MNN.Halide_Type_Float,\
                    image, MNN.Tensor_DimensionType_Caffe)
    # print("ssaddsasda")
    input_tensor.copyFrom(tmp_input)
    # print("ssaddsasda")
    interpreter.runSession(session)
    # print("ssaddsasda")
    # li0,li1,li2=interpreter.getSessionOutputAll(session)
    print(interpreter.getSessionOutputAll(session))
    output_tensor = interpreter.getSessionOutput(session)
    print(output_tensor)
    #constuct a tmp tensor and copy/convert in case output_tensor is nc4hw4
    tmp_output= MNN.Tensor((1, 10), MNN.Halide_Type_Float, np.ones([1, 10]).astype(np.float32), MNN.Tensor_DimensionType_Caffe)
    output_tensor.copyToHostTensor(tmp_output)#把output_tensor拷贝到tmp_output

    print(tmp_output)
    print("expect 983")
    print("output belong to class: {}".format(np.argmax(tmp_output.getData())))
def getExampleInput(sh=(3,32,32)):
    image = np.random.randint(0, 256, (sh[1], sh[2], sh[0]), dtype=np.uint8)
    #cv2 read as bgr format
    image = image[..., ::-1]
    #change to rgb format
    image = cv2.resize(image, (sh[1], sh[2]))
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
    tmp_input = MNN.Tensor((1,sh[0], sh[1], sh[2]), MNN.Halide_Type_Float,\
                    image, MNN.Tensor_DimensionType_Caffe)
    return tmp_input
exampleImg=getExampleInput()
def getMnnKernelTime(model_path,cnt=100):
    """ inference mobilenet_v1 using a specific picture """
    if not os.path.exists(model_path):
        print("模型不存在")
        return
    interpreter = MNN.Interpreter(model_path)
    session = interpreter.createSession()
    input_tensor = interpreter.getSessionInput(session)
    # print("ssaddsasda")
    realTime=0
    mint,maxt=100000,0

    for _ in range(cnt):
        t0=time.time()
        input_tensor.copyFrom(exampleImg)
        interpreter.runSession(session)
        nowtime = time.time() - t0
        ans=interpreter.getSessionOutputAll(session)
        realTime+= nowtime
        mint, maxt=min(mint,nowtime),max(maxt,nowtime)
        # print(nowtime)
    for k in ans.keys():
        tensor_shape = ans[k].getShape()
        ans[k]=np.array(ans[k].getData(), copy=False).reshape(tensor_shape)
    return realTime/cnt,mint,maxt
def getAllMnnKernelTime(model_pathList,cnt=100):
    li = []
    for model_path in model_pathList:
        t0,t1,t2=getMnnKernelTime(model_path,cnt)
        li.append([t0,t1,t2])
        print(li[-1])
    return np.array(li)
# ----------------------------------------------------


# example("F:/recfile/mnnname.mnn")
# example("/home/data/hw/Z_bing333/project/DGNAS/nasData/train_dist/mysupernet10.mnn")

# torchToMnn(None,"2121/212/model_supernet_7.onnx")
# onnxToMnn("model_supernet_7.onnx","mnnname.mnn")
# recModel("/home/data/hw/Z_bing333/package/modsave/")