import os
import socket
import time

import MNN
import cv2
import numpy as np
def capture_image_from_camera():
    # 打开摄像头，参数 0 表示默认的摄像头，如果有多个摄像头，可以逐个尝试 0, 1, 2, ...
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头。")
        return
    # 读取一帧图像
    ret, frame = cap.read()
    if not ret:
        print("无法捕获图像。")
        cap.release()
        return
    # 关闭摄像头
    cap.release()
    return frame


def getExampleInput(sh=(3,32,32),image=None):
    if image.all()==None:
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
pic=capture_image_from_camera()
pic=cv2.resize(pic, (32, 32))
exampleImg=getExampleInput(sh=(3,32,32),image=pic)
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

    # tmp_output= MNN.Tensor((1, 10), MNN.Halide_Type_Float, np.ones([1, 10]).astype(np.float32), MNN.Tensor_DimensionType_Caffe)
    # output_tensor.copyToHostTensor(tmp_output)#把output_tensor拷贝到tmp_output
    #
    # print(tmp_output)
    # print("expect 983")
    # print("output belong to class: {}".format(np.argmax(tmp_output.getData())))
    return realTime/cnt,mint,maxt

SERVER_HOST = '0.0.0.0'  # 监听所有网络接口，填自己就行
SERVER_PORT = 8888  # 服务器端口号
BUFFER_SIZE = 4096  # 缓冲区大小

def receive_file(file_path):
    # 创建服务器套接字
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((SERVER_HOST, SERVER_PORT))
    server_socket.listen(1)

    while(True):
        print(f"等待客户端连接...")

        client_socket, client_address = server_socket.accept()
        print(f"接受来自 {client_address} 的连接")
        data=client_socket.recv(1024)
        indi=data.rfind(b'EOF')
        if indi!=-1:
            dataStr=data[:indi]
            data=data[indi+len(b'EOF'):]
        else:
            dataStr=data
            data=b''
        # print(dataStr)
        file_name=dataStr.decode('utf-8')
        # # 接收文件
        # print("接收文件",dataStr)
        with open(file_path+file_name, 'wb') as file:
            file.write(data)
            indi=-1
            while indi==-1:
                data = client_socket.recv(BUFFER_SIZE)
                if not data:
                    break
                indi=data.rfind(b'EOF')
                if indi!=-1:
                    data=data[:indi]
                file.write(data)
            print(f"接收成功：{file_path+file_name},开始推理")
            #推理结果
        realTime,mint,maxt=getMnnKernelTime(file_path+file_name)
        pic_1 = pic.tobytes()
        client_socket.sendall(pic_1+b'EOF')
        client_socket.sendall(("avgTime:"+str(realTime)).encode('utf-8'))

        client_socket.close()
        break
    # server_socket.close()

# 使用示例
path="F:/recfile/"
receive_file(path)
