import os
import socket
import numpy as np
def recPic(client_socket,BUFFER_SIZE):
    pic_byts=b''
    data=None
    while(True):
        if data==None:
            data = client_socket.recv(BUFFER_SIZE)
        indi = data.rfind(b'EOF')
        if indi!=-1:
            pic_byts+=data[:indi]
            data=data[indi+len(b'EOF'):]
            break
        else:
            pic_byts+=data
            data=None
    pic=np.frombuffer(pic_byts, dtype=np.uint8)
    return pic,data
def send_model(model_path,host='10.12.11.248',port=8888,user='hong', passwd='c214216',BUFFER_SIZE=4096):#发送文件
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((host, port))
    file_name=os.path.basename(model_path)
    print(f"等待发送：{model_path}")
    client_socket.sendall(file_name.encode('utf-8')+b'EOF')
    with open(model_path, 'rb') as file:
        while True:
            data = file.read(BUFFER_SIZE)
            if not data:
                break
            client_socket.sendall(data)
        client_socket.sendall(b'EOF')
    print(f"文件发送完成：{model_path}")

    pic,data=recPic(client_socket,BUFFER_SIZE)
    print(pic.shape)
    messdata=data.decode('utf-8')
    print("返回消息："+messdata)
    client_socket.close()

def send_cmd_forward(model_path,host='10.12.11.248',port=8888,user='hong', passwd='c214216',BUFFER_SIZE=4096):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((host, port))
    file_name = os.path.basename(model_path)
    print(f"等待发送：{model_path}")
    client_socket.sendall("start run".encode('utf-8') + b'EOF')
    client_socket.sendall(file_name.encode('utf-8'))
    while(True):
        pic, data = recPic(client_socket, BUFFER_SIZE)
        print(pic.shape)
        messdata = data.decode('utf-8')
        print("返回消息：" + messdata)
    # client_socket.close()
# 使用示例

send_model('nasData/train_dist/mysupernet10.mnn')#nasData/train_dist/mysupernet10.mnn
send_cmd_forward('nasData/train_dist/mysupernet10.mnn')