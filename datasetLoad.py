import torch
import torchvision
import numpy as np
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
transform_train=transforms.Compose([
    # transforms.Resize([224,224]),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=(45, 45)),
    # transforms.RandomCrop(200),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.2892285, 0.14499652, -0.0267012], [0.37928897, 0.3529896, 0.33651316]),
])
datasetNameDir={
    "cifar10":"CIFAR10",
    "cifar100":"CIFAR100",
    "svhn":"SVHN",
    "mnist":"MNIST",
    "fashionmnist":"FashionMNIST",
    "omniglot":"Omniglot",
    "miniimagenet":"miniimagenet",
}
class datasetLoad():
    def __init__(self,datasetName):
        super(datasetLoad, self).__init__()
        self.datasetName=datasetName.lower()
        if not (self.datasetName in datasetNameDir):
            print("no exist dataset!!!")
            return
        print( "dataset is "+datasetNameDir[self.datasetName])
        self.sum_dataset = torchvision.datasets.__dict__[datasetNameDir[self.datasetName]](root="picsData/"+self.datasetName+"/", transform=transform_train,
                                                            download=False)
        self.len=len(self.sum_dataset)
        print(self.len)
        self.mid=self.len//4*3
        self.train_set=torch.utils.data.Subset(self.sum_dataset, range(0,self.mid))
        self.test_set=torch.utils.data.Subset(self.sum_dataset, range(self.mid+1,self.len))
        self.train_loader = DataLoader(self.train_set,batch_size=64,shuffle=True)#加载数据集，shuffle=True是打乱数据顺序
        self.test_loader = DataLoader(self.test_set,batch_size=64,shuffle=True)#加载数据集，shuffle=True是打乱数据顺序
        self.train_iter=iter(self.train_loader)
        self.test_iter=iter(self.test_loader)
    def getTrainPic(self):#可以无限调用
        try:
            return next(self.train_iter)
        except StopIteration:
            self.train_iter=iter(self.train_loader)
            return next(self.train_iter)

    def getTestPic(self):#可以无限调用

        try:
            return next(self.test_iter)
        except StopIteration:
            self.test_iter=iter(self.test_set)
            return next(self.test_iter)
d=datasetLoad("cifar10")
for i in range(4):
    example, labels= d.getTrainPic()
    print(example.shape,labels.shape)