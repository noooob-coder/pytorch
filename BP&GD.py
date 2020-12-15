import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import torch.nn.functional as F
from torch.autograd import Variable
batch_size=64#设置处理数据批次大小
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")#设置设备
loss=nn.CrossEntropyLoss() #选择损失函数
def Load():#加载数据函数
    train_data=torchvision.datasets.MNIST(root="./mnist",
                                          train=True,
                                          transform=transforms.ToTensor(),
                                          download=True)
    test_data=torchvision.datasets.MNIST(root='./mnist',
                                         train=False,
                                         transform=transforms.ToTensor(),
                                         )
#设置数据集，root参数指定存放地点，train参数指定是否为训练集，transform参数确定数据转换的方式，download参数确定是否进行下载
    train_loader=torch.utils.data.DataLoader(dataset=train_data,
                                             batch_size=batch_size,
                                             shuffle=True)
    test_loader=torch.utils.data.DataLoader(dataset=test_data,
                                            batch_size=batch_size,
                                            shuffle=True)
#加载数据，batch_size用于确定单次训练时送入的样本数量，shuffle决定是否打乱样本
    return train_loader,test_loader

class Module_1(nn.Module):#模型设计
    def __init__(self):
        super(Module_1,self).__init__()#继承父类的init函数即nn.Module的init
        self.linear1=nn.Linear(784,10)#定义一个全链接层第一个参数表示输入，第二个参数表示输出，一张图784个像素点，输出0到9十个数
    def forward(self,X):
        return F.relu(self.linear1(X)) #设置激活函数relu
module=Module_1().to(device) #将网络实例化
train_data,test_data=Load()
optimizer = torch.optim.Adam(module.parameters())#设置优化算法
#print(module)#打印模型
def train():#设置训练函数                                         transform=transforms.ToTensor(),
    num_epochs = 100
    for echo in range(num_epochs):
        train_loss=0
        train_acc=0
        module.train() #将网络设置为训练模式
        for i,(X,label) in enumerate(train_data):
            X,label=X.to(device),label.to(device)
            X=X.view(-1,784)#将X展开成784的向量
            X=Variable(X)
            label = Variable(label)
            out = module(X)#正向传播计算结果
            lossvalue=loss(out,label)#计算损失值
            optimizer.zero_grad()#将梯度归零
            lossvalue.backward()#反向传播刷新梯度
            optimizer.step() #优化器运行
            train_loss +=float(lossvalue)#计算损失
            _, pred = out.max(1)
            num_correct = (pred == label).sum()
            acc = int(num_correct) / X.shape[0]
            train_acc += acc # 计算精确度
        print("echo:" + ' ' + str(echo))
        print("lose:" + ' ' + str(train_loss / len(train_data)))
        print("accuracy:" + ' ' + str(train_acc / len(train_data)))
        eval_loss = 0
        eval_acc = 0
        module.eval()  # 模型转化为评估模式
        for X, label in test_data:
            X, label = X.to(device), label.to(device)
            X = X.view(-1, 784)
            X = Variable(X)
            label = Variable(label)
            testout = module(X)
            testloss = loss(testout, label)
            eval_loss += float(testloss)
            _, pred = testout.max(1)
            num_correct = (pred == label).sum()
            acc = int(num_correct) / X.shape[0]
            eval_acc += acc
        print("testlose: " + str(eval_loss / len(test_data)))
        print("testaccuracy:" + str(eval_acc / len(test_data)) + '\n')
train()














