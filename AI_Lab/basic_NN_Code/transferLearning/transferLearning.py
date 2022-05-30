import os
import matplotlib.pyplot as plt

import numpy as np
import torch
from torch import nn
import torch.optim as optim
import torchvision
#pip install torchvision
from torchvision import transforms, models, datasets
#https://pytorch.org/docs/stable/torchvision/index.html
import imageio
import time
import warnings
warnings.filterwarnings("ignore")
import random
import sys
import copy
import json
from PIL import Image

# 数据读取与预处理操作
data_dir = './flower_data/'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'

'''
制作好数据源：
data_transforms中指定了所有图像预处理操作
ImageFolder假设所有的文件按文件夹保存好，每个文件夹下面存贮同一类别的图片，文件夹的名字为分类的名字
'''
data_transforms = {
    'train':
        transforms.Compose([
        transforms.Resize([96, 96]), # 所有的图像数据都resize为相同的尺寸
#数据增强：
        transforms.RandomRotation(45),#随机旋转，-45到45度之间随机选
        transforms.CenterCrop(64),#从中心开始裁剪，图像块64 * 64，实际模型输入也是64 * 64，而不是96*96
        transforms.RandomHorizontalFlip(p=0.5),#随机水平翻转 选择一个概率概率
        transforms.RandomVerticalFlip(p=0.5),#随机垂直翻转
        transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),#参数1为亮度，参数2为对比度，参数3为饱和度，参数4为色相
        transforms.RandomGrayscale(p=0.025),#概率转换成灰度率，3通道就是R=G=B
        transforms.ToTensor(),# 数据转换为一个Tensor()
        #对 R G B 三个通道进行标准化，下面一个list是均值，一个是list是标准差
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])#均值，标准差
    ]),
    'valid': # 验证集不是用来数据的，不需要做数据增强
        transforms.Compose([
        transforms.Resize([64, 64]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

batch_size = 128
#torchvision.datasets.ImageFolder使用详解： https://blog.csdn.net/taylorman/article/details/118631209
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'valid']}
# 按照batch_size 去 load  数据
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True) for x in ['train', 'valid']}
# 分别计算训练集、验证集数据量

class_names = image_datasets['train'].classes

print(image_datasets)

'''
读取标签对应的实际名字
'''
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
print(cat_to_name)

'''加载models中提供的模型，并且直接用训练的好权重当做初始化参数'''
model_name = 'resnet'  #可选的比较多 ['resnet', 'alexnet', 'vgg', 'squeezenet', 'densenet', 'inception']
#是否用人家训练好的特征来做
feature_extract = True #迁移学习，都用人家特征，先不更新

# 是否用GPU训练
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

'''模型参数要不要更新'''
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():# 遍历模型的参数
            param.requires_grad = False #设置反向传播时不更新参数（迁移学习）
model_ft = models.resnet18()#18层的能快点，条件好点的也可以选152
model_ft

'''把模型输出层改成自己的分类数'''
def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    model_ft = models.resnet18(pretrained=use_pretrained)  # x选用模型，并且采用训练好的参数
    set_parameter_requires_grad(model_ft, feature_extract)

    num_ftrs = model_ft.fc.in_features  # 找到输出层
    model_ft.fc = nn.Linear(num_ftrs, num_classes)  # 重写输出层，类别数自己根据自己任务来（这里102）

    input_size = 64  # 输入大小根据自己配置来

    return model_ft, input_size

'''设置哪些层需要训练'''
model_ft, input_size = initialize_model(model_name, 102, feature_extract, use_pretrained=True)  # 分类数102

# GPU还是CPU计算
model_ft = model_ft.to(device)

#  模型保存，名字自己起
filename = 'best.pt'

# 是否训练所有层
params_to_update = model_ft.parameters()
print("Params to learn:")
if feature_extract:
    params_to_update = []  # 将需要更新的参数放在这个list中，一会交给优化器去更新
    for name, param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)

else:
    for name, param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t", name)

'''优化器设置'''
# 优化器设置
optimizer_ft = optim.Adam(params_to_update, lr=1e-2)#要训练啥参数，你来定
# 定义学习率衰减策略， step_size 个 epoch 进行一次衰减，衰减策略可以自定义
scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)#学习率每7个epoch衰减成原来的1/10
# 损失函数： 交叉熵，传入的是真实值和预测值
criterion = nn.CrossEntropyLoss()

'''训练模块'''


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, filename='best.pt'):
    # 咱们要算时间的
    since = time.time()
    # 也要记录最好的那一次
    best_acc = 0
    # 模型也得放到你的CPU或者GPU
    model.to(device)
    # 训练过程中打印一堆损失和指标
    val_acc_history = []
    train_acc_history = []
    train_losses = []
    valid_losses = []
    # 学习率
    LRs = [optimizer.param_groups[0]['lr']]  # optimizer.param_groups[0] 是一个字典，取出key = 'lr'的数据
    # 最好的那次模型，后续会变的，先初始化
    best_model_wts = copy.deepcopy(model.state_dict())
    # 一个个epoch来遍历
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # 训练和验证
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # 训练
            else:
                model.eval()  # 验证，验证集没有权重的更新

            running_loss = 0.0
            running_corrects = 0

            # 把数据都取个遍
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)  # 放到你的CPU或GPU
                labels = labels.to(device)

                # 梯度清零，每一次梯度更新前，先进行梯度清零
                optimizer.zero_grad()
                # 只有训练的时候计算和更新梯度
                outputs = model(inputs)  # 每个数据的 outputs 就是102*1
                loss = criterion(outputs, labels)  # 损失函数
                _, preds = torch.max(outputs, 1)
                # 训练阶段更新权重
                if phase == 'train':
                    loss.backward()  # 反向传播得到梯度更新值
                    optimizer.step()  # 更新梯度

                # 计算损失
                running_loss += loss.item() * inputs.size(0)  # 0表示batch那个维度
                running_corrects += torch.sum(preds == labels.data)  # 预测结果最大的和真实值是否一致

            # 遍历一个dataLoader，算是跑完一个epoch
            epoch_loss = running_loss / len(dataloaders[phase].dataset)  # 算平均
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            time_elapsed = time.time() - since  # 一个epoch我浪费了多少时间
            print('Time elapsed {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # 得到最好那次的模型
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                state = {
                    'state_dict': model.state_dict(),  # 字典里key就是各层的名字，值就是训练好的权重
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(state, filename)
            if phase == 'valid':
                val_acc_history.append(epoch_acc)
                valid_losses.append(epoch_loss)
                # scheduler.step(epoch_loss)#学习率衰减
            if phase == 'train':
                train_acc_history.append(epoch_acc)
                train_losses.append(epoch_loss)
        # 每个epoch 结束打印学习率，并按照学习率的衰减策略衰减
        print('Optimizer learning rate : {:.7f}'.format(optimizer.param_groups[0]['lr']))
        LRs.append(optimizer.param_groups[0]['lr'])
        print()
        scheduler.step()  # 学习率衰减

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # 训练完后用最好的一次当做模型最终的结果,等着一会测试
    model.load_state_dict(best_model_wts)
    return model, val_acc_history, train_acc_history, valid_losses, train_losses, LRs

'''开始训练,只训练了输出层'''
model_ft, val_acc_history, train_acc_history, valid_losses, train_losses, LRs  = train_model(model_ft, dataloaders, criterion, optimizer_ft, num_epochs=2)

'''再继续训练所有层'''
for param in model_ft.parameters():
    param.requires_grad = True

# 再继续训练所有的参数，学习率调小一点
optimizer = optim.Adam(model_ft.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

# 损失函数
criterion = nn.CrossEntropyLoss()

# 加载之前训练好的权重参数

checkpoint = torch.load(filename)
best_acc = checkpoint['best_acc']
model_ft.load_state_dict(checkpoint['state_dict'])

model_ft, val_acc_history, train_acc_history, valid_losses, train_losses, LRs  = train_model(model_ft, dataloaders, criterion, optimizer, num_epochs=2)

'''加载训练好的模型'''
model_ft, input_size = initialize_model(model_name, 102, feature_extract, use_pretrained=True)

# GPU模式
model_ft = model_ft.to(device)

# 保存文件的名字
filename='best.pt'

# 加载模型
checkpoint = torch.load(filename)
best_acc = checkpoint['best_acc']
model_ft.load_state_dict(checkpoint['state_dict'])

'''测试数据预处理
测试数据处理方法需要跟训练时一直才可以
crop操作的目的是保证输入的大小是一致的
标准化操作也是必须的，用跟训练数据相同的mean和std,但是需要注意一点训练数据是在0-1上进行标准化，所以测试数据也需要先归一化
最后一点，PyTorch中颜色通道是第一个维度，跟很多工具包都不一样，需要转换

'''

# 得到一个batch的测试数据
dataiter = iter(dataloaders['valid'])
images, labels = dataiter.next()

model_ft.eval()

if train_on_gpu:
    output = model_ft(images.cuda())
else:
    output = model_ft(images)

'''得到概率最大的那个'''
_, preds_tensor = torch.max(output, 1)

#numpy.squeeze() 从数组的形状中删除单维度条目，即把shape中为1的维度去掉
preds = np.squeeze(preds_tensor.numpy()) if not train_on_gpu else np.squeeze(preds_tensor.cpu().numpy())
print(preds)

'''展示预测结果，红色是识别错误的，绿色是识别正确的'''


def im_convert(tensor):
    """ 展示数据"""

    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    # Torch 中要求的格式是 channel * H * W, 所以数据如果一开始不是 channel， 先用transpose把channel
    image = image.transpose(1, 2, 0)
    # 之前数据的输入是经过标准化的，所以这里再乘以标准差+均值返回
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)# 防止数值越界、异常

    return image
fig=plt.figure(figsize=(20, 20))
columns =4
rows = 2

for idx in range (columns*rows):
    ax = fig.add_subplot(rows, columns, idx+1, xticks=[], yticks=[])
    plt.imshow(im_convert(images[idx]))
    ax.set_title("{} ({})".format(cat_to_name[str(preds[idx])], cat_to_name[str(labels[idx].item())]),
                 color=("green" if cat_to_name[str(preds[idx])]==cat_to_name[str(labels[idx].item())] else "red"))
plt.show()

# if __name__ == '__main__':























