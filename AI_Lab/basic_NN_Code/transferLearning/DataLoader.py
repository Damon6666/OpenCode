import os
import matplotlib.pyplot as plt

import numpy as np
import torch
from torch import nn

import torch.optim as optim
import torchvision

from torchvision import transforms, models, datasets

import imageio
import time
import warnings
import random
import sys
import copy
import json
from PIL import Image

'''1：读取txt文件中的路径和标签
第一个小任务，从标注文件中读取数据和标签
至于你准备存成什么格式，都可以的，一会能取出来东西就行
'''


def load_annotations(ann_file):
    data_infos = {}
    with open(ann_file) as f:
        samples = [x.strip().split(' ') for x in f.readlines()]
        for filename, gt_label in samples:
            data_infos[filename] = np.array(gt_label, dtype=np.int64)
    return data_infos


# print(load_annotations('./flower_data/train.txt'))

'''2：分别把数据和标签都存在list里,因为dataloader到时候会在这里取数据'''
img_label = load_annotations('./flower_data/train.txt')

image_name = list(img_label.keys())
label = list(img_label.values())

'''任务3：图像数据路径得完整
因为一会得用这个路径去读数据，所以路径得加上前缀
'''

data_dir = './flower_data/'
train_dir = data_dir + '/train_filelist'
valid_dir = data_dir + '/val_filelist'

image_path = [os.path.join(train_dir, img) for img in image_name]

'''
任务4：制作自定义数据集，继承Dataset
1.注意要使用from torch.utils.data import Dataset, DataLoader
2.类名定义class FlowerDataset(Dataset)，其中FlowerDataset可以改成自己的名字
3.def init(self, root_dir, ann_file, transform=None):咱们要根据自己任务重写
4.def getitem(self, idx):根据自己任务，返回图像数据和标签数据
'''
from torch.utils.data import Dataset, DataLoader


class FlowerDataset(Dataset):
    def __init__(self, root_dir, ann_file, transform=None):
        self.ann_file = ann_file
        self.root_dir = root_dir
        self.img_label = self.load_annotations()
        # 图像名一个list， 标签一个list， 对应的索引要相同
        self.img = [os.path.join(self.root_dir, img) for img in list(self.img_label.keys())]
        self.label = [label for label in list(self.img_label.values())]
        self.transform = transform

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        image = Image.open(self.img[idx])  # 根据路径打开
        label = self.label[idx]
        if self.transform:
            image = self.transform(image)
        label = torch.from_numpy(np.array(label))
        return image, label

    def load_annotations(self):
        data_infos = {}
        with open(self.ann_file) as f:
            samples = [x.strip().split(' ') for x in f.readlines()]
            for filename, gt_label in samples:
                data_infos[filename] = np.array(gt_label, dtype=np.int64)
        return data_infos


'''
任务5：数据预处理(transform)——pipeline
'''
data_transforms = {
    'train':
        transforms.Compose([
            transforms.Resize(64),
            transforms.RandomRotation(45),
            transforms.CenterCrop(64),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.RandomGrayscale(p=0.025),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    'valid':
        transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
}

'''
任务6：根据写好的class FlowerDataset(Dataset):来实例化dataloader
1.构建数据集：分别创建训练和验证用的数据集（如果需要测试集也一样的方法）
2.用Torch给的DataLoader方法来实例化(batch自定，根据显存来选合适的)
3.打印看看数据里面是不是有东西了
'''

train_dataset = FlowerDataset(root_dir=train_dir, ann_file='./flower_data/train.txt',
                              transform=data_transforms['train'])

val_dataset = FlowerDataset(root_dir=valid_dir, ann_file='./flower_data/val.txt',
                            transform=data_transforms['valid'])
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)

print(len(train_dataset))

print(len(val_dataset))

'''
任务7：用之前先试试，整个数据和标签对应下，看看对不对
1.别着急往模型里传
2.用这个方法：iter(train_loader).next()来试试，得到的数据和标签是啥
3.看不出来就把图画出来，标签打印出来，确保自己整的数据集没啥问题
'''
# image,label = iter(train_loader).next()
# sample = image[0].squeeze()
# sample = sample.permute((1,2,0)).numpy()
# sample *= [0.229, 0.224, 0.225]
# sample += [0.485, 0.456, 0.406]
# plt.imshow(sample)
# plt.show()
# print('Label is: {}'.format(label[0].numpy()))
#
# # 再跑一个
# image, label = iter(val_loader).next()
# sample = image[0].squeeze()
# sample = sample.permute((1, 2, 0)).numpy()
# sample *= [0.229, 0.224, 0.225]
# sample += [0.485, 0.456, 0.406]
# plt.imshow(sample)
# plt.show()
# print('Label is: {}'.format(label[0].numpy()))

'''训练'''
dataloaders = {'train': train_loader, 'valid': val_loader}
model_name = 'resnet'
feature_extract = True

train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_ft = models.resnet18()
print(model_ft)

num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, 102))
input_size = 64
print(model_ft)

# 优化器设置
optimizer_ft = optim.Adam(model_ft.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
criterion = nn.CrossEntropyLoss()


def train_model(model, dataloaders, criterion,  # 损失函数
                optimizer, num_epochs=25,
                is_inception=False,
                filename='best.pt'):
    since = time.time()
    best_acc = 0
    model.to(device)

    val_acc_history = []
    train_acc_history = []
    train_losses = []
    valid_losses = []

    LRs = [optimizer.param_groups[0]['lr']]

    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        print('epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            time_elapsed = time.time() - since
            print('Time elapsed {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            print('{} Loss: {:.4f} ACC: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # 得到最好那次的模型
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                state = {
                    'state_dict': model.state_dict(),
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict()
                }
                torch.save(state, filename)
            if phase == 'valid':
                val_acc_history.append(epoch_acc)
                valid_losses.append(epoch_loss)
                scheduler.step(epoch_loss)
            if phase == 'train':
                train_acc_history.append(epoch_acc)
                train_losses.append(epoch_loss)
        print('Optimizer learning rate : {:.7f}'.format(optimizer.param_groups[0]['lr']))
        LRs.append(optimizer.param_groups[0]['lr'])
        print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:.4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model, val_acc_history, train_acc_history, valid_losses, train_losses, LRs


model_ft, val_acc_history, train_acc_history, valid_losses, train_losses, LRs = train_model(model_ft, dataloaders,
                                                                                            criterion, optimizer_ft,
                                                                                            num_epochs=2,
                                                                                            filename='best.pt')
