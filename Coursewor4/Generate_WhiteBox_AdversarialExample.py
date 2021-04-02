import os
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import torchvision.utils
from torchvision import models
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torchattacks
from torchattacks import PGD

from models import Holdout, Target
from utils import imshow
from tqdm import tqdm
from mymodels import *

batch_size = 24

cifar10_train = dsets.CIFAR10(root='./data', train=True,
                              download=True, transform=transforms.ToTensor())
cifar10_test  = dsets.CIFAR10(root='./data', train=False,
                              download=True, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(cifar10_train,
                                           batch_size=1,
                                           shuffle=False)
test_loader = torch.utils.data.DataLoader(cifar10_test,
                                          batch_size=1,
                                          shuffle=False)
#model = ResNet18()
#model = Holdout()
#model = Target()
#model.load_state_dict(torch.load("./model/target.pth"))
#model.load_state_dict(torch.load("./model/GOOGLENET.pth"))['net']
#model.load_state_dict(torch.load("./model/VGG16.pth"))['net']
#model.load_state_dict(torch.load("./model/target.pth"))

model = VGG('VGG16')
checkoutpoint = torch.load("./model/VGG16.pth")['net']
#model = GoogLeNet()
#model = ResNet18()
model = torch.nn.DataParallel(model)
model.load_state_dict(checkoutpoint)

model = model.eval().cuda()

pgd_attack = PGD(model, eps=10/255, alpha=2/255, steps=7)
pgd_attack.set_return_type('int') # Save as integ
#pgd_attack.save(data_loader=test_loader, save_path="./data/cifar10_Resnet18_pgd.pt", verbose=True)
pgd_attack.save(data_loader=test_loader, save_path="./data/cifar10_VGG16_pgd.pt", verbose=True)
#pdb.set_trace()

#adv_images, adv_labels = torch.load("./data/cifar10_Process_pgd.pt")
#adv_data = TensorDataset(adv_images.float()/255, adv_labels)
#adv_loader = DataLoader(adv_data, batch_size=1, shuffle=False)


