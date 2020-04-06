"""

Preliminary tests for OTGAN

"""
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
import os


path = "C:/Users/Philo/Documents/3A -- ENSAE/Optimal transport/OTGAN"
os.chdir(path)


torchvision.datasets.MNIST(root, train=True, transform=None, target_transform=None, download=False)

#### Preliminary step : download dataset

transform = transforms.Compose([transforms.ToTensor(),
#                              transforms.Normalize((0.5,), (0.5,)),
                              transforms.Normalize((0.1307,), (0.3081,)) #use MNIST mean and std
                              ])



trainset = datasets.MNIST('PATH_TO_STORE_TRAINSET', download=True, train=True, transform=transform)
valset = datasets.MNIST('PATH_TO_STORE_TESTSET', download=True, train=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)



def loaders_mnist(dataset, batch_size=64, cuda=0,
                  train_size=50000, val_size=10000, test_size=10000,
                  test_batch_size=1000, **kwargs):

    assert dataset == 'mnist'
    root = '{}/{}'.format(os.environ['VISION_DATA'], dataset)

    # Data loading code
    normalize = transforms.Normalize(mean=(0.1307,),
                                     std=(0.3081,))

    transform = transforms.Compose([transforms.ToTensor(), normalize])

    # define two datasets in order to have different transforms
    # on training and validation
    dataset_train = datasets.MNIST(root=root, train=True, transform=transform)
    dataset_val = datasets.MNIST(root=root, train=True, transform=transform)
    dataset_test = datasets.MNIST(root=root, train=False, transform=transform)

    return create_loaders(dataset_train, dataset_val,
                          dataset_test, train_size, val_size, test_size,
                          batch_size=batch_size,
                          test_batch_size=test_batch_size,
                          cuda=cuda, num_workers=0) 