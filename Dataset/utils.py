import os
import sys
import torch
import pandas as pd
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch.utils.data as data
import torch.nn.functional as F
from Dataset.init_dataset import Cifar10FL,Cifar100FL,CIFAR_C,CIFAR10_C_origin, CIFAR100_RandomC
from torch.autograd import Variable
from Network.Models_Def.resnet import ResNet10,ResNet12
from Network.Models_Def.shufflenet import ShuffleNetG2
from Network.Models_Def.mobilnet_v2 import MobileNetV2
import torchvision.transforms as transforms
import random
from Dataset.dataaug import AugMixDataset
from torchvision import datasets
from Dataset.sampling import iid_sampling, non_iid_dirichlet_sampling
from PIL import ImageFilter

Seed = 0
seed = Seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
Project_Path = r'/home/fangxiuwen/'

class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform, transform_weak):
        self.transform = transform
        self.transform_weak = transform_weak
    def __call__(self, x):
        return [self.transform(x), self.transform_weak(x)]
    
class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""
    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma
    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

def init_logs(log_level=logging.INFO,log_path = Project_Path+'Logs/',sub_name=None):
    # logging：https://www.cnblogs.com/CJOKER/p/8295272.html
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level) 
    log_path = log_path
    mkdirs(log_path)
    filename = os.path.basename(sys.argv[0][0:-3])
    if sub_name == None:
        log_name = log_path + filename + '.log'
    else:
        log_name = log_path + filename + '_' + sub_name +'.log'
    logfile = log_name
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(log_level) 
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    console  = logging.StreamHandler()
    console.setLevel(log_level)
    console.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(console)
    return logger

def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass

def load_cifar10c_data(datadir, corrupt_rate, test_corrupt_rate):
    transform = transforms.Compose([transforms.ToTensor()])
    cifar10_train_ds = CIFAR_C(datadir, train=True, transform=transform, corrupt_rate=corrupt_rate)
    cifar10_test_ds = CIFAR_C(datadir, train=False, transform=transform, corrupt_rate=test_corrupt_rate)
    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target
    return (X_train, y_train, X_test, y_test)

def load_cifar10_data(datadir, noise_type=None, noise_rate=0):
    transform = transforms.Compose([transforms.ToTensor()])
    cifar10_train_ds = Cifar10FL(datadir, train=True, download=True, transform=transform, noise_type=noise_type, noise_rate=noise_rate)
    cifar10_test_ds = Cifar10FL(datadir, train=False, download=True, transform=transform)
    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target
    return (X_train, y_train, X_test, y_test)

def load_cifar100_data(datadir, noise_type=None, noise_rate=0):
    transform = transforms.Compose([transforms.ToTensor()])
    cifar100_train_ds = Cifar100FL(datadir, train=True, download=True, transform=transform, noise_type=noise_type, noise_rate=noise_rate)
    cifar100_test_ds = Cifar100FL(datadir, train=False, download=True, transform=transform)
    X_train, y_train = cifar100_train_ds.data, cifar100_test_ds.target
    X_test, y_test = cifar100_train_ds.data, cifar100_test_ds.target
    return (X_train, y_train, X_test, y_test)

def partition_data(dataset, datadir, partition, num_classes, num_users, corrupt_rate, test_corrupt_rate, dirichlet_beta=1.0, 
                   noise_type=None, noise_rate=0):
    if dataset == 'cifar10':
        X_train, y_train, X_test, y_test = load_cifar10c_data(datadir, corrupt_rate=corrupt_rate, test_corrupt_rate=test_corrupt_rate)
    elif dataset == 'cifar100':
        X_train, y_train, X_test, y_test = load_cifar100_data(datadir, noise_type=noise_type, noise_rate=noise_rate)
    if partition == 'iid':
        net_dataidx_map = iid_sampling(y_train, num_users)
    elif partition == 'noniid':
        net_dataidx_map = non_iid_dirichlet_sampling(y_train, num_classes, num_users, dirichlet_beta=dirichlet_beta)
    return y_train, net_dataidx_map

def generate_public_data_indexs(dataset,datadir,size):
    if dataset =='cifar100':
        X_train, y_train, X_test, y_test = load_cifar100_data(datadir)
    n_train = y_train.shape[0]
    all_idxs = [i for i in range(n_train)] # initial user and index for whole dataset
    idxs = np.random.choice(all_idxs, size, replace=False) # 'replace=False' make sure that there is no repeat
    return idxs

def get_pretrain_dataloader(dataset, datadir, train_bs, test_bs, dataidxs=None, corrupt_rate=0, test_corrupt_rate=0):
    # For the train dataset:
    if dataset == 'cifar10':
        if corrupt_rate == 0:
            normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                             std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        else:
            normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                             std=[0.2023, 0.1944, 0.2010])
        
        if corrupt_rate == 0:
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: F.pad(
                    Variable(x.unsqueeze(0), requires_grad=False),
                    (4, 4, 4, 4), mode='reflect').data.squeeze()),
                transforms.ToPILImage(),
                transforms.ColorJitter(),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
        else:
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                normalize
            ])
        
        train_ds = CIFAR_C(datadir, dataidxs=dataidxs, train=True, transform=transform_train, corrupt_rate=corrupt_rate)
        
        if test_corrupt_rate == 0:
            normalize_test = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                                  std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        else:
            normalize_test = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                                  std=[0.2023, 0.1944, 0.2010])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize_test
        ])
        test_ds = CIFAR_C(datadir, train=False, transform=transform_test, corrupt_rate=test_corrupt_rate)

    if dataset == 'cifar100':
        normalize = transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                         std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
        if corrupt_rate == 0:
            transform_train = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                normalize
            ])
        else:
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                normalize
            ])
        
        train_ds = CIFAR_C(datadir, dataidxs=dataidxs, train=True, transform=transform_train, corrupt_rate=corrupt_rate)
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
        test_ds = CIFAR_C(datadir, train=False, transform=transform_test, corrupt_rate=test_corrupt_rate)

    if dataset == 'tinyimagenet':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        # 根据 corrupt_rate 判断是否使用数据增强
        if corrupt_rate == 0:
            # 干净数据：使用完整数据增强
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                normalize
            ])
        else:
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                normalize
            ])
        if dataidxs is None:
            train_ds = datasets.ImageFolder(datadir+'/train', transform=transform_train)
        else:
            train_ds = datasets.ImageFolder(datadir+'/train', transform=transform_train)
            train_ds = torch.utils.data.Subset(train_ds, dataidxs)
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
        test_ds = datasets.ImageFolder(datadir+'/test', transform=transform_test)

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, drop_last=True, shuffle=True)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, drop_last=True, shuffle=False)

    return train_dl, test_dl, train_ds, test_ds


#-------Origin dataloader--------
def get_dataloader(dataset, datadir, train_bs, test_bs, dataidxs=None, corrupt_rate=0, test_corrupt_rate=0, augmix_module=None):
#-------Origin dataloader--------
    if dataset == 'cifar10':
        if augmix_module == None:
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: F.pad(
                    Variable(x.unsqueeze(0), requires_grad=False),
                    (4, 4, 4, 4), mode='reflect').data.squeeze()),
                transforms.ToPILImage(),
                transforms.ColorJitter(),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]], std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
            ])
            train_ds = CIFAR_C(datadir, dataidxs=dataidxs, train=True, transform=transform_train, corrupt_rate=corrupt_rate)
        else:
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: F.pad(
                    Variable(x.unsqueeze(0), requires_grad=False),
                    (4, 4, 4, 4), mode='reflect').data.squeeze()),
                transforms.ToPILImage(),
                transforms.ColorJitter(),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
            ])
            transform_train_weak = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: F.pad(
                    Variable(x.unsqueeze(0), requires_grad=False),
                    (4, 4, 4, 4), mode='reflect').data.squeeze()),
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                transforms.RandomHorizontalFlip(),
            ])
            preprocess = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]], std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
            ])
            # train_ds = CIFAR_C(datadir, dataidxs=dataidxs, train=True, transform=TwoCropTransform(transform_train), corrupt_rate=corrupt_rate)
            train_ds = CIFAR_C(datadir, dataidxs=dataidxs, train=True, transform=TwoCropTransform(transform_train, transform_train_weak), corrupt_rate=corrupt_rate)
            train_ds = AugMixDataset(train_ds, preprocess, jsd_or_nojsd=augmix_module)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]], std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        ])
        test_ds = CIFAR_C(datadir, train=False, transform=transform_test, corrupt_rate=test_corrupt_rate)
    if dataset =='cifar100':
        if augmix_module == None:
            transform_train = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343], 
                                    std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
            ])
            train_ds = CIFAR_C(datadir, dataidxs=dataidxs, train=True, transform=transform_train, corrupt_rate=corrupt_rate)
        else:
            transform_train = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15)
            ])
            preprocess = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343], 
                                    std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
            ])
            train_ds = CIFAR_C(datadir, dataidxs=dataidxs, train=True, transform=transform_train, corrupt_rate=corrupt_rate)
            train_ds = AugMixDataset(train_ds, preprocess, jsd_or_nojsd=augmix_module)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343], 
                                std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
        ])
        test_ds = CIFAR_C(datadir, train=False, transform=transform_test, corrupt_rate=test_corrupt_rate)
    if dataset == 'tinyimagenet':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        if dataidxs is None:
            train_ds = datasets.ImageFolder(datadir+'/train', transform=transform_train)
        else:
            train_ds = datasets.ImageFolder(datadir+'/train', transform=transform_train)
            train_ds = torch.utils.data.Subset(train_ds, dataidxs)
    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, drop_last=True, shuffle=True,num_workers=4)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, drop_last=True, shuffle=False)
    return train_dl, test_dl, train_ds, test_ds

def init_nets(n_parties,nets_name_list, num_classes=10):
    nets_list = {net_i: None for net_i in range(n_parties)}
    for net_i in range(n_parties):
        net_name = nets_name_list[net_i]
        if net_name=='ResNet10':
            net = ResNet10(num_classes=num_classes)
        elif net_name =='ResNet12':
            net = ResNet12(num_classes=num_classes)
        elif net_name =='ShuffleNet':
            net = ShuffleNetG2(num_classes=num_classes)
        elif net_name =='Mobilenetv2':
            net = MobileNetV2(num_classes=num_classes)
        nets_list[net_i] = net
    return nets_list

if __name__ =='__main__':
    logger = init_logs()
    public_data_indexs = generate_public_data_indexs(dataset='cifar100',datadir='./cifar_100',size=5000)
    train_dl, test_dl, train_ds, test_ds = get_dataloader(dataset='cifar100',datadir='./cifar_100',train_bs=256,test_bs=512,dataidxs=public_data_indexs)
    print(len(train_ds))
    # loss_list = [[1,2,3,4,5],[2,3,4,5,6]]
    # loss_name = 'test'
    # draw_epoch_loss(loss_list,loss_name,savepath='./sda.png')