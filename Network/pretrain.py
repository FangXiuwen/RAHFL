import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import sys
sys.path.append("..")
from Dataset.utils import init_logs, get_pretrain_dataloader, init_nets, mkdirs, partition_data
import torch.optim as optim
import torch.nn as nn
import numpy as np
from numpy import *
import random
import torch
import torch.backends.cudnn
import os

#20clients
'''
Global Parameters
'''
Seed = 0
TrainBatchSize = 256
TestBatchSize = 512
Pretrain_Epoch = 40
Pariticpant_Params = {
    'loss_funnction' : 'CE',
    'optimizer_name' : 'Adam',
    'learning_rate'  : 0.001
}
AugMix_Module = None #['jsd', 'nojsd', None]
"""Corruption Setting"""
Private_Corrupt_Rate = 1 #[0, 1, 0.5]
Test_Corrupt_Rate = 1 #[0, 1]
"""Model Setting"""
Model_setting = 'hetero' #['hetero', 'homo']
Nets_Name_List = ['ResNet10','ResNet12','ShuffleNet','Mobilenetv2']
# Nets_Name_List = ['ResNet12','ResNet12','ResNet12','ResNet12']
N_Participants = len(Nets_Name_List)
"""Dataset Setting"""
Private_Dataset_Name = 'cifar10' #['cifar10']
Private_Dataset_Dir = '../Dataset/cifar_10_c'
Private_Data_Len = 10000
Private_Output_Channel = 10
Data_Partition = 'iid' #['iid', 'noniid']
Noniid_Dirichlet_Beta = 0 #iid:0 ; noniid:1.0
"""Model Save Setting"""
Model_Save_Dir = '../Network/Model_Storage_' + Model_setting + '_4client/' + Private_Dataset_Name + '_' + Data_Partition + '_' + str(Noniid_Dirichlet_Beta) + '/random_corrupt_' + str(Private_Corrupt_Rate)

def pretrain_network(network,localtrain_epoch,private_dataloader,loss_function,optimizer_method,learning_rate,logger):
    if loss_function =='CE':
        criterion = nn.CrossEntropyLoss()
    criterion.to(device)
    if optimizer_method =='Adam':
        optimizer = optim.Adam(network.parameters(),lr=learning_rate)
    if optimizer_method =='SGD':
        optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    for _ in range(localtrain_epoch):
        for batch_idx, (images, labels) in enumerate(private_dataloader):
            images = images.to(device)
            labels = labels.to(device)
            outputs, _ = network(images)
            labels = labels.long()
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logger.info('Private Train : [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(batch_idx * len(images), len(private_dataloader.dataset), 
                                                                                 100. * batch_idx / len(private_dataloader), loss.item()))
    return network

def evaluate_network(network,dataloader,logger):
    network.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs,_ = network(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        acc = 100 * correct / total
        logger.info('Test Accuracy of the model on the test images: {} %'.format(acc))
    return acc

if __name__ =='__main__':
    mkdirs(Model_Save_Dir)
    logger = init_logs()
    logger.info("Random Seed and Server Config")
    seed = Seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device_ids = [0,1,2,3]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    logger.info("Initialize Participants' Data idxs")
    if Data_Partition == 'noniid':
        net_datanum_map = {}
        _, net_dataidx_map = partition_data(dataset=Private_Dataset_Name,datadir=Private_Dataset_Dir,partition=Data_Partition,
                                        num_classes=Private_Output_Channel,num_users=N_Participants,corrupt_rate=Private_Corrupt_Rate,
                                        test_corrupt_rate=Test_Corrupt_Rate,dirichlet_beta=Noniid_Dirichlet_Beta)
    elif Data_Partition == 'iid':
        net_dataidx_map = {}
        for index in range(N_Participants):
            idxes = np.random.permutation(50000)
            idxes = idxes[0:Private_Data_Len]
            net_dataidx_map[index]= idxes
    logger.info(net_dataidx_map)

    net_list = init_nets(n_parties=N_Participants,nets_name_list=Nets_Name_List,num_classes=Private_Output_Channel)

    logger.info('Pretrain Participants Models')
    for index in range(N_Participants):
        train_dl, _, train_ds, _= get_pretrain_dataloader(dataset=Private_Dataset_Name,datadir=Private_Dataset_Dir,train_bs=TrainBatchSize,test_bs=TestBatchSize,
                                                 dataidxs=net_dataidx_map[index],corrupt_rate=Private_Corrupt_Rate,augmix_module=AugMix_Module)
        network = net_list[index]
        network = nn.DataParallel(network, device_ids=device_ids).to(device)
        netname = Nets_Name_List[index]
        logger.info('Pretrain the '+str(index)+' th Participant Model with N_training: '+str(len(train_ds)))
        network = pretrain_network(network=network,localtrain_epoch=Pretrain_Epoch,private_dataloader=train_dl,loss_function=Pariticpant_Params['loss_funnction'],
                                   optimizer_method=Pariticpant_Params['optimizer_name'],learning_rate=Pariticpant_Params['learning_rate'],logger=logger)
        logger.info('Save the '+str(index)+' th Participant Model')
        torch.save(network.state_dict(), Model_Save_Dir + '/' +netname+'_'+str(index)+'.ckpt')

    logger.info('Evaluate Models')
    test_accuracy_list = []
    for index in range(N_Participants):
        _, test_dl, _, _= get_pretrain_dataloader(dataset=Private_Dataset_Name,datadir=Private_Dataset_Dir,train_bs=TrainBatchSize,test_bs=TestBatchSize,
                                         test_corrupt_rate=Test_Corrupt_Rate)
        network = net_list[index]
        network = nn.DataParallel(network, device_ids=device_ids).to(device)
        netname = Nets_Name_List[index]
        network.load_state_dict(torch.load(Model_Save_Dir + '/' + netname+'_'+str(index)+'.ckpt'))
        output = evaluate_network(network=network,dataloader=test_dl,logger=logger)
        test_accuracy_list.append(output)
    print('The average Accuracy of models on the test images:'+str(mean(test_accuracy_list)))
