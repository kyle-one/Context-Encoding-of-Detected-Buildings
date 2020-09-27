import torch,os,torchvision
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
import os
import logging
import time

class config:
    
    DATA_DIR='/Public/YongkunLiu/Datasets'
    #由于是test数据，所以reb，split中的都一样，因此随便取一个即可
    PATH_ISET=DATA_DIR+'/BEAUTY1.0_VOCLikeComb/VOC2007/ImageSets/Split0'
    PATH_IMG=DATA_DIR+'/BEAUTY1.0_VOCLikeComb/VOC2007/JPEGImages'
    IMG_SIZE = 224 # vgg的输入是224的所以需要将图片统一大小
    BATCH_SIZE= 64 #这个批次大小需要占用4.6-5g的显存，如果不够的化可以改下批次，如果内存超过10G可以改为512
    IMG_MEAN = [0.485, 0.456, 0.406]#imagenet的
    IMG_STD = [0.229, 0.224, 0.225]
    LOG_DIR='./log'
    CUDA=torch.cuda.is_available()
    ctx=torch.device("cuda" if CUDA else "cpu")
    labels = ['Residential','Public', 'Industrial', 'Commercial']
    PATH_PARAMS='/Public/YongkunLiu/beauty_cnn_work_dir/pth'
    PATH_PR='/Public/YongkunLiu/beauty_cnn_work_dir/pr'
    if not os.path.exists(PATH_PR):
        os.makedirs(PATH_PR)
    
def get_model_file_name():
    list_t=os.listdir(config.PATH_PARAMS)
    list_t=[b for b in list_t if b[-3:]=='pth']
    list_t.sort()
    return list_t

def get_test_DataSet():
    PATH_ISET=config.DATA_DIR+'/BEAUTY1.0_VOCLikeComb/VOC2007/ImageSets/Split0'
    test_dict_t={}
    test_dict={'id':[],'label':[]}
    list_t1=os.listdir(PATH_ISET)
    for file_list in list_t1:
        if(file_list.split('_')[-1]=='test.txt'and file_list.split('_')[0]=='Districts'):
            with open(PATH_ISET+'/'+file_list) as f:
                t1=f.readlines()
                t1=[t2.split('\n')[0] for t2 in t1]
                test_dict_t[file_list.split('_')[1]]=t1
    for t1 in test_dict_t:
        test_dict['id']+=test_dict_t[t1]
        for t2 in test_dict_t[t1]:
            test_dict['label']+=[t1]
    df = pd.DataFrame(test_dict,columns=['id', 'label']) 
    labels = ['Residential','Public', 'Industrial', 'Commercial']
    label2idx=dict((label,idx) for idx,label in enumerate(labels))
    idx2label=dict((idx,label) for idx,label in enumerate(labels))
    df['label_idx']=[label2idx[x] for x in df.label]  
    return df

def get_test_transform():
    test_transforms = transforms.Compose([
        transforms.Resize(config.IMG_SIZE),
        transforms.CenterCrop(config.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(config.IMG_MEAN, config.IMG_STD)
    ])
    return test_transforms

def get_model(model_name):
    model_ft = torch.load(config.PATH_PARAMS+'/'+model_name)
    for param in model_ft.parameters():
        param.requires_grad = False
    model_ft=model_ft.to(config.ctx)
    return model_ft

class BEAUTYDataset(Dataset):
    def __init__(self,labels_df,transform=None):
        super().__init__()
        self.labels_df = labels_df
        self.transform = transform

    def __len__(self):
        return self.labels_df.shape[0]

    def __getitem__(self, idx):
        image_name = config.PATH_IMG+'/'+self.labels_df.id[idx]+'.jpg'
        img = Image.open(image_name)
        label = self.labels_df.label_idx[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

def test(net,test_iter,ctx):
    list_pred=[]
    for X,y in test_iter:
        y=y.to(ctx)
        X=X.to(ctx)
        y_hat=net(X)
        pred = y_hat.max(1, keepdim=True)[1]
        pred_sum=pred.eq(y.view_as(pred)).sum().item()
        pred=y_hat.max(1, keepdim=True)[1].view(-1).cpu().numpy().tolist()
        list_pred=list_pred+pred
    return list_pred

if __name__=="__main__":
    config=config()
    model_list=get_model_file_name()
    print(config.PATH_PARAMS+'/'+model_list[0].split('.')[0]+'_pr.txt')
    model_list=get_model_file_name()
    test_df=get_test_DataSet()
    test_ds=BEAUTYDataset(test_df,get_test_transform())
    test_iter =DataLoader(test_ds,batch_size=config.BATCH_SIZE,shuffle=False,num_workers=1)
    for model_name in model_list:
        print(model_name)
        model=get_model(model_name)
        #print(model)
        l1=test(model,test_iter,config.ctx)
        with open (config.PATH_PR+'/'+model_name.split('.')[0]+'_pr.txt','w+') as f:
            for a in l1:
                f.write(str(a)+'\n')