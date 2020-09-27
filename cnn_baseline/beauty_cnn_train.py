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
import argparse

class config:
    parser = argparse.ArgumentParser()
    parser.add_argument("--SPLIT", help="display a square of a given number", type=int)
    parser.add_argument("--MODEL_NAME", help="display a square of a given number", type=str)
    parser.add_argument("--DATASET_NAME", help="display a square of a given number", type=str)
    args = parser.parse_args()

    SPLIT=args.SPLIT
    MODEL_NAME=args.MODEL_NAME #resnet50/resnet101
    DATASET_NAME=args.DATASET_NAME #BEAUTY/BEAUTY_REB
    
    DATA_DIR='/Public/YongkunLiu/Datasets'
    WORK_DIR='/Public/YongkunLiu/beauty_cnn_work_dir'
    
    LR=0.01
    NUM_EPOCHS=100
    BATCH_SIZE= 128 
    IMG_SIZE = 224 
    CUDA=torch.cuda.is_available()
    ctx=torch.device("cuda" if CUDA else "cpu")
    labels = ['Residential','Public', 'Industrial', 'Commercial']
    
    IMG_MEAN = [0.485, 0.456, 0.406]
    IMG_STD = [0.229, 0.224, 0.225]
    lr_period=10
    lr_decay= 0.1
    if(DATASET_NAME=='BEAUTY'):
        PATH_ISET=DATA_DIR+'/BEAUTY1.0_VOCLikeComb/VOC2007/ImageSets'
        PATH_IMG=DATA_DIR+'/BEAUTY1.0_VOCLikeComb/VOC2007/JPEGImages'
    elif(DATASET_NAME=='BEAUTY_REB'):
        PATH_ISET=DATA_DIR+'/BEAUTY1.0_VOCLikeRebComb/VOC2007/ImageSets'
        PATH_IMG=DATA_DIR+'/BEAUTY1.0_VOCLikeRebComb/VOC2007/JPEGImages'

    PATH_ISET+='/Split'+str(SPLIT)

def get_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(fmt="%(message)s")

    sHandler = logging.StreamHandler()
    sHandler.setFormatter(formatter)
    logger.addHandler(sHandler)

    if not os.path.exists(config.WORK_DIR):
        os.makedirs(config.WORK_DIR)
        
    if not os.path.exists(config.WORK_DIR+'/log'):
        os.makedirs(config.WORK_DIR+'/log')
        
    fHandler = logging.FileHandler(config.WORK_DIR +'/log/'+ config.MODEL_NAME+'_'+config.DATASET_NAME+'_split'+str(config.SPLIT)+'.log', mode='w')
    fHandler.setLevel(logging.DEBUG)
    fHandler.setFormatter(formatter)
    logger.addHandler(fHandler)
    return logger

def get_DataSet(split_name):
    
    label2idx=dict((label,idx) for idx,label in enumerate(config.labels))
    idx2label=dict((idx,label) for idx,label in enumerate(config.labels)) 
    
    train_dict_t={}
    train_dict={'id':[],'label':[]}
    list_t1=os.listdir(config.PATH_ISET)
    for file_list in list_t1:
        if(file_list.find('_train.txt')!=-1):
            with open(config.PATH_ISET+'/'+file_list) as f:
                t1=f.readlines()
                t1=[t2.split('\n')[0] for t2 in t1]
                train_dict_t[file_list.split('_')[1]]=t1
    for t1 in train_dict_t:
        train_dict['id']+=train_dict_t[t1]
        for t2 in train_dict_t[t1]:
            train_dict['label']+=[t1]
    train_df = pd.DataFrame(train_dict,columns=['id', 'label']) 
    train_df['label_idx']=[label2idx[x] for x in train_df.label]  
    
    val_dict_t={}
    val_dict={'id':[],'label':[]}
    list_t1=os.listdir(config.PATH_ISET)
    for file_list in list_t1:
        if(file_list.find('_val.txt')!=-1):
            with open(config.PATH_ISET+'/'+file_list) as f:
                t1=f.readlines()
                t1=[t2.split('\n')[0] for t2 in t1]
                val_dict_t[file_list.split('_')[1]]=t1
    for t1 in val_dict_t:
        val_dict['id']+=val_dict_t[t1]
        for t2 in val_dict_t[t1]:
            val_dict['label']+=[t1]
    val_df = pd.DataFrame(val_dict,columns=['id', 'label']) 
    val_df['label_idx']=[label2idx[x] for x in val_df.label] 
    
    return train_df,val_df

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

        if(self.transform):
            img = self.transform(img)
        return img,label

def get_transform():
    train_transforms = transforms.Compose([
        transforms.Resize(config.IMG_SIZE), 
        transforms.RandomResizedCrop(config.IMG_SIZE), 
        transforms.RandomHorizontalFlip(), 
        transforms.ColorJitter(brightness=(0.6,1.4), contrast=(0.6,1.4), saturation=(0.6,1.4), hue=0),
        transforms.ToTensor(), 
        transforms.Normalize(config.IMG_MEAN, config.IMG_STD)
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(config.IMG_SIZE),
        transforms.CenterCrop(config.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(config.IMG_MEAN, config.IMG_STD)
    ])
    return train_transforms,val_transforms

def get_model(name):
    if(name=='resnet50'):
        model_ft = models.resnet50(pretrained=True) 
        for param in model_ft.parameters():
            param.requires_grad = False
        num_fc_ftr = model_ft.fc.in_features 
        model_ft.fc = nn.Linear(num_fc_ftr, len(config.labels)) 
        model_ft=model_ft.to(config.ctx)
        return model_ft
    elif(name=='resnet101'):
        model_ft = models.resnet101(pretrained=True)
        for param in model_ft.parameters():
            param.requires_grad = False
        num_fc_ftr = model_ft.fc.in_features 
        model_ft.fc = nn.Linear(num_fc_ftr, len(config.labels)) 
        model_ft=model_ft.to(config.ctx)
        return model_ft

def evaluate_loss(data_iter, net, ctx,epoch):
    net.eval()
    l_sum, n,val_pred_sum= 0.0, 0,0
    with torch.no_grad():
        for bidx,(X, y) in enumerate(data_iter):
            y=y.to(config.ctx)
            X=X.to(config.ctx)
            y_hat = net(X)
            loss=criterion(y_hat, y).item()
            l_sum +=loss
            n += y.size()[0]
            pred=y_hat.max(1, keepdim=True)[1]
            pred_sum=pred.eq(y.view_as(pred)).sum().item()
            val_pred_sum+=pred_sum
            logger.info("[epoch {}][{}][batch {}] val_loss={:.5f},val_acc:{:.5f}({}/{})".format\
                        (epoch,'val',bidx,loss,pred_sum/y.size()[0],int(pred_sum),y.size()[0]))
    logger.info("[epoch {}][{}][end] val_loss={:.5f},val_acc:{:.5f}({}/{})".format\
    (epoch,'val',l_sum/(bidx+1),val_pred_sum/n,int(val_pred_sum),n))
        
    return l_sum / len(data_iter),pred_sum/n

def train(net, train_iter, valid_iter, num_epochs, lr,  ctx, lr_period, lr_decay):
    Max_Acc=0.0
    optimizer = torch.optim.Adam([{'params':net.fc.parameters()}], lr=lr)
    
    if not os.path.exists(config.WORK_DIR+'/pth'):
        os.makedirs(config.WORK_DIR+'/pth')
    
    for epoch in range(num_epochs):
        net.train()
        train_l_sum, n, start ,train_acc_sum,train_pred_sum,correct= 0.0, 0, time.time(),0,0,0
        if epoch > 0 and epoch % lr_period == 0:
            optimizer.param_groups[0]['lr']=optimizer.param_groups[0]['lr']*0.5
        for bidx, (X,y) in enumerate(train_iter):
            X=X.to(config.ctx)
            y=y.to(config.ctx)
            optimizer.zero_grad()
            y_hat= net(X)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            
            n+=y.size()[0]
            train_l_sum += loss.item()
            pred = y_hat.max(1, keepdim=True)[1]
            pred_sum=pred.eq(y.view_as(pred)).sum().item()
            train_pred_sum += pred_sum
            logger.info("[epoch {}][{}][batch {}] train_loss={:.5f},train_acc={:.5f}({}/{})".format\
                    (epoch,'train',bidx,loss,pred_sum/y.size()[0],int(pred_sum),y.size()[0]))
        logger.info("[epoch {}][{}][end] train_loss={:.5f},train_acc={:.5f}({}/{})".format\
                    (epoch,'train',train_l_sum/(bidx+1),train_pred_sum/n,int(train_pred_sum),n))

        if valid_iter is not None:
            valid_loss,valid_acc = evaluate_loss(valid_iter, net, ctx,epoch)
            
        time_s = "time %.2f sec" % (time.time() - start)
        print(time_s)
        
        if not os.path.exists(config.WORK_DIR):
            os.makedirs(config.WORK_DIR)
        if(valid_acc>Max_Acc):
            Max_Acc=valid_acc
            torch.save(net,'{}/{}_{}_split{}_best.pth'.format(config.WORK_DIR+'/pth',config.MODEL_NAME,config.DATASET_NAME,config.SPLIT))
            logger.info("[epoch {}][save_best_output_params]".format(epoch))

if __name__ == '__main__':
    config = config()
    logger=get_logger()
    logger.info("[message][time:{}](BATCH_SIZE:{},NUM_EPOCHS:{},LR:{},DATASET_NAME:{},MODEL_NAME:{})".format\
            (time.asctime(time.localtime(time.time()+60*60*8)),config.BATCH_SIZE,config.NUM_EPOCHS,config.LR,config.DATASET_NAME,config.MODEL_NAME))
    train_transforms,val_transforms=get_transform()
    train_df,val_df=get_DataSet(config.SPLIT)
    train_ds=BEAUTYDataset(train_df,train_transforms)
    valid_ds=BEAUTYDataset(val_df,val_transforms)
    train_iter =DataLoader(train_ds,batch_size=config.BATCH_SIZE,shuffle=True,num_workers=5)
    valid_iter =DataLoader(valid_ds,batch_size=config.BATCH_SIZE,shuffle=True,num_workers=5)

    criterion = nn.CrossEntropyLoss()

    net =get_model(config.MODEL_NAME)
    train(net, train_iter, valid_iter, config.NUM_EPOCHS, config.LR,  config.ctx, config.lr_period,config.lr_decay)

