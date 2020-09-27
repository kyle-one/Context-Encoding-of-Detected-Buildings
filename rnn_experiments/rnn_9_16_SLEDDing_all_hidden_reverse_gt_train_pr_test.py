# -*- coding: utf-8 -*-
import torch
import os
import pickle
import pandas as pd
import json
import logging
import time
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms
from torch import nn
import numpy as np
#import tensorwatch as tw
from torch.nn import functional as F

class config:
    #PATH_PKL="/Public/YongkunLiu/workdir_coco_reb_new/faster_rcnn_r101_fpn_2x_coco/results_train.pkl"
    PATH_GT_TRAIN="/Public/YongkunLiu/Datasets/BEAUTY1.0_COCOLike_8_16/Split0/annotations/train_reb.json"
    
    #PATH_PKL_TEST="//Public/YongkunLiu/workdir_coco_reb_new/faster_rcnn_r101_fpn_2x_coco/results_test.pkl"
    PATH_GT_TEST="/Public/YongkunLiu/Datasets/BEAUTY1.0_COCOLike_8_16/Split0/annotations/test.json"
    
    #PATH_PKL_VAL="Public/YongkunLiu/workdir_coco_reb_new/faster_rcnn_r101_fpn_2x_coco/results_val.pkl"
    PATH_GT_VAL="/Public/YongkunLiu/Datasets/BEAUTY1.0_COCOLike_8_16/Split0/annotations/val.json"
    
    PATH_PR={'faster_rcnn_r50':'/Public/YongkunLiu/workdir_coco_9_2/faster_rcnn_r50_fpn_2x_coco/',
          'cascade_rcnn_r101':'/Public/YongkunLiu/workdir_coco_9_2/cascade_rcnn_r101_fpn_20e_coco/',
          'faster_rcnn_r101':'/Public/YongkunLiu/workdir_coco_9_2/faster_rcnn_r101_fpn_2x_coco/',
          'cascade_rcnn_r50':'/Public/YongkunLiu/workdir_coco_9_2/cascade_rcnn_r50_fpn_20e_coco/'}
    
    
    
    #PATH_PKL_TEST="/Public/YongkunLiu/workdir_coco_reb_new/faster_rcnn_r101_fpn_2x_coco/results_test.pkl"
    #PATH_GT_TEST="/Public/YongkunLiu/Datasets/BEAUTY1.0_COCOLikeReb/Split0/annotations/test.json"
    
    
    
    PATH_CLASSIFICATION='/Public/YongkunLiu/Datasets/BEAUTY1.0_COCOLike/classification.json'
    labels=['Commercial','Residential','Industrial','Public']
    bboxs=['retail','house','roof','officebuilding','apartment','garage','industrial','church']
    labels2idx={label:idx for idx,label in enumerate(labels)}
    bboxs2idx={bbox:idx for idx,bbox in enumerate(bboxs)}
    
    WORK_DIR='/Public/YongkunLiu/beauty_rnn_work_dir_9_17'
    BATCH_SIZE=64
    #(8,128,4,1,False,'LSTM')
    #(self, input_size, hidden_size, output_size,num_layers,bidirectional,rnn_type):#输入，隐藏层单元，输出，隐藏层数，双向
    HIDDEN_SIZE=128
    NUM_LAYERS=2
    BIDIRECTIONAL='False'
    RNN_TYPE='LSTM'#/GRU
    NUM_EPOCHS=10
    LR=0.01
    lr_period, lr_decay= 10, 0.5
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    matrix=[[0.7, 0.01, 0.05, 0.07, 0.01, 0.02, 0.12, 0.02],
[0.01, 0.81, 0.01, 0.02, 0.03, 0.09, 0.01, 0.03],
[0.09, 0.01, 0.88, 0.01, 0.0, 0.0, 0.01, 0.0],
[0.09, 0.03, 0.0, 0.67, 0.01, 0.12, 0.06, 0.02],
[0.03, 0.1, 0.01, 0.02, 0.83, 0.0, 0.02, 0.0],
[0.02, 0.12, 0.0, 0.16, 0.0, 0.62, 0.02, 0.04],
[0.1, 0.0, 0.0, 0.04, 0.01, 0.01, 0.82, 0.03],
[0.04, 0.04, 0.0, 0.07, 0.0, 0.02, 0.04, 0.8]]
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0"


#计算4个分类下，bbox种类分布
def bbox_distribution(PATH_GT):
    with open(PATH_GT) as f:
        json_data=json.load(f)
    #json_data

    bboxs=[0.0 for i in range(8)]
    #classfication={i:bboxs.copy() for i in config.labels}
    classfication=[bboxs.copy() for i in range(4)]

    for bbox_idx in range(len(json_data['annotations'])):
        image_idx=json_data['annotations'][bbox_idx]['image_id']-1
        image_cate=json_data['images'][image_idx]['file_name'].split('_')[0]
        image_cate=config.labels2idx[image_cate]
        bbox_cate=json_data['annotations'][bbox_idx]['category_id']-1
        classfication[image_cate][bbox_cate]+=1

    for i in range(len(classfication)):
        sum_t=0
        for j in range(len(classfication[i])):
            sum_t+=classfication[i][j]
        for j in range(len(classfication[i])):
            classfication[i][j]/=sum_t

    return classfication
#distribution=bbox_distribution(config.PATH_GT)
#distribution
#bbox_distribution(config.PATH_GT_TEST)

def reconfidence(pr_list,bbox_distribution,classfication_pr):
    pr_list=pr_list.copy()
    pr_idx=0
    for image_idx in range(len(pr_list)):
        for bbox_idx in range(len(pr_list[image_idx])):
            #print(pr_list[image_idx][bbox_idx]['score'])
            s=pr_list[image_idx][bbox_idx]['score']
            #p=bbox_distribution[pr_list[image_idx][bbox_idx]['image_class']][pr_list[image_idx][bbox_idx]['cate']]
            #print(image_idx)
            t=classfication_pr[image_idx]
            p=0
            for t_idx in range(len(t)):
                p+=t[t_idx]*bbox_distribution[t_idx][pr_list[image_idx][bbox_idx]['cate']]
            pr_list[image_idx][bbox_idx]['score']=s*s+(1-s)*p
            pr_idx+=1
    return pr_list

def add_gt_bbox(pr_list,gt_json):#将pr_list中加入gt_bbox标签
    for i_idx in range(len(pr_list)):
        for b_idx in range(len(pr_list[i_idx])):
            pr_list[i_idx][b_idx]['gt_cate']=-1

    for gt_bbox_idx in range(len(gt_json['annotations'])):
        gt_bbox=gt_json['annotations'][gt_bbox_idx]['bbox']
        gt_bbox[3]+=gt_bbox[1]
        gt_bbox[2]+=gt_bbox[0]
        gt_bbox_cate=gt_json['annotations'][gt_bbox_idx]['category_id']-1
        image_idx=gt_json['annotations'][gt_bbox_idx]['image_id']-1
        #print(image_idx)
        for pr_bbox_idx in range(len(pr_list[image_idx])):
            if (pr_list[image_idx][pr_bbox_idx]==[]):
                continue
            pr_bbox=pr_list[image_idx][pr_bbox_idx]['bbox']
            iou_=iou(gt_bbox,pr_bbox)
            if(iou_>0.5):
                pr_list[image_idx][pr_bbox_idx]['gt_cate']=gt_bbox_cate
        for pr_bbox_idx in range(len(pr_list[image_idx])):
            if (pr_list[image_idx][pr_bbox_idx]==[]):
                continue
            if (pr_list[image_idx][pr_bbox_idx]['gt_cate']!=-1):
                continue
            pr_bbox=pr_list[image_idx][pr_bbox_idx]['bbox']
            iou_=SpaceSort.iou_small(gt_bbox,pr_bbox)
            if(iou_>0.6):
                pr_list[image_idx][pr_bbox_idx]['gt_cate']=gt_bbox_cate
      
    return pr_list

def get_pr_list(path_pkl,path_gt,confidence_thr=0.0,merge_iou=False,merge_iou2=False,bbox_num_thr=0):
    #merge_iou，同一位置只保留score最大的,merge_iou2：相同位置的归类，但是保留
    #bbox_num_thr等于2，则把一个bbox和两个bbox的image全pass了。以此类推
    with open(path_pkl,'rb') as f:
        pkl_data = pickle.load(f)
    with open(path_gt) as f:
        json_data=json.load(f)
    with open(config.PATH_CLASSIFICATION) as f:
        class_data=json.load(f)
        
    #list一个元素是一张图片
    #一张图片有多个bbox列表
    #一个bbox列表元素有 图片名id  bboxid  bbox种类  图片名字
    #将pkl预测文件转化为list
    #reb变化有三种：_HorizontalFlip _hue 和 
    pr_list=[]
    with open(config.PATH_CLASSIFICATION) as f:
        class_data=json.load(f)
    for image_idx,image in enumerate(pkl_data):
        image_list=[]
        bbox_class='None'
        if('_hue' in json_data['images'][image_idx]['file_name']):
            image_name_t=json_data['images'][image_idx]['file_name'].split('_hue.jpg')[0]
        elif('_HorizontalFlip' in json_data['images'][image_idx]['file_name']):
            image_name_t=json_data['images'][image_idx]['file_name'].split('_HorizontalFlip.jpg')[0]
        else:
            image_name_t=json_data['images'][image_idx]['file_name'].split('.jpg')[0]
        #for class_name in class_data:#查找分类文件的四个种类dict
        #    if(image_name_t in class_data[class_name]):
        #        bbox_class=class_name
        
        for label in config.labels:
            if(label in image_name_t):
                bbox_class=label
        
        for cate_idx,cate in enumerate(image):
            if(cate.size!=0):
                for bbox in cate:
                    if(bbox[4]<confidence_thr):
                        continue
                    bbox_dict={}
                    bbox_dict['score']=bbox[4]
                    bbox_dict['cate']=cate_idx
                    bbox_dict['bbox']=bbox[:4]
                    bbox_dict['image_name']=json_data['images'][image_idx]['file_name']
                    bbox_dict['image_class']=bbox_class
                    bbox_dict['image_class_idx']=config.labels2idx[bbox_class]
                    image_list.append(bbox_dict)
        pr_list.append(image_list)
    #print(len(pr_list))
    add_gt_bbox(pr_list,json_data)
    
    '''删除检测为空的图
    for idx,i in enumerate(pr_list):
        if(i==[]):
            del(pr_list[idx])
    '''
    
    if(merge_iou==True):
        for idx in range(len(pr_list)):
            if(pr_list[idx]==[]):
                continue
            t=[]
            for i in pr_list[idx]:
                if(len(t)==0):#初始，直接往t里面添加一个数组，数组中有i
                    t.append([i])
                    #print('12321321')
                else:
                    flag=0
                    for j in range(len(t)):#遍历数组t
                        if(flag==1):
                            break
                        for k in range(len(t[j])):#遍历数组t[j]
                            if(iou(i['bbox'],t[j][k]['bbox'])>0.6):#如果大于iou0.6，则添加到数组t[j]
                                #print(iou(i['bbox'],t[j][k]['bbox']))
                                t[j].append(i)
                                flag=1
                                break
                    if(flag==0):#遍历完数组t，若还是没有匹配的，则新添加到t
                        t.append([i])
            #print('\n')
            max_bbox=[]
            for i in t:
                max_b=i[0]
                for j in i:
                    #print(j['score']>max_score)
                    if(j['score']>max_b['score']):
                        max_b=j
                max_bbox.append(max_b)
            pr_list[idx]=max_bbox
    if(merge_iou2==True):
        for idx in range(len(pr_list)):
            if(pr_list[idx]==[]):
                continue
        for idx in range(len(pr_list)):
            t=[]
            for i in pr_list[idx]:
                if(len(t)==0):#初始，直接往t里面添加一个数组，数组中有i
                    t.append([i])
                    #print('12321321')
                else:
                    flag=0
                    for j in range(len(t)):#遍历数组t
                        if(flag==1):
                            break
                        for k in range(len(t[j])):#遍历数组t[j]
                            if(iou(i['bbox'],t[j][k]['bbox'])>0.6):#如果大于iou0.6，则添加到数组t[j]
                                #print(iou(i['bbox'],t[j][k]['bbox']))
                                t[j].append(i)
                                flag=1
                                break
                    if(flag==0):#遍历完数组t，若还是没有匹配的，则新添加到t
                        t.append([i])
            #print('\n')
            pr_list[idx]=t
    
    if(bbox_num_thr>0):
        i=0
        while(i<len(pr_list)):
            #print(i)
            if(len(pr_list[i])<=bbox_num_thr):
                del pr_list[i]
            i+=1
        
        
            
            
    return pr_list

def Bbox_labels(image,max_l):
    bbox_labels=[]
    for i in image:
        if (i==[]):
            bbox_labels.append(0.0)
        else:
            if (i['cate']==i['gt_cate']):
                bbox_labels.append(1.0)
            else:
                bbox_labels.append(0.0)
    if (len(bbox_labels)>max_l):
        bbox_labels=bbox_labels[:max_l]
    else:
        for i in range(max_l-len(bbox_labels)):
            bbox_labels.append(0.0)
    return bbox_labels
#fd=Bbox_labels(test_pr_list[59],10)
#fd

def iou(gt_bbox,pr_bbox):
    gt=gt_bbox.copy()
    pr=pr_bbox.copy()
    lou_w=min(gt[0],gt[2],pr[0],pr[2])+(gt[2]-gt[0])+(pr[2]-pr[0])-max(gt[0],gt[2],pr[0],pr[2])
    lou_h=min(gt[1],gt[3],pr[1],pr[3])+(gt[3]-gt[1])+(pr[3]-pr[1])-max(gt[1],gt[3],pr[1],pr[3])
    if(lou_w<0 or lou_h<0):
        return(-1)
    else:
        return(lou_w*lou_h/((gt[2]-gt[0])*(gt[3]-gt[1])+(pr[2]-pr[0])*(pr[3]-pr[1])-lou_w*lou_h))

#输入分数，种类 和 长度 转onehot编码
def OneHot(score,cate,size):#分数，种类，长度
    t=[0.0 for i in range(size)]
    t[cate]=score
    return t

def OneHotMatrix(score,cate,size):
    t1=config.matrix[cate]
    t2=[0.0 for i in range(size)]
    t_sum_7=1-t1[cate]
    for i in range(size):
        t2[i]=(1-score)*t1[i]/t_sum_7
    t2[cate]=score
    return t2
def OneHotAdd(score,cate,list_):#往list_t中添加onehot编码
    list_[cate]=score
    return list_

def OneHot2(score,cate,size):#分数，种类，长度
    t=[0.0 for i in range(size)]
    t[cate]=1
    return t

def get_gt_list(path_gt):
    with open(path_gt) as f:
        json_data=json.load(f)
    
    ann_idx=0
    gt_list=[]
    for img_idx in range(len(json_data['images'])):
        image_list=[]
        #print(img_idx)
        while(json_data['images'][img_idx]['id']==json_data['annotations'][ann_idx]['image_id']):
            t={}
            t['score']=1.0
            t['bbox']=json_data['annotations'][ann_idx]['bbox'].copy()
            t['bbox'][2]+=t['bbox'][0]
            t['bbox'][3]+=t['bbox'][1]
            t['bbox']=np.array(t['bbox'],dtype='float32')
            t['cate']=json_data['annotations'][ann_idx]['category_id']-1
            t['image_name']=json_data['images'][img_idx]['file_name']
            t['image_class']=t['image_name'].split('_')[0]
            #print(t['image_class'])
            t['image_class_idx']=config.labels2idx[t['image_class']]
            image_list.append(t)
            ann_idx+=1
            if(ann_idx>=len(json_data['annotations'])):
                break
        gt_list.append(image_list)
    return gt_list
#b=get_gt_list(config.PATH_GT)
#b[2]

'''
空间排序方法：
1.选择置信度*面积最大的为第一个，将其他的bbox按照相对第一个的距离进行排序
2.设置vis list，为遍历过的所有bbox
3.遍历不在vis中的bbox，和在vis中的所有bbox相比，选择最近的一个为minbbox
4.如果和minbbox的距离大于正在比的bbox，则判断距离远，中间空出零

'''
class SpaceSort:
    def getArea(bbox):#获得一个bbox的面积
        return (bbox[2]-bbox[0])*(bbox[3]-bbox[1])

    def getCenter(bbox):#获得一个bbox的中心点
        center=[bbox[0]+(bbox[2]-bbox[0])/2,bbox[1]+(bbox[3]-bbox[1])/2]
        return center

    def getDistance(bbox1,bbox2):#获得两个bbox中心点间的距离
        center1=SpaceSort.getCenter(bbox1)
        center2=SpaceSort.getCenter(bbox2)
        return (abs((center2[0]-center1[0])**2+(center2[1]-center1[1])**2))**0.5

    def getDiagonal(bbox1):
        return ((bbox1[3]-bbox1[1])**2+(bbox1[2]-bbox1[0])**2)**0.5

    def getMaxBbox(image):#获得面积*置信度最大的bbox
        maxIdx=0
        maxArea=0
        if(str(type(image[0]))=="<class 'dict'>"):
            for idx,i in enumerate(image):
                if(i['score']*SpaceSort.getArea(i['bbox'])>maxArea):
                    maxArea=SpaceSort.getArea(i['bbox'])*i['score']
                    maxIdx=idx
            return image[maxIdx]
        else:
            for idx,i in enumerate(image):
                if(i[0]['score']*SpaceSort.getArea(i[0]['bbox'])>maxArea):
                    maxArea=SpaceSort.getArea(i[0]['bbox'])*i[0]['score']
                    maxIdx=idx
            return image[maxIdx][0]
    
    def spaceSort(image):#最大的排前面，然后按照与最大的距离排序，但是不插空
        image_sorted=image.copy()
        max_bbox=SpaceSort.getMaxBbox(image)
        for i in range(len(image)-1):
            for j in range(len(image)-1-i):
                if(SpaceSort.getDistance(max_bbox['bbox'],image_sorted[j]['bbox'])>SpaceSort.getDistance(max_bbox['bbox'],image_sorted[j+1]['bbox'])):
                    t=image_sorted[j]
                    image_sorted[j]=image_sorted[j+1]
                    image_sorted[j+1]=t
        return image_sorted
                
    def iou_small(gt_bbox,pr_bbox):#判断一个bbox是不是在一个bbox中，是的话返回交/小的面积
        gt=gt_bbox.copy()
        pr=pr_bbox.copy()
        lou_w=min(gt[0],gt[2],pr[0],pr[2])+(gt[2]-gt[0])+(pr[2]-pr[0])-max(gt[0],gt[2],pr[0],pr[2])
        lou_h=min(gt[1],gt[3],pr[1],pr[3])+(gt[3]-gt[1])+(pr[3]-pr[1])-max(gt[1],gt[3],pr[1],pr[3])
        if(lou_w<0 or lou_h<0):
            return(-1)
        else:
            return(lou_w*lou_h/min(SpaceSort.getArea(gt),SpaceSort.getArea(pr)))
    
    def blankNum(bbox1,bbox2):
        #输入两个bbox，判断要插几个空
        #算法：两个bbox中心的距离减去bbox1的对角线/2，减去bbox2的对角线/2，上述结果除以小的bbox的对角线，四舍五入取整
        bbox1_diagonal=SpaceSort.getDiagonal(bbox1)
        bbox2_diagonal=SpaceSort.getDiagonal(bbox2)
        num_blank=SpaceSort.getDistance(bbox1,bbox2)-(bbox1_diagonal+bbox2_diagonal)/2
        num_blank=num_blank/min(SpaceSort.getDiagonal(bbox1),SpaceSort.getDiagonal(bbox2))
        num_blank=int(round(num_blank,0))
        if(num_blank<0):
            num_blank=0
        return num_blank
        
        
    def spaceSortMerge(image):#同上，但是包含关系的，则合并到一个list中去。比上述的return结果list要多一个维度
        image_sorted=SpaceSort.spaceSort(image)#先按照位置关系排序
        vis=[]
        for idx,i in enumerate(image_sorted):
            if(idx==0):#第一个直接添加到vis
                vis.append([i])
            else:
                flag_spaceSortMerge=0
                for visiteds_idx in range(len(vis)):  
                    for visited_idx in range(len(vis[visiteds_idx])):
                        if(SpaceSort.iou_small(vis[visiteds_idx][visited_idx]['bbox'],i['bbox'])>0.8):#有包含的，则加入list
                            vis[visiteds_idx].append(i)
                            flag_spaceSortMerge=1
                            break
                    if(flag_spaceSortMerge==1):
                        break
                if(flag_spaceSortMerge==0):#如果没有包含的，没加入任何list，则新建一个list并加入vis
                    vis.append([i])
        for idx in range(len(vis)):
            vis[idx]=SpaceSort.spaceSort(vis[idx])
        #合并完成，并将每个位置的最大bbox放在该list的最前面，开始插空
        
        return vis
                            

    def spaceSortBlank(image):#插空算法
        image_sorted=SpaceSort.spaceSort(image)
        vis=[]
        for idx,i in enumerate(image_sorted):
            if(idx==0):#第一个直接添加到vis
                vis.append(i)
            else:
                #min_visited 距离当前i最近的bbox
                min_distance=99999
                for visited in vis:  
                    if(visited==[]):#插空后要跳过
                        continue
                    if(SpaceSort.getDistance(visited['bbox'],i['bbox'])<min_distance):
                        min_distance=SpaceSort.getDistance(visited['bbox'],i['bbox'])
                        min_distance_bbox=visited
                num_blank=SpaceSort.blankNum(min_distance_bbox['bbox'],i['bbox'])
                #print(min_distance)
                #print('大对角线：'+str(max(SpaceSort.getDiagonal(i['bbox']),SpaceSort.getDiagonal(min_distance_bbox['bbox']))))
                #print('对角线：'+str(min(SpaceSort.getDiagonal(i['bbox']),SpaceSort.getDiagonal(min_distance_bbox['bbox']))))
                #print(num_blank)
                for blank in range(num_blank):
                    #print(123)
                    vis.append([])
                vis.append(i)
        return vis


def get_list_max(t):#sort排序算法用
    return max(t)

def List2OneHot(image_list,fun,length,matrix=False,merge=False,onehot='confidence'):#将一个image的信息转为10*8的onehot fun表示排序方式。

    if (image_list==[]):
        return [[0.0 for i in range(8)] for j in range(length)]

    if(fun=='None' or fun=='confidence' or fun=='spaceSort'):
        list_t=[]
        if(fun=='spaceSort'):
            image_list=SpaceSort.spaceSort(image_list)
        for bbox in image_list:
            if(matrix==True):
                list_t.append(OneHotMatrix(bbox['score'],bbox['cate'],8))
            elif(merge==True):
                t3=[0.0 for t0 in range(8)]
                #print(t3)
                for j in bbox:
                    #print(OneHotAdd(bbox['score'],bbox['cate'],t3)
                    #print(OneHotAdd(bbox['score'],bbox['cate'],t3)
                    t3=OneHotAdd(j['score'],j['cate'],t3)
                list_t.append(t3)
            else:
                list_t.append(OneHot(bbox['score'],bbox['cate'],8))
    if(fun=='confidence'):
        list_t.sort(key=get_list_max,reverse=True)
    elif(fun=='spaceSortBlank'):
        list_t=[]
        max_bbox=SpaceSort.getMaxBbox(image_list)
        max_bbox_area=SpaceSort.getArea(max_bbox['bbox'])
        image_list=SpaceSort.spaceSortBlank(image_list)
        
        for bbox in image_list:
            if(bbox==[]):
                list_t.append([0.0 for i in range(8)])
            else:
                bbox_area=SpaceSort.getArea(bbox['bbox'])
                if(onehot=='confidence'):
                    list_t.append(OneHot(bbox['score'],bbox['cate'],8))
                elif(onehot=='area'):
                    list_t.append(OneHot(bbox_area/max_bbox_area,bbox['cate'],8))
                elif(onehot=='1'):
                    list_t.append(OneHot(1,bbox['cate'],8))
                elif(onehot=='confidenceXarea'):
                    list_t.append(OneHot(bbox['score']*(bbox_area/max_bbox_area),bbox['cate'],8))
                #list_t.append(OneHot(bbox['score'],bbox['cate'],8))
                #list_t.append(OneHot(bbox['score']*bbox_area,bbox['cate'],8))
    elif(fun=='spaceSortMerge'):
        list_t=[]
        max_bbox=SpaceSort.getMaxBbox(image_list)
        max_bbox_area=SpaceSort.getArea(max_bbox['bbox'])
        image_list=SpaceSort.spaceSortMerge(image_list)
        for bbox in image_list:
            t3=[0.0 for t0 in range(8)]
            for j in bbox:
                bbox_area=SpaceSort.getArea(j['bbox'])
                if(onehot=='confidence'):
                    t3=OneHotAdd(j['score'],j['cate'],t3)
                elif(onehot=='area'):
                    t3=OneHotAdd(bbox_area/max_bbox_area,j['cate'],t3)
                elif(onehot=='confidenceXarea'):
                    t3=OneHotAdd(j['score']*(bbox_area/max_bbox_area),j['cate'],t3)
            list_t.append(t3)
                
    #print(list_t)
        
        
    
    if(len(list_t)<length):#固定二维长度
        for i in range(length-(len(list_t))):
            list_t.append([0.0 for j in range(8)])
        
    elif(len(list_t)>length):
        list_t=list_t[:length]
    
    
    
    return list_t

def get_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(fmt="%(message)s")

    sHandler = logging.StreamHandler()
    sHandler.setFormatter(formatter)
    logger.addHandler(sHandler)

    if not os.path.exists(config.WORK_DIR+'/log'):
        os.makedirs(config.WORK_DIR+'/log')
        #LOG_DIR +'/'+ RNN_TYPE+'_'+BIDIRECTIONAL+'_split'+str(SPLIT)+'.log'
    localtime = time.asctime( time.localtime(time.time()) )
    fHandler = logging.FileHandler("{}/{}.log".format(config.WORK_DIR+'/log','SLEDDing_all_hidden_reverse_GT_train_PR_test'), mode='w')
    fHandler.setLevel(logging.DEBUG)
    fHandler.setFormatter(formatter)
    logger.addHandler(fHandler)
    return logger
#logger=get_logger()

class BEAUTY_bbox_Dataset(Dataset):
    def __init__(self,pr_list,fun,max_l,merge=False,onehot='confidence',multi_task='False'):
        super().__init__()
        self.pr_list = pr_list
        self.fun=fun
        self.max_l=max_l
        self.merge=merge
        self.onehot=onehot
        self.multi_task=multi_task

    def __len__(self):
        return len(self.pr_list)

    def __getitem__(self, idx):
        #print(idx)
        #print(len(self.pr_list))
        while(len(self.pr_list[idx])==0):
            idx+=1
        #print(idx)
        image=self.pr_list[idx]
        #print(image[0]['image_name'])
        #print(image)
        input_onehot=List2OneHot(image,self.fun,self.max_l,onehot=self.onehot)#************True记得放到config里面去
        input_onehot.reverse()
        #print(input_onehot)
        if(self.merge==True):
            label=image[0][0]['image_class_idx']
        else:
            label=image[0]['image_class_idx']
        #print(image)
        #print(input_onehot)
        input_onehot=torch.Tensor(input_onehot)
        
        if (self.multi_task=='True'):
            bbox_label=Bbox_labels(image,self.max_l)
            bbox_label=torch.Tensor(bbox_label)
            return input_onehot,bbox_label,label

        return input_onehot,label


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,num_layers,bidirectional,rnn_type):#输入，隐藏层单元，输出，隐藏层数，双向
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        if(bidirectional=='True'):
            self.bidirectional=True
        elif(bidirectional=='False'):
            self.bidirectional=False
        if(rnn_type=='GRU'):
            self.rnn=nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,bidirectional=self.bidirectional)
        elif(rnn_type=='LSTM'):
            self.rnn=nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,bidirectional=self.bidirectional)
        elif(rnn_type=='RNN'):
            self.rnn=nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,bidirectional=self.bidirectional)
        
        if(bidirectional=='True'):
            self.out = nn.Linear(4*hidden_size, output_size)
        else:
            self.out = nn.Linear(1*hidden_size, output_size)

    def forward(self, input, hidden):
        #print(input.size())
        #10*64*8
        output1,hidden=self.rnn(input,hidden)
        #print("output1:{}".format(output1.size()))
        #10*64*32
        if(self.bidirectional==True):
            output2 = torch.cat((output1[0], output1[-1]), -1)
        else:
            output2 = output1[-1]
        #print("output2:{}".format(output2.size()))
        #64*64
        #print("hidden:{}".format(hidden.size()))
        #层数*64*结点数
        #2*64*32
        output=self.out(output2)
        #print(output.size())
        #64*4
        #output=F.softmax(output,dim=0)
        return output, hidden

class RNN2(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,num_layers,bidirectional,rnn_type):#输入，隐藏层单元，输出，隐藏层数，双向
        super(RNN2, self).__init__()

        self.hidden_size = hidden_size
        if(bidirectional=='True'):
            self.bidirectional=True
        elif(bidirectional=='False'):
            self.bidirectional=False
        if(rnn_type=='GRU'):
            self.rnn=nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,bidirectional=self.bidirectional)
        elif(rnn_type=='LSTM'):
            self.rnn=nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,bidirectional=self.bidirectional)
        elif(rnn_type=='RNN'):
            self.rnn=nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,bidirectional=self.bidirectional)
        if(bidirectional=='True'):
            self.out = nn.Linear(2*25*hidden_size, output_size)
        else:
            self.out = nn.Linear(25*hidden_size, output_size)

    def forward(self, input, hidden):
        #输入：10*64*8
        output1,hidden=self.rnn(input,hidden)
        #print("output1:{}".format(output1.size()))
        #10*64*32
        output2=output1[0]
        for i in range(1,25):
            output2=torch.cat((output2,output1[i]),-1)
        #output2 = torch.cat((output1[0],output1[1],output1[2],output1[3],output1[4],output1[5],output1[6],output1[7],output1[8], output1[9]), -1)

        #print("output2:{}".format(output2.size()))
        #64*64
        #print("hidden:{}".format(hidden.size()))
        #1*64*32
        #层数*64*结点数
        output=self.out(output2)
        #output=F.softmax(output,dim=0)
        return output, hidden


class RNN_multi_task(nn.Module):#直接指定output_size，这个参数用不上，但我懒得改
    def __init__(self, input_size, hidden_size, output_size,num_layers,bidirectional,rnn_type):#输入，隐藏层单元，输出，隐藏层数，双向
        super(RNN_multi_task, self).__init__()

        self.hidden_size = hidden_size
        if(bidirectional=='True'):
            self.bidirectional=True
        elif(bidirectional=='False'):
            self.bidirectional=False
        if(rnn_type=='GRU'):
            self.rnn=nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,bidirectional=self.bidirectional)
        elif(rnn_type=='LSTM'):
            self.rnn=nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,bidirectional=self.bidirectional)
        elif(rnn_type=='RNN'):
            self.rnn=nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,bidirectional=self.bidirectional)
        if(bidirectional=='True'):
            self.out = nn.Linear(50*hidden_size, output_size)
            self.out_confidence=nn.Linear(50*hidden_size, 25)
        else:
            self.out = nn.Linear(25*hidden_size, output_size)
            self.out_confidence=nn.Linear(25*hidden_size, 25)

    def forward(self, input, hidden):
        
        output1,hidden=self.rnn(input,hidden)
        #print("output1:{}".format(output1.size()))
        output2=output1[0]
        for i in range(1,25):
            output2=torch.cat((output2,output1[i]),-1)
        #output2 = torch.cat((output1[0],output1[1],output1[2],output1[3],output1[4],output1[5],output1[6],output1[7], output1[8],output1[-1]), -1)
     
        #print("output2:{}".format(output2.size()))
        output=self.out(output2)
        output_confidence=self.out_confidence(output2)
        #output=F.softmax(output,dim=0)
        return output,output_confidence,hidden


def evaluate_loss(data_iter, net,hidden,device, epoch):
    net.eval()
    l_sum, n,val_pred_sum= 0.0, 0,0
    with torch.no_grad():
        for bidx,(X, y) in enumerate(data_iter):
            X=X.float()
            X=X.permute(1,0,2)#[32, 10, 8]->[10,32,8]
            y=y.to(device)
            X=X.to(device)
            
            y_hat,hidden= net(X,hidden)
            loss = criterion(y_hat, y)
            
            l_sum+=loss
            #l_sum +=criterion(y_hat, y).item()
            n += X.size()[1]
            pred=y_hat.max(1, keepdim=True)[1].view(-1)
            pred_sum=pred.eq(y.view_as(pred)).sum().item()
            val_pred_sum+=pred_sum
    #print('val,epoch:{},pred:{},loss:{}'.format(epoch,val_pred_sum/n,l_sum/n))
    logger.info("[epoch {}][{}][end] val_loss={:.5f},val_acc:{:.5f}({}/{})".format\
    (epoch,'val',l_sum/(bidx+1),val_pred_sum/n,int(val_pred_sum),n))
    return l_sum / (bidx+1),val_pred_sum/n

    

def evaluate_loss_multi_task(data_iter, net,hidden,device, epoch):
    net.eval()
    l_sum, n,val_pred_sum= 0.0, 0,0
    with torch.no_grad():
        for bidx,(X, y_confidence,y) in enumerate(data_iter):
            X=X.float()
            X=X.permute(1,0,2)#[32, 10, 8]->[10,32,8]
            y=y.to(device)
            X=X.to(device)
            y_confidence=y_confidence.to(device)
            
            y_hat,y_confidence_hat,hidden= net(X,hidden)
            loss=nn.CrossEntropyLoss()(y_hat,y)+nn.MSELoss()(y_confidence,y_confidence_hat)
            
            l_sum+=loss
            #l_sum +=criterion(y_hat, y).item()
            n += X.size()[1]
            pred=y_hat.max(1, keepdim=True)[1].view(-1)
            pred_sum=pred.eq(y.view_as(pred)).sum().item()
            val_pred_sum+=pred_sum
    #print('val,epoch:{},pred:{},loss:{}'.format(epoch,val_pred_sum/n,l_sum/n))
    logger.info("[epoch {}][{}][end] val_loss={:.5f},val_acc:{:.5f}({}/{})".format\
    (epoch,'val',l_sum/(bidx+1),val_pred_sum/n,int(val_pred_sum),n))
    return l_sum / (bidx+1),val_pred_sum/n


def train_multi_task(net,train_iter,val_iter,num_epoch,lr_period,lr_decay):
    optimizer = torch.optim.Adam(net.parameters(),lr=0.01)
    hidden=None
    Max_Acc=0.0
    for epoch in range(num_epoch):
        net.train()
        n,train_l_sum,train_pred_sum=0,0,0
        loss_confidence_sum,loss_classfication_sum=0.0,0.0
        if epoch > 0 and epoch % lr_period == 0:
            optimizer.param_groups[0]['lr']=optimizer.param_groups[0]['lr']*lr_decay
            
        for bidx, (X,y_confidence,y) in enumerate(train_iter):
            X=X.float()
            X=X.permute(1,0,2)#[32, 10, 8]->[10,32,8]
            X=X.to(config.device)
            y=y.to(config.device)
            y_confidence=y_confidence.to(config.device)
            if(hidden is not None):
                
                if isinstance (hidden, tuple): # LSTM, state:(h, c)  
                    hidden[0].to(config.device)
                    hidden[1].to(config.device)
                    hidden = (hidden[0].detach(), hidden[1].detach())
                else:   
                    hidden.to(config.device)
                    hidden = hidden.detach()
            
            optimizer.zero_grad()
            #print(str(bidx)+'---------------')
            #print(hidden)
            y_hat,y_confidence_hat,hidden= net(X,hidden)
            #print(y_hat.size())
            #loss = criterion(y_hat, y)
            loss_classfication=nn.CrossEntropyLoss()(y_hat,y)
            loss_confidence=nn.MSELoss()(y_confidence,y_confidence_hat)
            loss=loss_classfication+loss_confidence
            #print(y_hat)
            #print(y)
            #print(loss.item())
            loss.backward()
            optimizer.step()
            #print(loss)
            pred=y_hat.max(1, keepdim=True)[1].view(-1)
            pred_sum=pred.eq(y.view_as(pred)).sum().item()
            #print(pred_sum)
            train_l_sum+=loss

            loss_confidence_sum+=loss_confidence
            loss_classfication_sum+=loss_classfication


            
            train_pred_sum+=pred_sum
            n+=X.size(1)
        #print("train,epoch:{},pred:{},loss:{}".format(epoch,train_pred_sum/n,train_l_sum/n))
        #print(str(train_pred_sum)+'/'+str(n))
        #print(train_pred_sum/n)
        if not os.path.exists('./params'):
            os.makedirs('./params')
        logger.info("[epoch {}][{}][end] train_loss={:.5f},loss_classfication={:.5f},loss_confidence={:.5f},train_acc={:.5f}({}/{})".format\
                    (epoch,'train',train_l_sum/(bidx+1),loss_classfication_sum/(bidx+1),loss_confidence_sum/(bidx+1),train_pred_sum/n,int(train_pred_sum),n))
        valid_loss,valid_acc=evaluate_loss_multi_task(val_iter, net,hidden,config.device,epoch)
        if(valid_acc>Max_Acc):
            Max_Acc=valid_acc
            model_best=net
            #torch.save(net,'./params/'+MODEL_NAME+'_'+DATASET_NAME+'_best.pth')
           # torch.save(net,'./params/{}_BIDI-{}_NUMLAYERS-{}_HIDDENSIZE-{}_best.pth'.format\
                      # (RNN_TYPE,BIDIRECTIONAL,NUM_LAYERS,HIDDEN_SIZE))
            logger.info("[epoch {}][save_best_output_params]".format(epoch))
            #n+=y.size()[0]
            #print(n)
    return model_best

#test后，得出acc
def test_multi_task(net,test_iter,device):
    net.eval()
    matrix=[[0 for j in range(4)] for i in range(4)]
    
    hidden=None
    sum=0
    n=0
    for bidx, (X,y_confidence,y) in enumerate(test_iter):
        X=X.float()
        X=X.permute(1,0,2)#[32, 10, 8]->[10,32,8]
        X=X.to(device)
        y=y.to(device)
        y_confidence=y_confidence.to(device)
        if(hidden is not None):

            if isinstance (hidden, tuple): # LSTM, state:(h, c)  
                hidden[0].to(device)
                hidden[1].to(device)
                hidden = (hidden[0].detach(), hidden[1].detach())
            else:   
                hidden.to(device)
                hidden = hidden.detach()
        y_hat,y_confidence_hat,hidden= net(X,hidden)


        pred=y_hat.max(1, keepdim=True)[1].view(-1)

        pred=pred.cpu().numpy().tolist()
        y=y.cpu().numpy().tolist()
        for idx in range(len(y)):
            #print(idx)
            matrix[y[idx]][pred[idx]]+=1

    return matrix


def train(net,train_iter,val_iter,num_epoch,lr_period,lr_decay):
    optimizer = torch.optim.Adam(net.parameters(),lr=0.01)
    hidden=None
    Max_Acc=0.0
    Min_loss=9999999.9
    for epoch in range(num_epoch):
        net.train()
        n,train_l_sum,train_pred_sum=0,0,0
        if epoch > 0 and epoch % lr_period == 0:
            optimizer.param_groups[0]['lr']=optimizer.param_groups[0]['lr']*lr_decay
            
        for bidx, (X,y) in enumerate(train_iter):
            X=X.float()
            X=X.permute(1,0,2)#[32, 10, 8]->[10,32,8]
            X=X.to(config.device)
            y=y.to(config.device)
            if(hidden is not None):
                
                if isinstance (hidden, tuple): # LSTM, state:(h, c)  
                    hidden[0].to(config.device)
                    hidden[1].to(config.device)
                    hidden = (hidden[0].detach(), hidden[1].detach())
                else:   
                    hidden.to(config.device)
                    hidden = hidden.detach()
            
            optimizer.zero_grad()
            #print(str(bidx)+'---------------')
            #print(hidden)
            y_hat,hidden= net(X,hidden)
            #print(y_hat.size())
            loss = criterion(y_hat, y)
            #print(y_hat)
            #print(y)
            #print(loss.item())
            loss.backward()
            optimizer.step()
            #print(loss)
            pred=y_hat.max(1, keepdim=True)[1].view(-1)
            pred_sum=pred.eq(y.view_as(pred)).sum().item()
            #print(pred_sum)
            train_l_sum+=loss
            train_pred_sum+=pred_sum
            n+=X.size(1)
        #print("train,epoch:{},pred:{},loss:{}".format(epoch,train_pred_sum/n,train_l_sum/n))
        #print(str(train_pred_sum)+'/'+str(n))
        #print(train_pred_sum/n)
        if not os.path.exists('./params'):
            os.makedirs('./params')
        logger.info("[epoch {}][{}][end] train_loss={:.5f},train_acc={:.5f}({}/{})".format\
                    (epoch,'train',train_l_sum/(bidx+1),train_pred_sum/n,int(train_pred_sum),n))
        valid_loss,valid_acc=evaluate_loss(val_iter, net,hidden,config.device,epoch)
        if(valid_loss<Min_loss):
            Min_loss=valid_loss
            model_best=net
            #torch.save(net,'./params/'+MODEL_NAME+'_'+DATASET_NAME+'_best.pth')
           # torch.save(net,'./params/{}_BIDI-{}_NUMLAYERS-{}_HIDDENSIZE-{}_best.pth'.format\
                      # (RNN_TYPE,BIDIRECTIONAL,NUM_LAYERS,HIDDEN_SIZE))
            logger.info("[epoch {}][save_best_output_params]".format(epoch))
            #n+=y.size()[0]
            #print(n)
        '''
        if(valid_acc>Max_Acc):
            Max_Acc=valid_acc
            model_best=net
            #torch.save(net,'./params/'+MODEL_NAME+'_'+DATASET_NAME+'_best.pth')
           # torch.save(net,'./params/{}_BIDI-{}_NUMLAYERS-{}_HIDDENSIZE-{}_best.pth'.format\
                      # (RNN_TYPE,BIDIRECTIONAL,NUM_LAYERS,HIDDEN_SIZE))
            logger.info("[epoch {}][save_best_output_params]".format(epoch))
            #n+=y.size()[0]
            #print(n)
        '''
    return model_best

#test后，得出acc
def test(net,test_iter,device):
    net.eval()
    matrix=[[0 for j in range(4)] for i in range(4)]
    
    hidden=None
    sum=0
    n=0
    for bidx, (X,y) in enumerate(test_iter):
        X=X.float()
        X=X.permute(1,0,2)#[32, 10, 8]->[10,32,8]
        X=X.to(device)
        y=y.to(device)
        if(hidden is not None):

            if isinstance (hidden, tuple): # LSTM, state:(h, c)  
                hidden[0].to(device)
                hidden[1].to(device)
                hidden = (hidden[0].detach(), hidden[1].detach())
            else:   
                hidden.to(device)
                hidden = hidden.detach()
        y_hat,hidden= net(X,hidden)
        pred=y_hat.max(1, keepdim=True)[1].view(-1)
        '''
        print(y_hat)
        print(gjifjsg)
        pred=pred.cpu().numpy().tolist()
        y=y.cpu().numpy().tolist()
        for idx in range(len(y)):
            #print(idx)
            matrix[y[idx]][pred[idx]]+=1
        '''
        #print(y_hat)
        #print(y)
        pred_sum=pred.eq(y.view_as(pred)).sum().item()
        sum+=pred_sum
        n+=len(y)
    return sum/n

#test后，得出acc
def test_matrix(net,test_iter,device):
    net.eval()
    matrix=[[0 for j in range(4)] for i in range(4)]
    
    hidden=None
    sum=0
    n=0
    for bidx, (X,y) in enumerate(test_iter):
        X=X.float()
        X=X.permute(1,0,2)#[32, 10, 8]->[10,32,8]
        X=X.to(device)
        y=y.to(device)
        if(hidden is not None):

            if isinstance (hidden, tuple): # LSTM, state:(h, c)  
                hidden[0].to(device)
                hidden[1].to(device)
                hidden = (hidden[0].detach(), hidden[1].detach())
            else:   
                hidden.to(device)
                hidden = hidden.detach()
        y_hat,hidden= net(X,hidden)
        pred=y_hat.max(1, keepdim=True)[1].view(-1)

        pred=pred.cpu().numpy().tolist()
        y=y.cpu().numpy().tolist()
        for idx in range(len(y)):
            #print(idx)
            matrix[y[idx]][pred[idx]]+=1

        #print(y_hat)
        #print(y)
        #pred_sum=pred.eq(y.view_as(pred)).sum().item()
        #sum+=pred_sum
        #n+=len(y)
    return matrix

#test后，得出acc
def test2(net,test_iter,device):
    net.eval()
    matrix=[[0 for j in range(4)] for i in range(4)]
    
    hidden=None
    sum=[]
    with torch.no_grad():
        for bidx, (X,y) in enumerate(test_iter):
            X=X.float()
            X=X.permute(1,0,2)#[32, 10, 8]->[10,32,8]
            X=X.to(device)
            y=y.to(device)
            if(hidden is not None):

                if isinstance (hidden, tuple): # LSTM, state:(h, c)  
                    hidden[0].to(device)
                    hidden[1].to(device)
                    hidden = (hidden[0].detach(), hidden[1].detach())
                else:   
                    hidden.to(device)
                    hidden = hidden.detach()
            y_hat,hidden= net(X,hidden)
            y_hat=F.softmax(y_hat,dim=1)
            y_hat=y_hat.cpu().numpy().tolist()
            #print(y_hat)
            #sum.append(y_hat)
            sum+=y_hat
    return sum

def F1(maxtir):
    sum_Pre=0
    sum_Re=0
    sum_all=0
    sum_TP=0
    for i in range(len(maxtir)):
        TP=maxtir[i][i]
        FP_and_TP=0
        FN_and_TP=0
        for j in range(len(maxtir)):
            sum_all+=maxtir[i][j]
            FN_and_TP+=maxtir[i][j]
            FP_and_TP+=maxtir[j][i]
        sum_TP+=TP
        if (FP_and_TP==0):
            sum_Pre+=0.0
        else:
            sum_Pre+=TP/FP_and_TP
        if FN_and_TP==0:
            sum_Re+=0.0
        else:
            sum_Re+=TP/FN_and_TP
        #print('Pression:'+str(TP/FP_and_TP))
        #print('recall:'+str(TP/FN_and_TP))
        #print('\n')
    sum_Pre/=4
    sum_Re/=4
    return {'Pression':sum_Pre,'Recall':sum_Re,'F1':(2*sum_Pre*sum_Re)/(sum_Pre+sum_Re),'Acc':sum_TP/sum_all}

if __name__ == '__main__':
    RNNs=['RNN','LSTM','GRU']
    DETECTIONs=['cascade_rcnn_r101','cascade_rcnn_r50','faster_rcnn_r101','faster_rcnn_r50']
    BIDIRECTIONALs=['True','False']
    FUNs=['None','multi_task','Spatial_Layout_Encoding']
    list_result={}
    logger=get_logger()

    #连接全节点
    logger.info('--------------------倒序 双向/单向 全连 空间排序  GT train  PR test--------------------')
    RNNs=['RNN','LSTM','GRU']
    DETECTIONs=['cascade_rcnn_r101','cascade_rcnn_r50','faster_rcnn_r101','faster_rcnn_r50']
    BIDIRECTIONALs=['True','False']
    FUNs=['None','multi_task','Spatial_Layout_Encoding']
    list_result1={}
    
    list_result1["[RNN={},DETECTION={},BIDIRECTIONAL={},FUN={}]".format('RNN','cascade_rcnn_r101','True','Spatial_Layout_Encoding')]=[]
    for b in range(5):
        logger.info("[RNN={},DETECTION={},BIDIRECTIONAL={},FUN={}]".format('RNN','cascade_rcnn_r101','True','Spatial_Layout_Encoding'))
        train_pr_list=get_gt_list(config.PATH_GT_TRAIN)
        val_pr_list=get_pr_list(config.PATH_PR['cascade_rcnn_r101']+'results_val.pkl',config.PATH_GT_VAL)
        test_pr_list=get_pr_list(config.PATH_PR['cascade_rcnn_r101']+'results_test.pkl',config.PATH_GT_TEST)

        train_ds=BEAUTY_bbox_Dataset(train_pr_list,'spaceSortBlank',25,onehot='confidence')
        val_ds=BEAUTY_bbox_Dataset(val_pr_list,'spaceSortBlank',25,onehot='confidence')
        test_ds=BEAUTY_bbox_Dataset(test_pr_list,'spaceSortBlank',25,onehot='confidence')

        train_iter=DataLoader(train_ds,batch_size=config.BATCH_SIZE,shuffle=True,num_workers=5,drop_last=True)
        val_iter=DataLoader(val_ds,batch_size=config.BATCH_SIZE,shuffle=True,num_workers=5,drop_last=True)
        test_iter=DataLoader(test_ds,batch_size=1,shuffle=False,num_workers=5,drop_last=True)

        lr_period, lr_decay= 5, 0.1
        criterion = nn.CrossEntropyLoss()
        model=RNN2(8,16,4,2,'True','RNN').to(config.device)
        model=train(model,train_iter,val_iter,20,lr_period,lr_decay)

        maxtir=test_matrix(model,test_iter,config.device)
        result=F1(maxtir)
        result['maxtir']=maxtir
        list_result1["[RNN={},DETECTION={},BIDIRECTIONAL={},FUN={}]".format('RNN','cascade_rcnn_r101','True','Spatial_Layout_Encoding')].append(result)
        logger.info(str(maxtir))
        logger.info(str(F1(maxtir))+'\n')
    list_result['SLEDDing_all_hidden_reverse_GT_train_PR_test']=list_result1
    with open(config.WORK_DIR+'/SLEDDing_all_hidden_reverse_GT_train_PR_test.json', 'w+') as f:
        json.dump(list_result, f)
