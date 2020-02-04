# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 22:36:46 2019

@author: zll
"""

import os
from utils.process_labels import encode_labels, decode_labels, decode_color_labels
from utils.loss import MySoftmaxCrossEntropyLoss, DiceLoss
from utils.lr_scheduler import PolynomialLR
#from models.deeplabv3plus import deeplabv3p
from models.UNet_Resnet50 import UNetWithResnet50Encoder


import pandas as pd

import torchvision
import torch
import numpy as np
import random
from tqdm import tqdm

import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms as tfs
import torchvision.models as models
import torchvision.datasets as datasets
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
#from mxnet import  image
import torchvision.transforms.functional as TF
import cv2


from torch.autograd import Variable

from sklearn.utils import shuffle

import matplotlib.image as mp
import sys
#argv = sys.argv[1]
#torch.cuda.set_device(int(argv))
import warnings

# 获取每个 GPU 的剩余显存数，并存放到 tmp 文件中
os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
memory_gpu = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmax(memory_gpu))
#torch.cuda.set_device(np.argmax(memory_gpu)) 
os.system('rm tmp')  # 删除临时生成的 tmp 文件

def read_images():
    label_list = []
    image_list = []
    '''
    image_dir = 'E:/baidu lane detetion dataset/Image_Data/'
    label_dir = 'E:/baidu lane detetion dataset/Gray_Label/'
    '''
    image_dir = '/root/data/LaneSeg/Image_Data/'
    label_dir = '/root/data/LaneSeg/Gray_Label/'
    
    for s1 in os.listdir(image_dir):
        if s1 != 'Road03':  #过滤掉曝光严重的Road03
            image_sub_dir1 = os.path.join(image_dir, s1)
            label_sub_dir1 = os.path.join(label_dir, 'Label_' + str.lower(s1),
                                          'Label')
            # print(image_sub_dir1, label_sub_dir1)

            for s2 in os.listdir(image_sub_dir1):
                image_sub_dir2 = os.path.join(image_sub_dir1, s2)
                label_sub_dir2 = os.path.join(label_sub_dir1, s2)
                # print(image_sub_dir2, label_sub_dir2)

                for s3 in os.listdir(image_sub_dir2):
                    image_sub_dir3 = os.path.join(image_sub_dir2, s3)
                    label_sub_dir3 = os.path.join(label_sub_dir2, s3)
                    # print(image_sub_dir3, label_sub_dir3)

                    for s4 in os.listdir(image_sub_dir3):
                        s44 = s4.replace('.jpg', '_bin.png')
                        image_sub_dir4 = os.path.join(image_sub_dir3, s4)
                        label_sub_dir4 = os.path.join(label_sub_dir3, s44)
                        if not os.path.exists(image_sub_dir4):
                            print(image_sub_dir4)
                        if not os.path.exists(label_sub_dir4):
                            print(label_sub_dir4)
                        # print(image_sub_dir4, label_sub_dir4)
                        image_list.append(image_sub_dir4)
                        label_list.append(label_sub_dir4)
    #对应打乱image_list,label_list排序
    randnum = random.randint(0, 100)
    random.seed(randnum)
    random.shuffle(image_list)
    random.seed(randnum)
    random.shuffle(label_list)
    return image_list, label_list


'''
image_list,label_list=read_images(train=True)
image_1 = Image.open(image_list[0]).convert('RGB')
plt.imshow(image_1)
plt.show()
label_1 = Image.open(label_list[0])
plt.imshow(label_1)
plt.show()
#label_1=np.array(label_1)
'''


def onehot_label(label):
    zs = np.array([np.zeros_like(label) for i in range(8)], dtype=np.float32)
    for i in range(8):
        zs[i][label == i] = 1
    return zs



'''
image_1 = Image.open(image_list[0]).convert('RGB')
label_1 = Image.open(label_list[0])
offset=690
image_size=[512,1536]
img_transforms(image_1, label_1,offset, image_size)
'''


def crop_resize_data(image, label, image_size, offset):
        roi_image = image[offset:, :]
        roi_label = label[offset:, :]
        image_crop = cv2.resize(roi_image, (image_size[0], image_size[1]),
                                 interpolation=cv2.INTER_LINEAR)
        label_crop = cv2.resize(roi_label, (image_size[0], image_size[1]),
                                 interpolation=cv2.INTER_NEAREST)
        return image_crop, label_crop


class BaiduLane_Dataset(Dataset):
    def __init__(self, data_list, label_list, offset, crop_size):
        self.offset = offset
        self.crop_size = crop_size
        self.data_list = data_list
        self.label_list = label_list

        print('Read ' + str(len(data_list)) + ' images')

    def __getitem__(self, idx):
        img = self.data_list[idx]
        label = self.label_list[idx]
        img = cv2.imread(img)
        label = cv2.imread(label,0)
        img,label = crop_resize_data(img,label,self.crop_size,self.offset )
        img = tfs.ToTensor()(img)
        label = encode_labels(label)
        olabel = onehot_label(label)
        return img,olabel,label

    def __len__(self):
        return len(self.data_list)

# =============================================================================
# M_IOU
# =============================================================================
def M_IOU(label_true, label_pred, n_class):
    i = 0
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(n_class * label_true[mask].astype(int) \
                       +label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)

    #print((hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)))

    iou = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    m_iou = np.nanmean(iou[1:])
    return m_iou

def compute_iou(pred, gt):
    result = {
            "TP": {i: 0
                   for i in range(8)},
            "TA": {i: 0
                   for i in range(8)}
        }
    pred = pred.detach().cpu().numpy()
    gt = gt.detach().cpu().numpy()
#    print("pred.shape:",pred.shape)
#    print("gt.shape:",gt.shape)
    for i in range(8):
        single_gt = gt==i
        single_pred = pred==i
        temp_tp = np.sum(single_gt * single_pred)
        temp_ta = np.sum(single_pred) + np.sum(single_gt) - temp_tp
        result["TP"][i] += temp_tp
        result["TA"][i] += temp_ta
    return result

# =============================================================================
# Train_Val
# =============================================================================
def Train_Val(epoches, net, train_data,val_data):    
    net = net.train()
    net = net.cuda()
    loss1 = nn.BCEWithLogitsLoss().cuda()
    loss2 = DiceLoss().cuda()
    Sum_Train_miou = 0
    Sum_Val_miou=0
    for e in range(epoches):
        #train_loss = 0
        train_mean_iou = 0
        j = 0
        process = tqdm(train_data)
        losses = []
        
        for data in process:
            j+=1
            with torch.no_grad():
                im = Variable(data[0].cuda())
                label = Variable(data[1].cuda())  #lable_onehot
                #label1 = Variable(data[2].cuda())
            #print("im.shape:",im.shape) #torch.Size([2, 3, 256, 768])
            #print("label.shape:",label.shape) #torch.Size([2, 8, 256, 768])
            out = net(im)
            #out_softmax=F.log_softmax(out, dim=1) 
            sig = torch.sigmoid(out)

            loss = loss1(out,label)+loss2(sig,label)
            losses.append(loss.item())

            #backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Update learning rate
            process.set_postfix_str(f"loss {np.mean(losses)}")
            pred = torch.argmax(F.softmax(out, dim=1), dim=1)
            mask = torch.argmax(F.softmax(label, dim=1), dim=1)
            #print("pred.shape:",pred.shape)#torch.Size([2, 256, 768])
            #print("mask.shape:",mask.shape) #  torch.Size([2, 256, 768])         
            result = compute_iou(pred, mask)
            if j % 200 == 0:
                tmiou =[]
                for i in range(1, 8):
                    if result["TA"][i] !=0: 
                        t_miou_i=result["TP"][i] / result["TA"][i]
                        result_string = "{}: {:.4f} \n".format(i, t_miou_i)
                        print(result_string)
                        tmiou.append(t_miou_i)     
                #tmiou = tmiou / 7
                t_miou=np.mean(tmiou)
                print("train_mean_iou:",t_miou)
            if j % 500 == 0:
                torch.save(net.state_dict(), 'deeplabv3p_baidulane.pth')

        torch.save(net.state_dict(), 'deeplabv3p_baidulane.pth')
        
        j=0
        #net.load_state_dict(torch.load('./deeplabv3p_baidulane.pth'))
        #net=net.cuda()
        process = tqdm(val_data)
        losses = []
        result = {
            "TP": {i: 0
                   for i in range(8)},
            "TA": {i: 0
                   for i in range(8)}
        }

        net = net.eval()
        val_mean_iou = 0        
        for data in process:
            j+=1
            with torch.no_grad():
              im = Variable(data[0].cuda())
              label = Variable(data[1].cuda())
              #label_1 = Variable(data[2].cuda())
            # forward
            out = net(im)
            sig = torch.sigmoid(out)
            loss = loss1(out,label)+loss2(sig,label)
            losses.append(loss.item())
            
            pred = torch.argmax(F.softmax(out, dim=1), dim=1)
            mask = torch.argmax(F.softmax(label, dim=1), dim=1)
            result = compute_iou(pred, mask)
            process.set_postfix_str(f"loss {np.mean(losses)}")

            if j % 200 == 0:
                vmiou = []
                for i in range(1, 8):
                    if result["TA"][i] !=0: 
                        v_miou_i=result["TP"][i] / result["TA"][i]
                        result_string = "{}: {:.4f} \n".format(i, v_miou_i)
                        print(result_string)
                        vmiou.append(v_miou_i)  
                v_miou=np.mean(vmiou)
                print("val_mean_iou:",v_miou)                    
            
#                    result_string = "{}: {:.4f} \n".format(i, result["TP"][i] / result["TA"][i])
#                    print(result_string)
#                    vmiou += result["TP"][i] / result["TA"][i]
#                vmiou = vmiou / 7
#                print("val_mean_iou:", vmiou)

        epoch_str = ('Epoch: {},  Train Mean IoU: {:.5f},  Valid Mean IU: {:.5f} '.format(e, t_miou,v_miou))    
        print(epoch_str)  

# =============================================================================
# Generate_List
# =============================================================================
def Generate_List(Val_Rate,Test_Rate):
    data_list, label_list = read_images()
    
    val_num = int(Val_Rate * len(data_list))
    test_num = int(Test_Rate * len(data_list))
    train_num=int(len(data_list)-val_num-test_num)
    
    train_data_list = data_list[val_num:train_num+val_num]
    train_label_list = label_list[val_num:train_num+val_num]
    save_train = pd.DataFrame({'image': train_data_list,'label': train_label_list})
    save_train_shuffle = shuffle(save_train)
    save_train_shuffle.to_csv('./img_list_csv/train.csv', index=False)

    val_data_list = data_list[:val_num]
    val_label_list = label_list[:val_num]
    save_val = pd.DataFrame({'image': val_data_list, 'label': val_label_list})
    save_val_shuffle = shuffle(save_val)
    save_val_shuffle.to_csv('./img_list_csv/val.csv', index=False)
    

    test_data_list = data_list[train_num+val_num:]
    test_label_list = label_list[train_num+val_num:]
    save_test = pd.DataFrame({'image': test_data_list, 'label': test_label_list})
    save_test_shuffle = shuffle(save_test)
    save_test_shuffle.to_csv('./img_list_csv/test.csv', index=False)    
   
    return


if __name__ == "__main__":

    Val_Rate = 0.2
    Test_Rate = 0.1
    warnings.filterwarnings("ignore")
    # 实例化数据集
    if not (os.path.isfile('./img_list_csv/val.csv')
            or os.path.isfile('./img_list_csv/train.cs')
            or os.path.isfile('./img_list_csv/test.cs')
            ):
        Generate_List(Val_Rate,Test_Rate)
        
    '''
    List_Gen = input(
        "Input 'y' to generate the CSV list,others skip:")  #设置是否需要重新生成类别
    if List_Gen == 'y':
        Generate_List(Val_Rate,Test_Rate)
        print("Complete generating the CSV list")
    '''

    train_list = pd.read_csv(os.path.join(os.getcwd(), "img_list_csv", "train.csv"),
                             header=None,names=["image", "label"])
    train_data_list = train_list["image"].values[1:]
    train_label_list = train_list["label"].values[1:]

    val_list = pd.read_csv(os.path.join(os.getcwd(), "img_list_csv","val.csv"),
                           header=None,names=["image", "label"])
    val_data_list = val_list["image"].values[1:]
    val_label_list = val_list["label"].values[1:]

    test_list = pd.read_csv(os.path.join(os.getcwd(), "img_list_csv","test.csv"),
                           header=None,names=["image", "label"])
    test_data_list = test_list["image"].values[1:]
    test_label_list =test_list["label"].values[1:]
    
    #crop_shape = (512,1536)   
    crop_shape = [768, 256]    
    #crop_shape = (256, 768)
    offset = 690

    train_Dataset = BaiduLane_Dataset(train_data_list, train_label_list,offset, crop_shape)
    #print("len(train_data_list):",len(train_data_list))
    val_Dataset = BaiduLane_Dataset(val_data_list, val_label_list, offset, crop_shape)
    
    test_Dataset = BaiduLane_Dataset(test_data_list, test_label_list, offset,crop_shape)

    batch_size = 2
    train_data = DataLoader(train_Dataset, batch_size,shuffle=True,num_workers=2, drop_last=True)
    val_data = DataLoader(val_Dataset,batch_size,shuffle=False,num_workers=2,drop_last=True)
    test_data = DataLoader(test_Dataset, batch_size, shuffle=False,num_workers=2,drop_last=True)

    #调用模型
    n_classes = 8

    net = UNetWithResnet50Encoder(n_classes)  #8个分类

    #criterion=nn.BCEWithLogitsLoss()
    Mode_Setting = input("Please input a number,'0' for train and 'else' for test:")  #设置模式 训练/测试

    if Mode_Setting == '0':
        #训练及验证   

        epoches =6
        #optimizer = torch.optim.Adam(net.parameters(), lr= 0.0007, weight_decay=1e-2)
        optimizer = torch.optim.AdamW(net.parameters(),lr=0.0006,weight_decay=1e-2)
        # Learning rate scheduler
        #scheduler = PolynomialLR(optimizer=optimizer,iter_max=epoches,power=Config.POLY_POWER,)
        if (os.path.isfile('./unet_baidulane.pth')):
                net.load_state_dict(torch.load('./unet_baidulane.pth',map_location='cpu'))      
        Train_Val(epoches, net, train_data,val_data)

    else:
        #测试
        path=["./predict/pred","./predict/label"] #生成两个文件夹存放 预测图片及处理后的原图片
        for k in path:
            if not (os.path.exists(k)):os.makedirs(k)    
        net.load_state_dict(torch.load('./unet_baidulane.pth',map_location='cpu'))
        net = net.cuda()
        net = net.eval()
        test_all_miou =[]
        acc_all=[]
        process = tqdm(test_data)
        j=0
        for data in process:
            
            with torch.no_grad():
                im = Variable(data[0].cuda())
                label = Variable(data[1].cuda())
               #label_1 = Variable(data[2].cuda())
            # forward          
            out = net(im)
            
            pred = torch.argmax(F.softmax(out, dim=1), dim=1)
            mask = torch.argmax(F.softmax(label, dim=1), dim=1)
            result = {
                     "TP": {i: 0 for i in range(8)},
                     "TA": {i: 0 for i in range(8)}
                     }
            result = compute_iou(pred, mask)
            test_miou = [] 
            for i in range(1, 8):
                if result["TA"][i] !=0: 
                    test_miou_i=result["TP"][i] / result["TA"][i]
                    test_miou.append(test_miou_i)     
            print("test_mean_iou:",np.mean(test_miou))       
            test_all_miou.append(np.mean(test_miou))
            
            TP_sum=[]
            TA_sum=[]
            for i,j in result["TP"].items():
                TP_sum.append(j)
            for i,j in result["TA"].items():
                TA_sum.append(j)
            TP_sum=np.array(TP_sum)
            TA_sum=np.array(TA_sum)
            TP_sum=TP_sum[1:].sum()
            TA_sum=TA_sum[1:].sum()
            acc='%.5f' %(TP_sum/TA_sum)
            print("acc:",acc)
            acc_all.append(acc)   
            

            pred = pred.detach().cpu().numpy()
            mask = mask.detach().cpu().numpy()
#            print("pred.shape:",pred.shape) 
#            print("mask.shape:",mask.shape)

            for k in range(batch_size):
                pred_=pred[k,:]
                pred_ = decode_labels(pred_)
                mask_=mask[k,:]
#                print("pred_.shape:",pred_.shape)
#                print("mask_.shape:",mask_.shape)
                mp.imsave('./predict/pred/' + str(j*batch_size+k) + '.png', pred_)
                mp.imsave('./predict/label/' + str(j*batch_size+k) + '.png', mask_)
            j=j+1

        print("test_miou_Avr:", np.mean(test_all_miou))  
        acc_all = np.array(acc_all, dtype=np.float32)
        print("acc_Avr:", np.mean(acc_all))



