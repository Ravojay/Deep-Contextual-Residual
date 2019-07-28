#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 13:39:34 2019

@author: dyp
"""

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from copy import deepcopy
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random
from torchvision import transforms
from PIL import Image
import torchvision.transforms.functional as TF

def clone(layer,n):
    return nn.ModuleList([deepcopy(layer) for _ in range(n)])

class LayerNorm(nn.Module):
    def __init__(self,channel_dim,epsilon = 1e-6):
        super(LayerNorm,self).__init__()
        self.a = nn.Parameter(torch.ones(channel_dim,1,1))
        self.b = nn.Parameter(torch.zeros(channel_dim,1,1))
        self.eps = epsilon
        
    def forward(self,tensor):
        mean = tensor.mean(1,keepdim=True)
        std = tensor.std(1,keepdim=True)
        return self.a*(tensor-mean)/(std+self.eps)+self.b


class Preparor(nn.Module):
    def __init__(self,input_size = 512):
        super(Preparor,self).__init__()
        self.Conv_1 = nn.Conv2d(1,64,kernel_size=3,stride=1,padding=(1,1))
        self.Pool_1 = nn.MaxPool2d(kernel_size=3,stride=2)
        self.input_size = input_size
        self.norm = LayerNorm(64)
        
    def forward(self,input_tensor):
        conved = self.Conv_1(input_tensor)
        conved = F.elu(self.norm(conved))
        padded = F.pad(conved,(1,0,1,0))
        return self.Pool_1(padded)

class Sublayer_connection(nn.Module):
    def __init__(self,input_channel,output_channel,input_size,out_size):
        super(Sublayer_connection,self).__init__()
        self.channel = input_channel
        self.norm = LayerNorm(output_channel)
        self.cross_dim_res = None
        self.change = False
        if input_channel != output_channel and input_size != out_size:
            self.cross_dim_res = nn.Conv2d(input_channel,output_channel,kernel_size=1,stride=2)
            self.change = True
        elif input_channel != output_channel:
            self.cross_dim_res = nn.Conv2d(input_channel,output_channel,kernel_size=1)
            self.change = True

    def forward(self,x,sublayer):
        if not self.change:
            return F.elu(self.norm(x+sublayer(x)))
        else:
            transformed_res = self.cross_dim_res(x)
            sub = sublayer(x)
            return F.elu(self.norm(transformed_res+sub))


class Res_unit(nn.Module):
    def __init__(self,input_channel,out_channel,first_stride,second_stride,dialat=1):
        super(Res_unit,self).__init__()
        if first_stride==2:
            self.first_padding = (1,0,1,0)
            self.second_padding = (1,1,1,1)
        else:
            self.first_padding = (1,1,1,1)
            self.second_padding = (1,1,1,1)
        if dialat!=1:
            self.first_padding = (2,2,2,2)
            self.second_padding = (2,2,2,2)
        self.dialat = dialat
        self.first_conv = nn.Conv2d(input_channel,out_channel,kernel_size=3,\
                                    stride=first_stride,dilation=dialat)

        self.second_conv = nn.Conv2d(out_channel,out_channel,kernel_size=3,\
                                     stride=second_stride,dilation=dialat)           
        self.norm = LayerNorm(out_channel)
        
    def forward(self,tensor):
        conved = self.first_conv(F.pad(tensor,(self.first_padding)))
        normed = F.elu(self.norm(conved))
        return self.second_conv(F.pad(normed,self.second_padding))
    

class Res_unit_b4(nn.Module):
    def __init__(self,input_channel,mid_out_channel,out_channel):
        super(Res_unit_b4,self).__init__()
        self.padding = (2,2,2,2)
        self.first_conv = nn.Conv2d(input_channel,mid_out_channel,kernel_size=3,\
                                    dilation=2)
        self.second_conv = nn.Conv2d(mid_out_channel,out_channel,kernel_size=3,\
                                     dilation=2)
            
        self.norm = LayerNorm(mid_out_channel)
        
    def forward(self,tensor):
        conved = self.first_conv(F.pad(tensor,self.padding))
        normed = F.elu(self.norm(conved))
        return self.second_conv(F.pad(normed,self.padding))


class Block_4(nn.Module):
    def __init__(self):
        super(Block_4,self).__init__()
        self.sub_block_1 = Res_unit_b4(512,512,1024)
        self.trans = Sublayer_connection(512,1024,64,64)
        self.sub_block_2 = clone(Res_unit_b4(1024,512,1024),2)
        self.sub_connects = clone(Sublayer_connection(1024,1024,64,64),2)
    
    def forward(self,tensor):
        first = self.trans(tensor,self.sub_block_1)
        for i in range(len(self.sub_block_2)):
            first = self.sub_connects[i](first,self.sub_block_2[i])
        return first
    
    
class Block_1(nn.Module):
    def __init__(self):
        super(Block_1,self).__init__()
        self.sub_block_1 = Res_unit(64,128,2,1)
        self.trans = Sublayer_connection(64,128,256,128)
        self.sub_block_2 = clone(Res_unit(128,128,first_stride=1,second_stride=1),2)
        self.sub_connects = clone(Sublayer_connection(128,128,128,128),2)
        
    def forward(self,tensor):
        first_conved = self.trans(tensor,self.sub_block_1)
        for i in range(len(self.sub_block_2)):
            first_conved = self.sub_connects[i](first_conved,self.sub_block_2[i])
        return first_conved
 
    
class Block_2(nn.Module):
    def __init__(self):
        super(Block_2,self).__init__()
        self.sub_block1 = Res_unit(128,256,first_stride=2,second_stride=1)
        self.sub_block2 = clone(Res_unit(256,256,first_stride=1,second_stride=1),2)
        self.trans = Sublayer_connection(128,256,128,64)
        self.sub_connects = clone(Sublayer_connection(256,256,64,64),2)
        
    def forward(self,tensor):
        first = self.trans(tensor,self.sub_block1)
        for i in range(len(self.sub_block2)):
            first = self.sub_connects[i](first,self.sub_block2[i])
        return first
 
    
class Block_3(nn.Module):
    def __init__(self):
        super(Block_3,self).__init__()
        self.sub_block1 = Res_unit(256,512,1,1,dialat=2)
        self.trans = Sublayer_connection(256,512,64,64)
        self.sub_block2 = clone(Res_unit(512,512,1,1,dialat=2),5)
        self.sub_connects = clone(Sublayer_connection(512,512,64,64),5)
        
    def forward(self,tensor):
        first = self.trans(tensor,self.sub_block1)
        for i in range(len(self.sub_block2)):
            first = self.sub_connects[i](first,self.sub_block2[i])
        return first


class Block5_6res(nn.Module):
    def __init__(self,input_channel):
        super(Block5_6res,self).__init__()
        self.padding = (2,2,2,2)
        self.first_conv = nn.Conv2d(input_channel,input_channel//2,1)
        self.second_conv = nn.Conv2d(input_channel//2,input_channel,3,dilation=2)
        self.third_conv = nn.Conv2d(input_channel,input_channel*2,1)
        self.norm1 = LayerNorm(input_channel//2)
        self.norm2 = LayerNorm(input_channel)
        
    def forward(self,tensor):
        first = F.elu(self.norm1(self.first_conv(tensor)))
        second = F.elu(self.norm2(self.second_conv(F.pad(first,self.padding))))
        return self.third_conv(second)
    

class Block_5(nn.Module):
    def __init__(self):
        super(Block_5,self).__init__()
        self.sub_connect = Sublayer_connection(1024,2048,64,64)
        self.trans = Block5_6res(1024)
    
    def forward(self,tensor):
        return self.sub_connect(tensor,self.trans)
    
        
class Block_6(nn.Module):
    def __init__(self):
        super(Block_6,self).__init__()
        self.sub_connect = Sublayer_connection(2048,4096,64,64)
        self.trans = Block5_6res(2048)
    
    def forward(self,tensor):
        return self.sub_connect(tensor,self.trans)
    

class Expansive_path(nn.Module):
    def __init__(self):
        super(Expansive_path,self).__init__()
        self.first_conv = nn.Conv2d(64,32,3,padding=(1,1))
        self.second_conv = nn.Conv2d(32,16,3,padding=(1,1))
        self.norm1 = LayerNorm(32)
        self.norm2 = LayerNorm(16)
        #self.dropout1 = nn.Dropout2d()
        #self.dropout2 = nn.Dropout2d()
        
    def forward(self,tensor):
        droped = F.elu(self.norm1(self.first_conv(tensor)))
        return F.elu(self.norm2(self.second_conv(droped)))


class Output(nn.Module):
    def __init__(self):
        super(Output,self).__init__()
        self.conv = nn.Conv2d(16,1,1)
        
    def forward(self,tensor):
        res = self.conv(tensor)
        return torch.squeeze(res,dim=1)
    
    
class CRN(nn.Module):
    def __init__(self,Pre,Block1,Block2,Block3,Block4,Block5,Block6,expansive,output):
        super(CRN,self).__init__()
        self.blocks = nn.ModuleList([Block1,Block2,Block3,Block4,Block5,Block6])
        self.Deconv_downsample = nn.ConvTranspose2d(64,64,4,stride=2,padding=(1,1))
        self.norm = LayerNorm(64)
        self.Deconv_Block1 = nn.ConvTranspose2d(128,64,8,4,padding=(2,2))
        self.Deconv = nn.ConvTranspose2d(4096,64,16,8,padding=(4,4))
        self.Pre = Pre
        self.expansive = expansive
        self.output_layer = output
        
    def forward(self,tensor):
        prepared = self.Pre(tensor)
        copy_pre = prepared
        block1_d = self.blocks[0](prepared)
        copy_block1_d = block1_d
        for i in range(1,len(self.blocks)):
            block1_d = self.blocks[i](block1_d)
        res_pre = self.Deconv_downsample(copy_pre)
        res_1_d = self.Deconv_Block1(copy_block1_d)
        res_fin = self.Deconv(block1_d)
        tens_for_expan = res_pre+res_1_d+res_fin
        tens_for_expan = F.elu(self.norm(tens_for_expan))
        expansed = self.expansive(tens_for_expan)
        return self.output_layer(expansed)


def get_one_image():
    cur_path = os.getcwd()
    data_file = cur_path+'/data/images'
    label_file = cur_path+'/data/labels'
    data_images = sorted(os.listdir(data_file))
    label_images = sorted(os.listdir(label_file))
    if data_images[0] == '.DS_Store':
        data_images = data_images[1:]
    first_image_name = data_file+'/'+data_images[0]
    img = cv2.imread(first_image_name,0)
    return img


class EMDataset_val_test(Dataset):
    def __init__(self,data_names,label_names):
        self.data_names = data_names
        self.label_names = label_names
        
    def __len__(self):
        return len(self.data_names)
    
    def __getitem__(self,idx):
        data_img = cv2.imread(self.data_names[idx],0)
        data_img = np.expand_dims(data_img,axis=3)
        data_img = data_img/255
        data_img = torch.tensor(data_img).float().permute(2,0,1)
        
        label_img = cv2.imread(self.label_names[idx],0)
        _,label_img = cv2.threshold(label_img,127,255,cv2.THRESH_BINARY)
        label_img = label_img/255
        label_img = torch.tensor(label_img).float()
        return self.data_names[idx],data_img,label_img
    
class EMDataset(Dataset):
    def __init__(self,data_names,label_names):
        self.data_names = data_names
        self.label_names = label_names
        
    def __len__(self):
        return len(self.data_names)
    
    def transform(self,image,label):
        mode = random.random()
        if mode<0.5:
            return image,label
        image,label = Image.fromarray(image),Image.fromarray(label)
        if mode>=0.5:
            i,j,h,w = transforms.RandomResizedCrop.get_params(image,scale=(0.5,1.0),ratio=(1,1))
            image = TF.resized_crop(image,i,j,h,w,512,interpolation=Image.BICUBIC)
            label = TF.resized_crop(label,i,j,h,w,512,interpolation=Image.BICUBIC)
# =============================================================================
#         if mode>=0.75:
#             angle = transforms.RandomRotation.get_params([180,-180])
#             image = TF.rotate(image,angle)
#             label = TF.rotate(label,angle)
#             image,label = TF.center_crop(image,256),TF.center_crop(label,256)
#             image,label = TF.resize(image,512,interpolation=Image.BICUBIC),TF.resize(label,512,interpolation=Image.BICUBIC)
# =============================================================================
        return np.asarray(image),np.asarray(label)
    
    def __getitem__(self,idx):
        data_img = cv2.imread(self.data_names[idx],0)
        label_img = cv2.imread(self.label_names[idx],0)
        data_img,label_img = self.transform(data_img,label_img)
        
        data_img = np.expand_dims(data_img,axis=3)
        data_img = data_img/255
        data_img = torch.tensor(data_img).float().permute(2,0,1)
        
        _,label_img = cv2.threshold(label_img,127,255,cv2.THRESH_BINARY)
        label_img = label_img/255
        label_img = torch.tensor(label_img).float()
        return self.data_names[idx],data_img,label_img


def create_model():
    Pre = Preparor()
    Block1 = Block_1()
    Block2 = Block_2()
    Block3 = Block_3()
    Block4 = Block_4()
    Block5 = Block_5()
    Bolck6 = Block_6()
    expansive = Expansive_path()
    output = Output()
    return CRN(Pre,Block1,Block2,Block3,Block4,Block5,Bolck6,expansive,output) 


def train(epoches,batch_size):
    #read data
    root = os.getcwd()
    data_file = root+'/data/images'
    label_file = root+'/data/labels'
    
    data_images = sorted(os.listdir(data_file))
    label_images = sorted(os.listdir(label_file))
    if '.DS_Store' == data_images[0]:
        data_images = data_images[1:]
    if '.DS_Store' == label_images[0]:
        label_images = label_images[1:]
    data_image_names = list(map(lambda x:data_file+'/'+x,data_images))  
    label_image_names = list(map(lambda x:label_file+'/'+x,label_images))
    
    test_data_file = root+'/test_data'
    test_label_file = root+'/test_label'
    test_data_images = sorted(os.listdir(test_data_file))
    test_label_images = sorted(os.listdir(test_label_file))
    if '.DS_Store' == test_data_images[0]:
        test_data_images = test_data_images[1:]
    if '.DS_Store' == test_label_images[0]:
        test_label_images = test_label_images[1:]
    test_data_names = list(map(lambda x:test_data_file+'/'+x,test_data_images))
    test_label_names = list(map(lambda x:test_label_file+'/'+x,test_label_images)) 
    #make dataset
    train_data,val_data,train_label,val_label = train_test_split(data_image_names,label_image_names,test_size=0.1)
    
    Em_train_dataset = EMDataset(train_data,train_label)
    train_loader = DataLoader(Em_train_dataset,batch_size=batch_size,shuffle=True)
    
    Em_val_dataset = EMDataset_val_test(val_data,val_label)
    val_loader = DataLoader(Em_val_dataset,batch_size=batch_size)
    
    Em_test_dataset = EMDataset_val_test(test_data_names,test_label_names)
    test_loader = DataLoader(Em_test_dataset,batch_size=batch_size)
    
    #make model optimizer and loss_func
    model = create_model()
    for parameter in model.parameters():
        if parameter.dim()>1:
            nn.init.kaiming_normal_(parameter)
    model = model.float()
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss(reduction='none').to(device)
    optimizer = torch.optim.Adamax(model.parameters())
    
    min_loss = np.inf
    save_dict = {}
    
    train_loss_list = []
    val_loss_lost = []
    x_axis = list(range(epoches))
    
    #train
    train_size = len(data_image_names)
    max_acc = torch.tensor(0).float().to(device)
    for epoch in range(epoches):
        #training step for 1 epoch
        training_loss = torch.tensor(0).float().to(device)
        total_pixel = torch.tensor(0).float().to(device)
        hit_pixel = torch.tensor(0).float().to(device)
        count = torch.tensor(0).float().to(device)
        count_v = 0
        for _,train_x,train_y in tqdm(train_loader):
            train_x = train_x.to(device)
            train_y = train_y.to(device)
            output = model(train_x)
            loss = criterion(output,train_y)
            zeros = ((train_y==0)+0).float()
            neg_weight = 1.5
            pix_weight = neg_weight*zeros+train_y
            pix_wise_loss = torch.mean(loss*pix_weight)
            
            count+=train_x.shape[0]
            count_v += 1
            training_loss += pix_wise_loss*train_x.shape[0]
            total_pixel += torch.tensor(512*512*batch_size).float().to(device)
            hit = ((torch.sigmoid(output)>=0.5).float()==train_y).float().to(device)
            hit_pixel += torch.sum(hit)
# =============================================================================
#             pix_wise_loss.backward()
#             optimizer.step()
#             optimizer.zero_grad()
# =============================================================================
            if count_v==train_size:
                pix_wise_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            else:
                if count%2 == 0:
                    pix_wise_loss = pix_wise_loss/2
                    pix_wise_loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                else:
                    pix_wise_loss = pix_wise_loss/2
                    pix_wise_loss.backward()                    
                
        m_loss = training_loss/count
        acc = hit_pixel/total_pixel
        print('train_loss:{d}'.format(d=m_loss.data.cpu().numpy()))
        print('train_accuracy: {d}'.format(d=acc.data.cpu().numpy()))
        train_loss_list.append(m_loss.data.cpu().numpy())
        #test on validation set
        with torch.no_grad():
            total_pixel = torch.tensor(0).float().to(device)
            hit_pixel = torch.tensor(0).float().to(device)
            total_loss = torch.tensor(0).float().to(device)
            count = torch.tensor(0).float().to(device)
            for _,val_x,val_y in val_loader:
                val_x,val_y = val_x.to(device),val_y.to(device)
                count+=val_x.shape[0]
                output = model(val_x)
                loss = criterion(output,val_y)
                zeros = ((val_y==0)+0).float()
                neg_weight = 1.5
                pix_weight = neg_weight*zeros+val_y
                pix_wise_loss = torch.mean(loss*pix_weight)
                total_loss += pix_wise_loss*val_x.shape[0]
                #for accuracy
                total_pixel += torch.tensor(512*512*batch_size).float().to(device)
                hit = ((torch.sigmoid(output)>=0.5).float()==val_y).float().to(device)
                hit_pixel+=torch.sum(hit)
            mean_loss = total_loss/count
            acc = hit_pixel/total_pixel
            if mean_loss<min_loss:
                save_dict['model_parameters'] = model.state_dict()
                save_dict['hyper_params'] = optimizer.state_dict()
                torch.save(save_dict,'{d}_epoch_parameters_{f}.tar.gz'.format(f=np.round(mean_loss.data.cpu().numpy(),3),d=epoch))
                min_loss = mean_loss
            if acc > max_acc:
                save_dict['model_parameters'] = model.state_dict()
                save_dict['hyper_params'] = optimizer.state_dict()
                torch.save(save_dict,'acc_{e}_{d}_epoch_parameters_{f}.tar.gz'.format(f=np.round(mean_loss.data.cpu().numpy(),3),d=epoch,e=acc))
                max_acc = acc
            print('val_loss: {d}'.format(d=mean_loss.data.cpu().numpy()))
            print('val_accuracy: {d}'.format(d=acc.data.cpu().numpy()))
            val_loss_lost.append(mean_loss.data.cpu().numpy())
    #test on test set
    print('Testing on test set')
    with torch.no_grad():
        total_pixel = torch.tensor(0).float().to(device)
        hit_pixel = torch.tensor(0).float().to(device)
        total_loss = torch.tensor(0).float().to(device)
        count = torch.tensor(0).float().to(device)
        for pic_name,test_x,test_y in test_loader:
            test_x,test_y = test_x.to(device),test_y.to(device)
            count+=test_x.shape[0]
            output = model(test_x)
            loss = criterion(output,test_y)
            zeros = ((test_y==0)+0).float()
            neg_weight = 1.5
            pix_weight = neg_weight*zeros+test_y
            pix_wise_loss = torch.mean(loss*pix_weight)
            total_loss += pix_wise_loss*test_x.shape[0]
            #for accuracy
            total_pixel += torch.tensor(512*512*batch_size).float().to(device)
            hit = ((torch.sigmoid(output)>=0.5).float()==test_y).float().to(device)
            hit_pixel+=torch.sum(hit)
        mean_loss = total_loss/count
        acc = hit_pixel/total_pixel
        print('mean_loss on test set: {d}'.format(d=mean_loss.data.cpu().numpy()))
        print('accuracy on test set: {d}'.format(d=acc.data.cpu().numpy()))   
    plt.figure(figsize=(10,10))
    plt.plot(x_axis,train_loss_list)
    plt.plot(x_axis,val_loss_lost)
    plt.show()
            
    
if __name__ == '__main__':    
    train(40,1)






         