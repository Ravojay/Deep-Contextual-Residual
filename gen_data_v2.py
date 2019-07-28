# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 08:37:45 2019

@author: 18846
"""

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from random import randint
from math import fabs,sin,cos,radians
from copy import deepcopy
cur_path = os.getcwd()
data_file = cur_path+'/data/images'
label_file = cur_path+'/data/labels'

def report_progress(count):
    if count%100 != 0:
        return
    print('{d} number of data generated'.format(d=count))

from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

def Elastic_transform(image, alpha, sigma,random_s=None):
    random_state = np.random.RandomState(random_s)
    shape = image.shape
    shape_size = shape[:2]
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha


    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))

    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)

def shift(img_o,distance,mode):
    img = deepcopy(img_o)
    if mode==0:
        temp = deepcopy(img[:,:distance])
        img[:,:-distance] = img[:,distance:]
        img[:,-distance:] = temp
        return img
    else:
        temp = deepcopy(img[:distance,:])
        img[:-distance,:] = img[distance:,:]
        img[-distance:,:] = temp
        return img


data_images = sorted(os.listdir(data_file))
label_images = sorted(os.listdir(label_file))
if '.DS_Store' == data_images[0]:
    data_images = data_images[1:]
if '.DS_Store' == label_images[0]:
    label_images = label_images[1:]

count = 0
for i in range(len(data_images)):
    cur_data_file_name = data_file+'/'+data_images[i]
    cur_label_file_name = label_file+'/'+label_images[i]

    data_img = cv2.imread(cur_data_file_name,0)
    label_img = cv2.imread(cur_label_file_name,0)
    
    h,w = data_img.shape[0],data_img.shape[1]
    center = (w//2,h//2)    
    
    for angle in range(0,360,90):
        M = cv2.getRotationMatrix2D(center, angle,1.0)
        
        data_rotated = cv2.warpAffine(data_img, M, (w, h))        
        label_rotated = cv2.warpAffine(label_img,M,(w,h))
        
        V_flip_rotated =  cv2.flip(data_rotated, 0)
        V_flip_rotated_label = cv2.flip(label_rotated, 0)
        
        if angle!=0:
            cv2.imwrite(data_file+'/'+data_images[i].split('.')[0]+'_r{d}.jpg'.format(d=angle),data_rotated,[int(cv2.IMWRITE_JPEG_QUALITY),100])
            cv2.imwrite(label_file+'/'+label_images[i].split('.')[0]+'_r{d}.jpg'.format(d=angle),label_rotated,[int(cv2.IMWRITE_JPEG_QUALITY),100])    
            count+=1
            report_progress(count)
        
        cv2.imwrite(data_file+'/'+data_images[i].split('.')[0]+'_r{d}_V.jpg'.format(d=angle),V_flip_rotated,[int(cv2.IMWRITE_JPEG_QUALITY),100])
        cv2.imwrite(label_file+'/'+label_images[i].split('.')[0]+'_r{d}_V.jpg'.format(d=angle),V_flip_rotated_label,[int(cv2.IMWRITE_JPEG_QUALITY),100])    
        count+=1
        report_progress(count)
        
        
        for j in range(5):
            h_dis = randint(20,210)
            shifted_h = shift(data_rotated,h_dis,0)
            shifted_h_label = shift(label_rotated,h_dis,0)
            cv2.imwrite(data_file+'/'+data_images[i].split('.')[0]+'_r{d}_sh{f}.jpg'.format(d=angle,f=j),shifted_h,[int(cv2.IMWRITE_JPEG_QUALITY),100])
            cv2.imwrite(label_file+'/'+label_images[i].split('.')[0]+'_r{d}_sh{f}.jpg'.format(d=angle,f=j),shifted_h_label,[int(cv2.IMWRITE_JPEG_QUALITY),100])    
            count+=1
            report_progress(count)
            
            v_dis = randint(20,210)
            shifted_v = shift(data_rotated,v_dis,1)
            shifted_v_label = shift(label_rotated,v_dis,1)
            cv2.imwrite(data_file+'/'+data_images[i].split('.')[0]+'_r{d}_sv{f}.jpg'.format(d=angle,f=j),shifted_v,[int(cv2.IMWRITE_JPEG_QUALITY),100])
            cv2.imwrite(label_file+'/'+label_images[i].split('.')[0]+'_r{d}_sv{f}.jpg'.format(d=angle,f=j),shifted_v_label,[int(cv2.IMWRITE_JPEG_QUALITY),100]) 
            count+=1
            report_progress(count)
            
            V_h_dis = randint(20,210)
            V_v_dis = randint(20,210)
            V_shifted_h = shift(V_flip_rotated,V_h_dis,0)
            V_shifted_h_label = shift(V_flip_rotated_label,V_h_dis,0)
            cv2.imwrite(data_file+'/'+data_images[i].split('.')[0]+'_r{d}_Vh{f}.jpg'.format(d=angle,f=j),V_shifted_h,[int(cv2.IMWRITE_JPEG_QUALITY),100])
            cv2.imwrite(label_file+'/'+label_images[i].split('.')[0]+'_r{d}_Vh{f}.jpg'.format(d=angle,f=j),V_shifted_h_label,[int(cv2.IMWRITE_JPEG_QUALITY),100]) 
            count+=1
            report_progress(count)
            
            V_shifted_v = shift(V_flip_rotated,V_v_dis,1)
            V_shifted_v_label = shift(V_flip_rotated_label,V_v_dis,1)
            cv2.imwrite(data_file+'/'+data_images[i].split('.')[0]+'_r{d}_Vv{f}.jpg'.format(d=angle,f=j),V_shifted_v,[int(cv2.IMWRITE_JPEG_QUALITY),100])
            cv2.imwrite(label_file+'/'+label_images[i].split('.')[0]+'_r{d}_Vv{f}.jpg'.format(d=angle,f=j),V_shifted_v_label,[int(cv2.IMWRITE_JPEG_QUALITY),100]) 
            count+=1
            report_progress(count)
            
            randstate = randint(0,100)
            distorted_data = Elastic_transform(data_rotated,100,8,randstate)
            distorted_label = Elastic_transform(label_rotated,100,8,randstate)
            cv2.imwrite(data_file+'/'+data_images[i].split('.')[0]+'_r{d}_dis{f}.jpg'.format(d=angle,f=j),distorted_data,[int(cv2.IMWRITE_JPEG_QUALITY),100])
            cv2.imwrite(label_file+'/'+label_images[i].split('.')[0]+'_r{d}_dis{f}.jpg'.format(d=angle,f=j),distorted_label,[int(cv2.IMWRITE_JPEG_QUALITY),100])    
            count+=1
            report_progress(count)
            
                  
