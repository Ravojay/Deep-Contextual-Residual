#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 15:55:04 2019

@author: dyp
"""

import cv2
import os
from sklearn.model_selection import train_test_split


def create(split):
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
    
    #train_data,test_data,train_label,test_label = train_test_split(data_image_names,label_image_names,test_size=split)
    os.mkdir('test_data')
    os.mkdir('test_label')    
    for i in range(len(data_image_names)):
        train_name = data_image_names[i].split('/')[-1]
        name = train_name.split('_')
        if len(name)==1:
            pic = cv2.imread(data_image_names[i],0)
            label = cv2.imread(label_image_names[i],0)
            pic_save_name = os.getcwd()+'/test_data/'+data_image_names[i].split('/')[-1]
            label_save_name = os.getcwd()+'/test_label/'+label_image_names[i].split('/')[-1]
            cv2.imwrite(pic_save_name,pic,[int(cv2.IMWRITE_JPEG_QUALITY),100])
            cv2.imwrite(label_save_name,label,[int(cv2.IMWRITE_JPEG_QUALITY),100])
            os.remove(data_image_names[i])
            os.remove(label_image_names[i])
if __name__ == '__main__':
    create(0.1)
        
        