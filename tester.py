#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 13:44:51 2019

@author: dyp
"""

import numpy as np
import cv2
import os
import torch
from torch.utils.data import DataLoader
from model_pixel_weight import create_model,EMDataset_val_test
from tqdm import tqdm



def predict_and_output():
    root = os.getcwd()
    model = create_model()
    model = model.float()
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    
    params = torch.load('acc_0.9347759485244751_14_epoch_parameters_0.14900000393390656.tar.gz',map_location=device)
    model_params = params['model_parameters']
    model.load_state_dict(model_params)
    model.eval()
    model.to(device)
    
    #print(hyper_prams)
    dir_suffix = 0
    while os.path.exists('Outputs_{d}'.format(d=dir_suffix)):
        dir_suffix+=1
    dir_name = 'Outputs_{d}'.format(d=dir_suffix)
    os.mkdir(dir_name)
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
    Em_test_dataset = EMDataset_val_test(test_data_names,test_label_names)
    test_loader = DataLoader(Em_test_dataset,batch_size=1)
    
    with torch.no_grad():
        for pic_name,test_x,test_y in tqdm(test_loader):
            pic_name = pic_name[0].split('/')[-1]
            test_x,test_y = test_x.to(device),test_y.to(device)
            output = torch.sigmoid(model(test_x))
            #for accuracy
            output = (output.cpu().numpy()>=0.5+0)*255
            output.astype(np.uint8)
            output = np.squeeze(output)
            save_path = os.getcwd()+'/'+dir_name+'/result_'+pic_name[6:]
            cv2.imwrite(save_path,output,[int(cv2.IMWRITE_JPEG_QUALITY),100])

if __name__ == '__main__':
    predict_and_output()

    
        
