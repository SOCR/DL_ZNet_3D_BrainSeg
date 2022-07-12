#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: maottom
"""

import torch
import torch.nn as nn
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import nibabel as nib
from skimage.transform import resize
from albumentations import Compose
import matplotlib.pyplot as plt

class TestCustumDataset(Dataset):
    def __init__(self, img_path, mask_path,classs, is_resize: bool=True):
        self.mask_path = mask_path
        self.img_path = img_path
        self.is_resize = is_resize
        self.classs=classs
    
    
    def __len__(self):
        return 1
    def __getitem__(self,idx):
        #load image
        try:
            img = self.load_img(self.img_path)
        except:
            self.img_path=self.img_path+".gz"
            img = self.load_img(self.img_path)
            
            
        if self.is_resize:
            img = self.resize(img)
            img = self.normalize(img)
        img = np.stack([img])
        img = np.moveaxis(img, (0, 1, 2, 3), (0, 3, 2, 1))
        
        #load mask
        try:
            mask = self.load_img(self.mask_path)
        except:
            self.mask_path=self.mask_path+".gz"
            mask = self.load_img(self.mask_path)
            
            
        if self.is_resize:
            mask = self.resize(mask)
                
        mask = self.preprocess_mask_labels(mask,self.classs)
            
        
        return {
                "image": img.astype(np.float32),
                "mask": mask.astype(np.float32),
                "img_path":self.img_path,
                "seg_path":self.mask_path
                }

    
    def load_img(self, file_path):
        data = nib.load(file_path)
        data = np.asarray(data.dataobj)
        return data
    
    def normalize(self, data: np.ndarray):
        data_min = np.min(data)
        return (data - data_min) / (np.max(data) - data_min)
    
    def resize(self, data: np.ndarray):
        data = resize(data, (128, 128, 128), preserve_range=True, order=0)
        
        return data
    
    def preprocess_mask_labels(self, mask: np.ndarray,classs):
        print(np.unique(mask,return_counts=True))
        #mask[mask == 3] = 4
        print(classs)
        
        if classs=="WT":
            mask = mask.copy()
            mask[mask == 1] = 1
            mask[mask == 2] = 1
            mask[mask == 3] = 1
            mask[mask == 4] = 1
        
        if classs=="TC":
            mask = mask.copy()
            mask[mask == 1] = 1
            mask[mask == 2] = 0
            mask[mask == 3] = 0
            mask[mask == 4] = 1
            
        if classs=="ET":
            mask = mask.copy()
            mask[mask == 1] = 0
            mask[mask == 2] = 0
            mask[mask == 3] = 0
            mask[mask == 4] = 1
        
        #mask = np.stack([mask_WT,mask_TC,mask_ET])
        mask = np.stack([mask])
        
        mask = np.clip(mask, 0, 1)
        print(np.unique(mask,return_counts=True))
        
        mask = np.moveaxis(mask, (0, 1, 2, 3), (0, 3, 2, 1))
        print("mask shape in preprocess: ",mask.shape)
        return mask
    


def compute_results(model,
                    sample,
                    treshold=0.5):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = {"image": [] , "GT": [] ,"Prediction": [] ,"dice": [] }

    with torch.no_grad():
        #for i, data in enumerate(dataloader):
            imgs, targets = torch.Tensor(sample['image']), torch.Tensor(sample['mask'])
            imgs=imgs[None,:,:,:,:]
            targets=targets[None,:,:,:,:]
            print("img shape ", imgs.shape)
            print("targets shape ", targets.shape)
            imgs, targets = imgs.to(device), targets.to(device)
            logits = model(imgs)
            probs = torch.sigmoid(logits)
            predictions = (probs >= treshold).float()
            #print("prediction ",predictions.shape)
            predictions =  predictions.cpu()
            print("pred shape ",predictions.shape)
            targets = targets.cpu()
            #predictions=post_process(predictions)
            #targets=post_process(targets)
            #dice=dice_coef_metric(predictions,targets)
            #jaccard=jaccard_coef_metric(predictions,targets)
            
            results["image"].append(imgs.cpu())
            results["GT"].append(targets)
            results["Prediction"].append(predictions)
            #results["dice"].append(dice)
            #results["jaccard"].append(jaccard)

    return results


def display_monatge_images(img,gt,prediction):
    img_=img.squeeze().cpu().detach().numpy()
    gt_=gt.squeeze().cpu().detach().numpy()
    prediction_=prediction.squeeze().cpu().detach().numpy()
    print("img shape ", img_.shape)
    print("gt shape ", gt_.shape)
    print("pred shape ", prediction_.shape)

    from skimage.util import montage
    gt_montaged =np.rot90(montage(gt_))
    img_montaged =np.rot90(montage(img_))
    prediction_montage=np.rot90(montage(prediction_))

    fig, ax = plt.subplots(1, 1, figsize = (50, 50))
    ax.imshow(img_montaged)
    ax.imshow(np.ma.masked_where(gt_montaged == False, gt_montaged),cmap='Greys', alpha=0.6)
    #ax.imshow(np.ma.masked_where(prediction_montage == False, prediction_montage),cmap='cool', alpha=0.6)
    
    fig, ax = plt.subplots(1, 1, figsize = (50, 50))
    ax.imshow(img_montaged)
    ax.imshow(np.ma.masked_where(prediction_montage == False, prediction_montage),cmap='Greys', alpha=0.6)
    #ax.imshow(np.ma.masked_where(prediction_montage == False, prediction_montage),cmap='cool', alpha=0.6)
    
    fig, ax = plt.subplots(1, 1, figsize = (50, 50))
    ax.imshow(img_montaged)
    ax.imshow(np.ma.masked_where(gt_montaged == False, gt_montaged), alpha=0.6,cmap='Greys')
    ax.imshow(np.ma.masked_where(prediction_montage == False, prediction_montage), alpha=0.6,cmap='hsv')

def display_roi(img,pred,bg_image_ni,affine_ni,cut_coords=0):
    #cut_coords=(-100,100,100)
    img=img.squeeze().cpu().detach().numpy()
    img = resize(img, (155, 240, 240),preserve_range=True)
    img=np.moveaxis(img,(0, 1, 2), (2, 1, 0))
    img_ni=nib.Nifti1Image(img,affine_ni.affine)
    
    pred=pred.squeeze().cpu().detach().numpy()
    pred = resize(pred, (155, 240, 240),preserve_range=True)
    pred=np.moveaxis(pred,(0, 1, 2), (2, 1, 0))
    pred_ni=nib.Nifti1Image(pred,affine_ni.affine)
    
    a=nlplt.plot_roi(img_ni, title='',bg_img=bg_image_ni,cmap='Greys') 
    print(a.cut_coords)

    nlplt.plot_roi(pred_ni, title='',bg_img=bg_image_ni,cmap='Greys',cut_coords=a.cut_coords)
    
    nlplt.plot_roi(pred_ni, title='',bg_img=img_ni,cmap='cool',cut_coords=a.cut_coords)
    
    
    

i_path="../BraTS2020_TrainingData/BraTS20_Training_004/BraTS20_Training_004_flair.nii" 
m_path="../BraTS2020_TrainingData/BraTS20_Training_004/BraTS20_Training_004_seg.nii" 
import nilearn.plotting as nlplt
import nilearn as nl
img_ni=nl.image.load_img(i_path)
mask_ni=nl.image.load_img(m_path)

for i in ['WT']:
#for i in ['WT','TC','ET']:
    
    sample_ds=TestCustumDataset(i_path,m_path,i) #add i
    
    from new05_znet3d import znet3d
    #TC model
    model = znet3d(in_channels=1, n_classes=1, n_channels=24).to('cuda')
    model=nn.DataParallel(model)
    if i=="WT":
        model.load_state_dict(torch.load("best_model_WT.pth"))  
    if i=="TC":
        model.load_state_dict(torch.load("best_model_TC.pth")) 
    if i=="ET":
        model.load_state_dict(torch.load("best_model_ET.pth")) 
    model.eval()
    
    sample=sample_ds[0]
    
    print(sample["image"].shape)
    
    results=compute_results(model,sample,0.3)
    
    img=results['image'][0][0]
    gt=results['GT'][0][0]
    prediction=results['Prediction'][0][0]
    
    print(prediction.shape)
    display_monatge_images(img,gt,prediction)
    
    display_roi(gt,prediction,img_ni,mask_ni)
    #display_roi(prediction,img_ni,mask_ni,cut_coords=(-146,86,88))




    