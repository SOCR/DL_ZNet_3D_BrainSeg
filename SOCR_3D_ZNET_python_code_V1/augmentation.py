#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: maottom
"""
import torch.nn as nn
import torch
import nilearn as nl
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import nilearn.plotting as nlplt
import random
import scipy
import os
import glob

def rotate_vol(img_np,mask_np,angle):
    '''
    axes = (1,0) The rotation plane is x-y plane.

    axes = (1,2) The rotation plane is y-z plane.

    axes = (2,0) The rotation plane is x-z plane.
    '''
    r1_img=scipy.ndimage.rotate(img_np, angle, axes=(0, 1), reshape=False, output=None, order=3, mode='nearest', cval=0.0, prefilter=False)
    r1_mask=scipy.ndimage.rotate(mask_np, angle, axes=(0, 1), reshape=False, output=None, order=3, mode='nearest', cval=0.0, prefilter=False)

    r2_img=scipy.ndimage.rotate(r1_img, angle, axes=(1, 2), reshape=False, output=None, order=3, mode='nearest', cval=0.0, prefilter=False)
    r2_mask=scipy.ndimage.rotate(r1_mask, angle, axes=(1, 2), reshape=False, output=None, order=3, mode='nearest', cval=0.0, prefilter=False)

    r3_img=scipy.ndimage.rotate(r2_img, angle, axes=(0, 2), reshape=False, output=None, order=3, mode='nearest', cval=0.0, prefilter=False)
    r3_mask=scipy.ndimage.rotate(r2_mask, angle, axes=(0, 2), reshape=False, output=None, order=3, mode='nearest', cval=0.0, prefilter=False)
    
    return r3_img, r3_mask

def display_image(img):
    from mpl_toolkits.axes_grid1 import ImageGrid
    #img_np=np.moveaxis(img_np, (0,1,2), (2,1,0))
    fig = plt.figure(figsize=(40,40))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(12, 13),  # creates 2x2 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.
                     )
    for i, ax in zip(range(img.shape[2]),grid):
        ax.imshow(img[:,:,i])
        
root_dir = '../BraTS2020_TrainingData'
out_dir="./augmentation/"
types = ["/*_flair.nii", "/*_seg.nii"]
image_mask=[]
count=0
for d in os.listdir(root_dir):
    d_path=os.path.join(root_dir,d)
    if os.path.isdir(d_path):
        count=count+1
        #print(d_path)
        for type in types:
            image_mask+=glob.glob(d_path+type)
        image_mask
        print(sorted(image_mask))
        if image_mask:
            new_floder_name=image_mask[0].split("/")[-2]
            print(new_floder_name)
            niimg = nl.image.load_img(image_mask[0])
            img_np=niimg.get_fdata()
            nimask = nl.image.load_img(image_mask[1])
            mask_np=nimask.get_fdata()
            augr10img,augr10mask=rotate_vol(img_np,mask_np, -5)
            #print(augr10img.shape,augr10mask.shape)

            try:
                new_path=os.path.join(out_dir, "Rotate-5_"+new_floder_name)
                image_file_path=os.path.join(new_path,new_floder_name+"_R-5_flair.nii.gz")
                mask_file_path=os.path.join(new_path,new_floder_name+"_R-5_seg.nii.gz")
                os.mkdir(new_path)
                #np.save(image_file_path,augr10img)
                #np.save(mask_file_path,augr10mask)
                new_ni_img = nib.Nifti1Image(augr10img, niimg.affine)
                nib.save(new_ni_img, image_file_path)
                #save new mask
                new_ni_mask = nib.Nifti1Image(augr10mask, nimask.affine)
                nib.save(new_ni_mask, mask_file_path)
            except OSError as error:
                print(error)
                new_ni_img = nib.Nifti1Image(augr10img, niimg.affine)
                nib.save(new_ni_img, image_file_path)
                #save new mask
                new_ni_mask = nib.Nifti1Image(augr10mask, nimask.affine)
                nib.save(new_ni_mask, mask_file_path)
           
            #display_image(img_np)
        image_mask=[]    
        
print(count)

# /augmentation/Rotate10_BraTS20_Training_001
# /BraTS2020_TrainingData/BraTS20_Training_001

org_nii=os.path.join(root_dir,"BraTS20_Training_007/BraTS20_Training_007_flair.nii")
org_mask=os.path.join(root_dir,"BraTS20_Training_007/BraTS20_Training_007_seg.nii")
niimg = nl.image.load_img(org_nii)
img_np=niimg.get_fdata()
nimask = nl.image.load_img(org_mask)
mask_np=nimask.get_fdata()
#display_image(img_np)
#display_image(mask_np)

new_nii=os.path.join(out_dir,"Rotate-5_BraTS20_Training_007/BraTS20_Training_007_R-5_flair.nii.gz")
new_mask=os.path.join(out_dir,"Rotate-5_BraTS20_Training_007/BraTS20_Training_007_R-5_seg.nii.gz")
niimg = nl.image.load_img(new_nii)
img_np=niimg.get_fdata()
nimask = nl.image.load_img(new_mask)
mask_np=nimask.get_fdata()
#display_image(img_np)
#display_image(mask_np)

#import nilearn.plotting as nlplt
nlplt.plot_roi(org_mask, title='org',bg_img=org_nii,cmap='Greys',cut_coords=(-84,104,68))
nlplt.plot_roi(new_mask, title='Aug',bg_img=new_nii,cmap='Greys', cut_coords=(-84,104,68))

