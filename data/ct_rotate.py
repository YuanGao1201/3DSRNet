import logging
import cv2
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import os
import SimpleITK as sitk
import skimage
from sklearn.externals import joblib
# import joblib
from skimage import measure
from cfgs.Bone_Predict import BonePredict
from cfgs.util import (plot_yzd, sparse_df_to_arr, arr_to_sparse_df, timer, loop_morphology_binary_opening,
                       source_hu_value_arr_to_binary, arr_to_sparse_df_only, plot_binary_array)
from cfgs.Bone_prior import BonePrior
from cfgs.Bone_Spine_Predict import BoneSpine
from cfgs.Spine_Remove import SpineRemove
from cfgs.Remove_Sternum import SternumRemove
import gc
import time
# import pycuda
# from pycuda import gpuarray
# import pycuda.driver as cuda
# import pycuda.autoinit
# from pycuda.compiler import SourceModule
# from numba import cuda
import torch
from torch import backends
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from volumentations import *
import SimpleITK as sitk
import nibabel as nib


def view_batch(imgs, lbls):
    '''
    imgs: [D, H, W, C], the depth or batch dimension should be the first.
    '''
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.set_title('image')
    ax2.set_title('label')
    """
    if init with zeros, the animation may not update? seems bug in animation.
    """
    img1 = ax1.imshow(np.random.rand(*imgs.shape[1:]))
    img2 = ax2.imshow(np.random.rand(*lbls.shape[1:]))
    def update(i):
        plt.suptitle(str(i))
        img1.set_data(imgs[i])
        img2.set_data(lbls[i])
        return img1, img2
    ani = animation.FuncAnimation(fig, update, frames=len(imgs), interval=10, blit=False, repeat_delay=0)
    plt.show()


def view_batch_nolabel(imgs):
    '''
    imgs: [D, H, W, C], the depth or batch dimension should be the first.
    '''
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.set_title('image')
    """
    if init with zeros, the animation may not update? seems bug in animation.
    """
    img1 = ax1.imshow(np.random.rand(*imgs.shape[1:]))
    def update(i):
        plt.suptitle(str(i))
        img1.set_data(imgs[i])
        return img1
    ani = animation.FuncAnimation(fig, update, frames=len(imgs), interval=10, blit=False, repeat_delay=0)
    plt.show()


def get_augmentation(patch_size):
    return Compose([
        # RemoveEmptyBorder(always_apply=True),
        # RandomScale((0.8, 1.2)),
        # PadIfNeeded(patch_size, always_apply=True),
        #RandomCrop(patch_size, always_apply=True),
        #CenterCrop(patch_size, always_apply=True),
        #RandomCrop(patch_size, always_apply=True),
        #Resize(patch_size, always_apply=True),
        # CropNonEmptyMaskIfExists(patch_size, always_apply=True),
        # Normalize(always_apply=True),
        #ElasticTransform((0, 0.25)),
        Rotate((0,0),(0,0),(90,90)),
        #Flip(0),
        #Flip(1),
        #Flip(2),
        #Transpose((1,0,2)), # only if patch.height = patch.width
        # RandomRotate90((0,1)),
        #RandomGamma(),
        #GaussianNoise(),
    ], p=1)


if __name__ == '__main__':
    # torch.backends.cudnn.enabled = True
    # torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True
    input_nii_path = '/media/gy/Data/VerSe/ctpro_RF'
    output_nii_path = '/media/gy/Data/VerSe/ctpro_rotate'
    input_nii_txt = '/media/gy/Data/VerSe/error_name.txt'
    with open(input_nii_txt, 'r') as f:
        error_name = f.readlines()
    for name in error_name:
        name = name.strip('\n')
        t0 = time.time()
        print('start: ', name)
        try:
            file_nii_path = os.path.join(input_nii_path, name, name + '_new.nii.gz')
            file_nii = sitk.ReadImage(file_nii_path)
            image_arr = sitk.GetArrayFromImage(file_nii)
            if not os.path.exists(os.path.join(output_nii_path, name)):
                os.makedirs(os.path.join(output_nii_path, name))
            size = file_nii.GetSize()
            aug = get_augmentation(size)
            lbl = np.ones(image_arr.shape)
            data = {
                'image': image_arr,
                'mask': lbl,
            }
            # aug_data = aug(**data)
            aug_data = Rotate((0,0),(0,0),(90,90))(True, [['image'],['mask']], **data)
            img = aug_data['image']
            # print(img.shape, np.max(img))
            # view_batch_nolabel(img)
            img = sitk.GetImageFromArray(img)
            sitk.WriteImage(img, os.path.join(output_nii_path, name, name + '_new.nii.gz'))
        except:
            logging.error('error_name: ', name)
        t1 = time.time()
        time_single = t1-t0
        print('end, time_single: ', time_single)