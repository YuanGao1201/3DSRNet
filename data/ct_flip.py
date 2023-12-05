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


# 旋转，axis为旋转轴，0,1,2分别代表x,y,z轴
# theta为旋转角度，单位已改为度，非弧度
# center为旋转中心，其为一维np数组[x,y,z]，默认值为图像中心点
def rotation(data, axis, theta, c=np.array([])):  # c代表旋转点
    theta = -np.pi * theta / 180
    if c.size == 0:
        c = np.array(
            [np.floor((data.shape[0] - 1) / 2), np.floor((data.shape[1] - 1) / 2), np.floor((data.shape[1] - 1) / 2)])

    s = data.shape
    mean = np.mean(data)
    # new_data = np.ones(s) * mean # 补均值
    new_data = np.zeros(s)  # 补零

    # 绕x轴旋转
    if axis == 0:
        for i in range(0, s[0]):
            for j in range(0, s[1]):
                for k in range(0, s[2]):
                    x = i
                    y = (j - c[1]) * np.cos(theta) - (k - c[2]) * np.sin(theta) + c[1]
                    if (y < 0 or y > s[1] - 1):
                        continue
                    z = (j - c[1]) * np.sin(theta) + (k - c[2]) * np.cos(theta) + c[2]
                    if (z < 0 or z > s[2] - 1):
                        continue
                    y1 = np.floor(y).astype(int)
                    y2 = np.ceil(y).astype(int)
                    z1 = np.floor(z).astype(int)
                    z2 = np.ceil(z).astype(int)
                    dy = y - y1
                    dz = z - z1
                    new_data[i, j, k] = (data[x, y1, z1] * (1 - dy) + data[x, y2, z1] * dy) * (1 - dz) + (
                                data[x, y1, z2] * (1 - dy) + data[x, y2, z2] * dy) * dz

    # 绕y轴旋转
    elif axis == 1:
        for i in range(0, s[0]):
            for j in range(0, s[1]):
                for k in range(0, s[2]):
                    y = j
                    x = (i - c[0]) * np.cos(theta) - (k - c[2]) * np.sin(theta) + c[0]
                    if (x < 0 or x > s[0] - 1):
                        continue
                    z = (i - c[0]) * np.sin(theta) + (k - c[2]) * np.cos(theta) + c[2]
                    if (z < 0 or z > s[2] - 1):
                        continue
                    x1 = np.floor(x).astype(int)
                    x2 = np.ceil(x).astype(int)
                    z1 = np.floor(z).astype(int)
                    z2 = np.ceil(z).astype(int)
                    dx = x - x1
                    dz = z - z1
                    new_data[i, j, k] = (data[x1, y, z1] * (1 - dx) + data[x2, y, z1] * dx) * (1 - dz) + (
                                data[x1, y, z2] * (1 - dx) + data[x2, y, z2] * dx) * dz

    # 绕z轴旋转
    else:
        for i in range(0, s[0]):
            for j in range(0, s[1]):
                for k in range(0, s[2]):
                    z = k
                    x = (i - c[0]) * np.cos(theta) - (j - c[1]) * np.sin(theta) + c[0]
                    if (x < 0 or x > s[0] - 1):
                        continue
                    y = (i - c[0]) * np.sin(theta) + (j - c[1]) * np.cos(theta) + c[1]
                    if (y < 0 or y > s[1] - 1):
                        continue
                    x1 = np.floor(x).astype(int)
                    x2 = np.ceil(x).astype(int)
                    y1 = np.floor(y).astype(int)
                    y2 = np.ceil(y).astype(int)
                    dx = x - x1
                    dy = y - y1
                    new_data[i, j, k] = (data[x1, y1, z] * (1 - dx) + data[x2, y1, z] * dx) * (1 - dy) + (
                                data[x1, y2, z] * (1 - dx) + data[x2, y2, z] * dx) * dy
    image_new = sitk.GetImageFromArray(new_data)
    return image_new


def rotation_gpu(data, axis, theta, c=np.array([])):  # c代表旋转点
    # gpu_data = cuda.mem_alloc(data.nbytes)
    # cuda.memcpy_htod(gpu_data, data)
    gpu_data = torch.from_numpy(data).cuda()
    theta = -np.pi * theta / 180
    theta = torch.tensor(theta).double().cuda()
    if c.size == 0:
        c = np.array(
            [np.floor((gpu_data.shape[0] - 1) / 2), np.floor((gpu_data.shape[1] - 1) / 2), np.floor((gpu_data.shape[1] - 1) / 2)])
        c = torch.from_numpy(c).cuda()

    s = data.shape
    mean = torch.mean(gpu_data).cuda()
    # new_data = np.ones(s) * mean # 补均值
    new_data = np.zeros(s)  # 补零
    new_data = torch.from_numpy(new_data).cuda()

    # 绕x轴旋转
    if axis == 0:
        for i in range(0, s[0]):
            for j in range(0, s[1]):
                for k in range(0, s[2]):
                    x = i
                    y = (j - c[1]) * torch.cos(theta) - (k - c[2]) * torch.sin(theta) + c[1]
                    if (y < 0 or y > s[1] - 1):
                        continue
                    z = (j - c[1]) * torch.sin(theta) + (k - c[2]) * torch.cos(theta) + c[2]
                    if (z < 0 or z > s[2] - 1):
                        continue
                    y1 = torch.floor(y).int()
                    y2 = torch.ceil(y).int()
                    z1 = torch.floor(z).int()
                    z2 = torch.ceil(z).int()
                    dy = y - y1
                    dz = z - z1
                    new_data[i, j, k] = (gpu_data[x, y1, z1] * (1 - dy) + gpu_data[x, y2, z1] * dy) * (1 - dz) + (
                                gpu_data[x, y1, z2] * (1 - dy) + gpu_data[x, y2, z2] * dy) * dz

    # 绕y轴旋转
    elif axis == 1:
        for i in range(0, s[0]):
            for j in range(0, s[1]):
                for k in range(0, s[2]):
                    y = j
                    x = (i - c[0]) * torch.cos(theta) - (k - c[2]) * torch.sin(theta) + c[0]
                    if (x < 0 or x > s[0] - 1):
                        continue
                    z = (i - c[0]) * torch.sin(theta) + (k - c[2]) * torch.cos(theta) + c[2]
                    if (z < 0 or z > s[2] - 1):
                        continue
                    x1 = torch.floor(x).int()
                    x2 = torch.ceil(x).int()
                    z1 = torch.floor(z).int()
                    z2 = torch.ceil(z).int()
                    dx = (x.int() - x1)
                    dz = z.int() - z1
                    new_data[i, j, k] = (gpu_data[x1, y, z1] * (1 - dx).double() + gpu_data[x2, y, z1] * dx.double()) * (1 - dz).double() + (
                            gpu_data[x1, y, z2] * (1 - dx).double() + gpu_data[x2, y, z2] * dx.double()) * dz.double()

    # 绕z轴旋转
    else:
        for i in range(0, s[0]):
            for j in range(0, s[1]):
                for k in range(0, s[2]):
                    z = k
                    x = (i - c[0]) * torch.cos(theta) - (j - c[1]) * torch.sin(theta) + c[0]
                    if (x < 0 or x > s[0] - 1):
                        continue
                    y = (i - c[0]) * torch.sin(theta) + (j - c[1]) * torch.cos(theta) + c[1]
                    if (y < 0 or y > s[1] - 1):
                        continue
                    x1 = torch.floor(x).int()
                    x2 = torch.ceil(x).int()
                    y1 = torch.floor(y).int()
                    y2 = torch.ceil(y).int()
                    dx = x - x1
                    dy = y - y1
                    new_data[i, j, k] = (gpu_data[x1, y1, z] * (1 - dx) + gpu_data[x2, y1, z] * dx) * (1 - dy) + (
                                gpu_data[x1, y2, z] * (1 - dx) + gpu_data[x2, y2, z] * dx) * dy
    # new_data = np.empty_like(data)
    # cuda.memcpy_dtoh(new_data, gpu_data)
    new_data = new_data.numpy()
    image_new = sitk.GetImageFromArray(new_data)
    return image_new


def flip(file_nii, image_arr):
    size = file_nii.GetSize()
    origin = file_nii.GetOrigin()  # order: x, y, z
    spacing = file_nii.GetSpacing()  # order:x, y, z
    direction = file_nii.GetDirection()
    print(spacing)

    pixelType = sitk.sitkUInt8
    image_new = sitk.Image(size, pixelType)

    image_arr_new = image_arr[:, :, ::-1]
    image_new = sitk.GetImageFromArray(image_arr_new)
    image_new.SetDirection(direction)
    image_new.SetSpacing(spacing)
    image_new.SetOrigin(origin)

    return image_new


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
        # try:
        file_nii_path = os.path.join(input_nii_path, name, name + '_new.nii.gz')
        file_nii = sitk.ReadImage(file_nii_path)
        image_arr = sitk.GetArrayFromImage(file_nii)
        if not os.path.exists(os.path.join(output_nii_path, name)):
            os.makedirs(os.path.join(output_nii_path, name))
        size = file_nii.GetSize()

        # image_new = flip(file_nii, image_arr) # flip

        # image_new = rotation(image_arr, 1, 90)  # rotate
        image_new = rotation_gpu(image_arr, 1, 90)  # rotate

        sitk.WriteImage(image_new, os.path.join(output_nii_path, name, name+'_new.nii.gz'))

        # except:
        #     logging.error('error_name: ', name)
        t1 = time.time()
        time_single = t1-t0
        print('end, time_single: ', time_single)