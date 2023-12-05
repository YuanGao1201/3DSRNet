# ------------------------------------------------------------------------------
# Copyright (c) Tencent
# Licensed under the GPLv3 License.
# Created by Kai Ma (makai0324@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from lib.dataset.baseDataSet import Base_DataSet
from lib.dataset.utils import *
import h5py
import numpy as np
import SimpleITK as sitk
from PIL import Image


class AlignDataSet(Base_DataSet):
    '''
    DataSet For unaligned data
    '''

    def __init__(self, opt):
        super(AlignDataSet, self).__init__()
        self.opt = opt
        self.ext = '.h5'
        self.dataset_paths = get_dataset_from_txt_file(self.opt.datasetfile)
        self.dataset_paths = sorted(self.dataset_paths)
        self.dataset_size = len(self.dataset_paths)
        self.dir_root = self.get_data_path
        self.data_augmentation = self.opt.data_augmentation(opt)

    @property
    def name(self):
        return 'AlignDataSet'

    @property
    def get_data_path(self):
        path = os.path.join(self.opt.dataroot)
        return path

    @property
    def num_samples(self):
        return self.dataset_size

    def get_image_path(self, root, index_name):
        # img_path = os.path.join(root, index_name, 'ct_xray_data' + self.ext)
        # assert os.path.exists(img_path), 'Path do not exist: {}'.format(img_path)
        # return img_path
        ct_path = os.path.join(root, 'ctpro_RF', index_name, index_name + '_new.nii.gz')
        xray1_path = os.path.join(root, 'drr_RF', index_name, 'xray1' + '.png')
        assert os.path.exists(ct_path), 'Path do not exist: {}'.format(ct_path)
        assert os.path.exists(xray1_path), 'Path do not exist: {}'.format(xray1_path)
        return ct_path, xray1_path

    # def load_file(self, file_path):
    #     hdf5 = h5py.File(file_path, 'r')
    #     ct_data = np.asarray(hdf5['ct'])
    #     x_ray1 = np.asarray(hdf5['xray1'])
    #     x_ray1 = np.expand_dims(x_ray1, 0)
    #     hdf5.close()
    #     return ct_data, x_ray1

    def load_file(self, ct_path, xray1_path):
        img_itk = sitk.ReadImage(ct_path)
        ct_data = sitk.GetArrayFromImage(img_itk)
        x_ray1 = Image.open(xray1_path)
        if not x_ray1.mode=='L':
            x_ray1 = x_ray1.convert('L')
        x_ray1 = np.expand_dims(x_ray1, 0)
        return ct_data, x_ray1


    '''
    generate batch
    '''

    def pull_item(self, item):
        file_path = self.get_image_path(self.dir_root, self.dataset_paths[item])
        # ct_data, x_ray1 = self.load_file(file_path)
        ct_path, xray1_path = self.get_image_path(self.dir_root, self.dataset_paths[item])
        ct_data, x_ray1 = self.load_file(ct_path, xray1_path)

        # Data Augmentation
        ct, xray1 = self.data_augmentation([ct_data, x_ray1])

        return ct, xray1, file_path
