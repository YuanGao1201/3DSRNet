# ------------------------------------------------------------------------------
# Copyright (c) Tencent
# Licensed under the GPLv3 License.
# Created by Kai Ma (makai0324@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import torch
import numpy as np
from lib.model.base_model import Base_Model
import lib.model.nets.factory as factory
from .loss.multi_gan_loss import GANLoss, GANLoss_D, RestructionLoss, WGANLoss
from lib.utils.image_pool import ImagePool
import lib.utils.metrics as Metrics

from focal_frequency_loss import FocalFrequencyLoss as FFL
from focal_frequency_loss import FocalFrequencyLoss_3D as FFL_3D

from .loss.image_gradient_difference_loss import igdl_loss, igdl_loss_3D
from lib.utils.metrics_np import Peak_Signal_to_Noise_Rate_3D
from lib.utils.metrics_np import Structural_Similarity

# import torch.nn.utils.prune as prune
import torch.nn.functional as F
from memonger import SublinearSequential
from apex import amp
import os
import cv2


def tensor_to_image(tensor, imtype=np.uint8):
    '''
    :param tensor:
      (c,h,w)
    :return:
    '''
    img = tensor.cpu().numpy()
    img = np.transpose(img, (1,2,0))

    # if np.uint8 == imtype:
    #     if np.max(img) > 1:
    #         print(np.max(img))
    #         raise ValueError('Image value should range from 0 to 1.')
    #     img = img * 255.0

    if np.uint8 == imtype:
        if np.max(img)>0:
            img = (img - np.min(img)) / (np.max(img) - np.min(img))
            img = img * 255.0

    return img.astype(imtype)

def save_image(image_numpy, image_path):
    image_pil = image_numpy
    cv2.imwrite(image_path, image_pil)

class CTGAN(Base_Model):
    def __init__(self):
        super(CTGAN, self).__init__()

    @property
    def name(self):
        return 'multiView_CTGAN'

    '''
    Init network architecture
    '''
    def init_network(self, opt):
        Base_Model.init_network(self, opt)

        self.if_pool = opt.if_pool
        self.multi_view = opt.multi_view
        self.conditional_D = opt.conditional_D
        self.auxiliary_loss = False
        '''focal frequency loss'''
        self.ffl_flag = False
        # self.ffl_flag = True
        '''image gradient difference loss'''
        self.igdl_flag = False
        # self.igdl_flag = True
        '''transformer'''
        self.transformer_flag = False
        # self.transformer_flag = True
        '''ssim'''
        self.ssim_flag = False
        # self.ssim_flag = True
        '''psnr'''
        self.psnr_flag = False
        # self.psnr_flag = True

        assert len(self.multi_view) > 0

        self.loss_names = ['D', 'G']
        self.metrics_names = ['Mse', 'CosineSimilarity', 'PSNR']
        self.visual_names = ['G_real', 'G_fake', 'G_input1', 'G_input2', 'G_Map_fake_F', 'G_Map_real_F', 'G_Map_fake_S', 'G_Map_real_S']

        # identity loss
        if self.opt.idt_lambda > 0:
            self.loss_names += ['idt']

        # feature metric loss
        if self.opt.feature_D_lambda > 0:
            self.loss_names += ['fea_m']

        # map loss
        if self.opt.map_projection_lambda > 0:
            self.loss_names += ['map_m']

        # auxiliary loss
        if self.opt.auxiliary_lambda > 0:
            self.loss_names += ['auxiliary']
            self.auxiliary_loss = True

        # focal frequency loss
        if self.opt.ffl_lambda > 0:
            self.loss_names += ['ffl']
            self.ffl_flag = True

        # image gradient difference loss
        if self.opt.igdl_lambda > 0:
            self.loss_names += ['igdl']
            self.igdl_flag = True

        '''transformer'''
        if self.opt.transformer_lambda > 0:
            self.loss_names += ['G_t']
            self.loss_names += ['G_t_temp']
            self.transformer_flag = True

        '''conformer'''
        if self.opt.conformer_lambda > 0:
            self.conformer_flag = True

        '''ssim loss'''
        if self.opt.ssim_lambda > 0:
            self.loss_names += ['ssim']
            self.ssim_flag = True


        '''psnr loss'''
        if self.opt.psnr_lambda > 0:
            self.loss_names += ['psnr']
            self.psnr_flag = True

        if self.training:
            self.model_names = ['G', 'D']
        else:  # during test time, only load Gs
            self.model_names = ['G']

        self.netG = factory.define_3DG(opt.noise_len, opt.input_shape, opt.output_shape,
                                       opt.input_nc_G, opt.output_nc_G, opt.ngf, opt.which_model_netG,
                                       opt.n_downsampling, opt.norm_G, not opt.no_dropout,
                                       opt.init_type, self.gpu_ids, opt.n_blocks,
                                       opt.encoder_input_shape, opt.encoder_input_nc, opt.encoder_norm,
                                       opt.encoder_blocks, opt.skip_number, opt.activation_type, opt=opt,
                                       transformer_flag=False, stl_flag=False, conformer_flag=False)
        '''stl'''
        # self.netG = factory.define_3DG(opt.noise_len, opt.input_shape, opt.output_shape,
        #                                opt.input_nc_G, opt.output_nc_G, opt.ngf, opt.which_model_netG,
        #                                opt.n_downsampling, opt.norm_G, not opt.no_dropout,
        #                                opt.init_type, self.gpu_ids, opt.n_blocks,
        #                                opt.encoder_input_shape, opt.encoder_input_nc, opt.encoder_norm,
        #                                opt.encoder_blocks, opt.skip_number, opt.activation_type, opt=opt,
        #                                transformer_flag=False, stl_flag=True, conformer_flag=False)
        '''transformer'''
        # if self.transformer_flag is True:
        #     self.netG_t = factory.define_3DG(opt.noise_len, opt.input_shape, opt.output_shape,
        #                                    opt.input_nc_G, opt.output_nc_G, opt.ngf, opt.which_model_netG,
        #                                    opt.n_downsampling, opt.norm_G, not opt.no_dropout,
        #                                    opt.init_type, self.gpu_ids, opt.n_blocks,
        #                                    opt.encoder_input_shape, opt.encoder_input_nc, opt.encoder_norm,
        #                                    opt.encoder_blocks, opt.skip_number, opt.activation_type, opt=opt,
        #                                    transformer_flag=True, stl_flag=False, conformer_flag=False)
        '''conformer'''
        # if self.conformer_flag is True:
        #     self.netG = factory.define_3DG(opt.noise_len, opt.input_shape, opt.output_shape,
        #                                      opt.input_nc_G, opt.output_nc_G, opt.ngf, opt.which_model_netG,
        #                                      opt.n_downsampling, opt.norm_G, not opt.no_dropout,
        #                                      opt.init_type, self.gpu_ids, opt.n_blocks,
        #                                      opt.encoder_input_shape, opt.encoder_input_nc, opt.encoder_norm,
        #                                      opt.encoder_blocks, opt.skip_number, opt.activation_type, opt=opt,
        #                                      transformer_flag=False, stl_flag=False, conformer_flag=True)

        if self.training:
            # out of discriminator is not probability when
            # GAN loss is LSGAN
            use_sigmoid = False
            #apex
            # use_sigmoid = True

            # conditional Discriminator
            if self.conditional_D:
                d_input_channels = opt.input_nc_D + 2
            else:
                d_input_channels = opt.input_nc_D
            self.netD = factory.define_D(d_input_channels, opt.ndf,
                                         opt.which_model_netD,
                                         opt.n_layers_D, opt.norm_D,
                                         use_sigmoid, opt.init_type, self.gpu_ids,
                                         opt.discriminator_feature, num_D=opt.num_D, n_out_channels=opt.n_out_ChannelsD)
            if self.if_pool:
                self.fake_pool = ImagePool(opt.pool_size)

    # correspond to visual_names
    def get_normalization_list(self):
        return [
            [self.opt.CT_MEAN_STD[0], self.opt.CT_MEAN_STD[1]],
            [self.opt.CT_MEAN_STD[0], self.opt.CT_MEAN_STD[1]],
            [self.opt.XRAY1_MEAN_STD[0], self.opt.XRAY1_MEAN_STD[1]],
            [self.opt.XRAY2_MEAN_STD[0], self.opt.XRAY2_MEAN_STD[1]],
            [self.opt.CT_MEAN_STD[0], self.opt.CT_MEAN_STD[1]],
            [self.opt.CT_MEAN_STD[0], self.opt.CT_MEAN_STD[1]],
            [self.opt.CT_MEAN_STD[0], self.opt.CT_MEAN_STD[1]],
            [self.opt.CT_MEAN_STD[0], self.opt.CT_MEAN_STD[1]]
        ]

    def init_loss(self, opt):
        Base_Model.init_loss(self, opt)

        # #####################
        # define loss functions
        # #####################
        # GAN loss
        # self.criterionGAN = GANLoss(use_lsgan=False).to(self.device)
        self.criterionGAN = GANLoss(use_lsgan=True).to(self.device)
        self.criterionWGAN = WGANLoss().to(self.device)
        #apex
        # self.criterionGAN_D = GANLoss_D(use_lsgan=True).to(self.device)

        # identity loss
        self.criterionIdt = RestructionLoss(opt.idt_loss, opt.idt_reduction).to(self.device)

        # feature metric loss
        self.criterionFea = torch.nn.L1Loss().to(self.device)

        # map loss
        self.criterionMap = RestructionLoss(opt.map_projection_loss).to(self.device)

        # auxiliary loss
        self.criterionAuxiliary = RestructionLoss(opt.auxiliary_loss).to(self.device)

        # feature metric loss
        self.criterionFFL = FFL(loss_weight=1.0, alpha=1.0).to(self.device)
        self.criterionFFL_3D = FFL_3D(loss_weight=1.0, alpha=1.0).to(self.device)

        # image gradient difference loss
        self.criterionIGDL = igdl_loss(igdl_p=1.0).to(self.device)
        self.criterionIGDL_3D = igdl_loss_3D(igdl_p=1.0).to(self.device)

        # #####################
        # initialize optimizers
        # #####################
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                            lr=opt.lr, betas=(opt.beta1, opt.beta2))
        '''transformer'''
        if self.transformer_flag is True:
            self.optimizer_G_t = torch.optim.Adam(self.netG_t.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, opt.beta2))
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                            lr=opt.lr, betas=(opt.beta1, opt.beta2))
        #apex
        # self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
        #                                     lr=opt.lr, eps=1e-3)
        # self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
        #                                     lr=opt.lr, betas=(opt.beta1, opt.beta2))
        # self.optimizer_G = torch.optim.SGD(self.netG.parameters(), lr=opt.lr)
        # self.optimizer_D = torch.optim.SGD(self.netD.parameters(), lr=opt.lr)

        self.optimizers = []
        self.optimizers.append(self.optimizer_G)
        '''transformer'''
        if self.transformer_flag is True:
            self.optimizers.append(self.optimizer_G_t)
        self.optimizers.append(self.optimizer_D)

    '''
      Train -Forward and Backward
    '''
    def set_input(self, input):
        self.G_input1 = input[1][0].to(self.device)
        self.G_input2 = input[1][1].to(self.device)
        self.G_real = input[0].to(self.device)

        # apex
        # self.G_input1 = input[1][0].to(self.device).half()
        # self.G_input2 = input[1][1].to(self.device).half()
        # self.G_real = input[0].to(self.device).half()

        self.image_paths = input[2:]

        # apex test
        # im = torch.squeeze(self.G_real, dim=0)
        # im = tensor_to_image(im)
        # for i in range(im.shape[0]):
        #     image_name = 'test_G_real_' + str(i) + '.png'
        #     save_path = os.path.join('/home/gy/SE/X2CT/3DGAN/save_models/multiView_CTGAN/VerSe_apex/real', image_name)
        #     save_image(im[i], save_path)
        # im = torch.squeeze(self.G_input1, dim=0)
        # im = tensor_to_image(im)
        # image_name = 'test_G_input1.png'
        # save_path = os.path.join('/home/gy/SE/X2CT/3DGAN/save_models/multiView_CTGAN/VerSe_apex/real', image_name)
        # save_image(im, save_path)
        # im = torch.squeeze(self.G_input2, dim=0)
        # im = tensor_to_image(im)
        # image_name = 'test_G_input2.png'
        # save_path = os.path.join('/home/gy/SE/X2CT/3DGAN/save_models/multiView_CTGAN/VerSe_apex/real', image_name)
        # save_image(im, save_path)

    # map function
    def output_map(self, v, dim):
        '''
        :param v: tensor
        :param dim:  dimension be reduced
        :return:
          N1HW
        '''
        ori_dim = v.dim()
        # tensor [NDHW]
        if ori_dim == 4:
            map = torch.mean(torch.abs(v), dim=dim)
            # [NHW] => [NCHW]
            return map.unsqueeze(1)
        # tensor [NCDHW] and c==1
        elif ori_dim == 5:
            # [NCHW]
            map = torch.mean(torch.abs(v), dim=dim)
            return map
        else:
            raise NotImplementedError()

    def transition(self, predict):
        p_max, p_min = predict.max(), predict.min()
        new_predict = (predict - p_min) / (p_max - p_min)
        return new_predict

    def ct_unGaussian(self, value):
        return value * self.opt.CT_MEAN_STD[1] + self.opt.CT_MEAN_STD[0]

    def ct_Gaussian(self, value):
        return (value - self.opt.CT_MEAN_STD[0]) / self.opt.CT_MEAN_STD[1]

    def post_process(self, attributes_name):
        if not self.training:
            if self.opt.CT_MEAN_STD[0] == 0 and self.opt.CT_MEAN_STD[0] == 0:
                for name in attributes_name:
                    setattr(self, name, torch.clamp(getattr(self, name), 0, 1))
            elif self.opt.CT_MEAN_STD[0] == 0.5 and self.opt.CT_MEAN_STD[0] == 0.5:
                for name in attributes_name:
                    setattr(self, name, torch.clamp(getattr(self, name), -1, 1))
            else:
                raise NotImplementedError()

    def projection_visual(self):
        # map F is projected in dimension of H
        self.G_Map_real_F = self.transition(self.output_map(self.ct_unGaussian(self.G_real), 2))
        self.G_Map_fake_F = self.transition(self.output_map(self.ct_unGaussian(self.G_fake), 2))
        # map S is projected in dimension of W
        self.G_Map_real_S = self.transition(self.output_map(self.ct_unGaussian(self.G_real), 3))
        self.G_Map_fake_S = self.transition(self.output_map(self.ct_unGaussian(self.G_fake), 3))

    def metrics_evaluation(self):
        # 3D metrics including mse, cs and psnr
        g_fake_unNorm = self.ct_unGaussian(self.G_fake)
        g_real_unNorm = self.ct_unGaussian(self.G_real)

        self.metrics_Mse = Metrics.Mean_Squared_Error(g_fake_unNorm, g_real_unNorm)
        self.metrics_CosineSimilarity = Metrics.Cosine_Similarity(g_fake_unNorm, g_real_unNorm)
        self.metrics_PSNR = Metrics.Peak_Signal_to_Noise_Rate(g_fake_unNorm, g_real_unNorm, PIXEL_MAX=1.0)

    def dimension_order_std(self, value, order):
        # standard CT dimension
        return value.permute(*tuple(np.argsort(order)))

    def forward(self):
        '''
        self.G_fake is generated object
        self.G_real is GT object
        '''
        # G_fake_D is [B 1 D H W]
        self.G_fake_D1, self.G_fake_D2, self.G_fake_D = self.netG([self.G_input1, self.G_input2])
        '''transformer'''
        if self.transformer_flag is True:
            self.G_fake_D1_t, self.G_fake_D2_t, self.G_fake_D_t = self.netG_t([self.G_input1, self.G_input2])
        # visual object should be [B D H W]
        self.G_fake = torch.squeeze(self.G_fake_D, 1)
        '''transformer'''
        if self.transformer_flag is True:
            self.G_fake_t = torch.squeeze(self.G_fake_D_t, 1)

        # apex test
        # im = torch.squeeze(self.G_fake, dim=0).detach()
        # im = tensor_to_image(im)
        # for i in range(im.shape[0]):
        #     image_name = 'test_G_fake_D_' + str(i) + '.png'
        #     save_path = os.path.join('/home/gy/SE/X2CT/3DGAN/save_models/multiView_CTGAN/VerSe_apex/fake', image_name)
        #     save_image(im[i], save_path)
        # im = torch.squeeze(self.G_fake_D1, dim=0).detach()
        # im = torch.squeeze(im, dim=0).detach()
        # im = tensor_to_image(im)
        # for i in range(im.shape[0]):
        #     image_name = 'test_G_fake_D1_' + str(i) + '.png'
        #     save_path = os.path.join('/home/gy/SE/X2CT/3DGAN/save_models/multiView_CTGAN/VerSe_apex/fake', image_name)
        #     save_image(im[i], save_path)

        # input of Discriminator is [B 1 D H W]
        self.G_real_D = torch.unsqueeze(self.G_real, 1)
        # if add auxiliary loss to generator
        if self.auxiliary_loss and self.training:
            self.G_real_D1 = self.G_real_D.permute(*self.opt.CTOrder_Xray1).detach()
            self.G_real_D2 = self.G_real_D.permute(*self.opt.CTOrder_Xray2).detach()
        # if add condition to discriminator, expanding x-ray
        # as the same shape and dimension order as CT
        if self.conditional_D and self.training:
            self.G_condition_D = torch.cat((
                self.dimension_order_std(self.G_input1.unsqueeze(1).expand_as(self.G_real_D), self.opt.CTOrder_Xray1),
                self.dimension_order_std(self.G_input2.unsqueeze(1).expand_as(self.G_real_D), self.opt.CTOrder_Xray2)
            ), dim=1).detach()
        # post processing, used only in testing
        self.post_process(['G_fake'])
        if not self.training:
            # visualization of x-ray projection
            self.projection_visual()
            # metrics
            self.metrics_evaluation()
        # multi-view projection maps for training
        # Note: self.G_real_D and self.G_fake_D are in dimension order of 'NCDHW'
        if self.training:
            self.projection_visual()
            for i in self.multi_view:
                out_map = self.output_map(self.ct_unGaussian(self.G_real_D), i + 1)
                out_map = self.ct_Gaussian(out_map)
                setattr(self, 'G_Map_{}_real'.format(i), out_map)
                # apex
                # im = torch.squeeze(out_map, dim=0).detach()
                # im = tensor_to_image(im)
                # image_name = 'G_Map_{}_real_' + str(i) + '.png'
                # save_path = os.path.join('/home/gy/SE/X2CT/3DGAN/save_models/multiView_CTGAN/VerSe_apex/fake', image_name)
                # save_image(im, save_path)

                out_map = self.output_map(self.ct_unGaussian(self.G_fake_D), i + 1)
                out_map = self.ct_Gaussian(out_map)
                setattr(self, 'G_Map_{}_fake'.format(i), out_map)
                # apex
                # im = torch.squeeze(out_map, dim=0).detach()
                # im = tensor_to_image(im)
                # image_name = 'G_Map_{}_fake_' + str(i) + '.png'
                # save_path = os.path.join('/home/gy/SE/X2CT/3DGAN/save_models/multiView_CTGAN/VerSe_apex/fake',
                #                          image_name)
                # save_image(im, save_path)

    # feature metrics loss
    def feature_metric_loss(self, D_fake_pred, D_real_pred, loss_weight, num_D, feat_weights, criterionFea):
        fea_m_lambda = loss_weight
        loss_G_fea = 0
        feat_weights = feat_weights
        D_weights = 1.0 / num_D
        weight = feat_weights * D_weights

        # multi-scale discriminator
        if isinstance(D_fake_pred[0], list):
            for di in range(num_D):
                for i in range(len(D_fake_pred[di]) - 1):
                    loss_G_fea += weight * criterionFea(D_fake_pred[di][i], D_real_pred[di][i].detach()) * fea_m_lambda
        # single discriminator
        else:
            for i in range(len(D_fake_pred) - 1):
                loss_G_fea += feat_weights * criterionFea(D_fake_pred[i], D_real_pred[i].detach()) * fea_m_lambda

        return loss_G_fea

    def backward_D_basic(self, D_network, input_real, input_fake, fake_pool, criterionGAN, loss_weight):
        D_real_pred = D_network(input_real)
        gan_loss_real = criterionGAN(D_real_pred, True)

        if self.if_pool:
            g_fake_pool = fake_pool.query(input_fake)
        else:
            g_fake_pool = input_fake
        D_fake_pool_pred = D_network(g_fake_pool.detach())
        gan_loss_fake = criterionGAN(D_fake_pool_pred, False)
        gan_loss = (gan_loss_fake + gan_loss_real) * loss_weight

        gan_loss.backward()
        return gan_loss

        # apex lossD.backward() and return
        # with amp.scale_loss(gan_loss, self.optimizer_D, loss_id=1) as scaled_lossD:
        #     scaled_lossD.backward()
        # return scaled_lossD

        # WGAN
        # D_real_pred = D_network(input_real)
        # if self.if_pool:
        #     g_fake_pool = fake_pool.query(input_fake)
        # else:
        #     g_fake_pool = input_fake
        # D_fake_pool_pred = D_network(g_fake_pool.detach())
        # gan_loss = criterionGAN(input_fake=D_fake_pool_pred, input_real=D_real_pred, is_G=False)
        # gan_loss.backward()
        # return gan_loss


    def backward_D(self):
        if self.conditional_D:
            fake_input = torch.cat([self.G_condition_D, self.G_fake_D], 1)
            real_input = torch.cat([self.G_condition_D, self.G_real_D], 1)
        else:
            fake_input = self.G_fake_D
            real_input = self.G_real_D

        # apex test
        # im = torch.squeeze(real_input, dim=0).detach()
        # for n in range(im.shape[0]):
        #     imn = tensor_to_image(im[n])
        #     for i in range(imn.shape[0]):
        #         image_name = 'real_input_G_' + str(n) + '_' + str(i) + '.png'
        #         save_path = os.path.join('/home/gy/SE/X2CT/3DGAN/save_models/multiView_CTGAN/VerSe_apex/real_D', image_name)
        #         save_image(imn[i], save_path)
        # im = torch.squeeze(fake_input, dim=0).detach()
        # for n in range(im.shape[0]):
        #     imn = tensor_to_image(im[n])
        #     for i in range(imn.shape[0]):
        #         image_name = 'fake_input_G_' + str(n) + '_' + str(i) + '.png'
        #         save_path = os.path.join('/home/gy/SE/X2CT/3DGAN/save_models/multiView_CTGAN/VerSe_apex/fake_D', image_name)
        #         save_image(imn[i], save_path)

        self.loss_D = self.backward_D_basic(self.netD, real_input, fake_input,self.fake_pool if self.if_pool else None, self.criterionGAN, self.opt.gan_lambda)
        # WGAN
        # self.loss_D = self.backward_D_basic(self.netD, real_input, fake_input,self.fake_pool if self.if_pool else None, self.criterionWGAN, self.opt.gan_lambda)

    def backward_G_basic(self, D_network, input_fake, criterionGAN, loss_weight):
        D_fake_pred = D_network(input_fake)

        # apex test
        # im = torch.squeeze(D_fake_pred[0], dim=0).detach()
        # im32 = torch.squeeze(im, dim=0).detach()
        # im = torch.squeeze(D_fake_pred[0], dim=0).half().detach()
        # im16 = torch.squeeze(im, dim=0).detach()
        # if torch.equal(im32, im16):
        #     print('equal')
        # else:
        #     print('error')
        # D_fake_pred[0] = D_fake_pred[0].half()

        loss_G = criterionGAN(D_fake_pred, True) * loss_weight
        return loss_G, D_fake_pred

    def backward_G(self):
        idt_lambda = self.opt.idt_lambda
        fea_m_lambda = self.opt.feature_D_lambda
        map_m_lambda = self.opt.map_projection_lambda

        ############################################
        # BackBone GAN
        ############################################
        # GAN loss
        if self.conditional_D:
            fake_input = torch.cat([self.G_condition_D, self.G_fake_D], 1)
            '''transformer'''
            if self.transformer_flag is True:
                fake_input_t = torch.cat([self.G_condition_D, self.G_fake_D_t], 1)
                fake_input += fake_input_t.clone().detach() * 0.2
            real_input = torch.cat([self.G_condition_D, self.G_real_D], 1)
        else:
            fake_input = self.G_fake_D
            '''transformer'''
            if self.transformer_flag is True:
                fake_input_t = self.G_fake_D_t
                fake_input += fake_input_t.clone().detach() * 0.2
            real_input = self.G_real_D

        # apex test
        # im = torch.squeeze(real_input, dim=0).detach()
        # for n in range(im.shape[0]):
        #     imn = tensor_to_image(im[n])
        #     for i in range(imn.shape[0]):
        #         image_name = 'real_input_G_' + str(n) + '_' + str(i) + '.png'
        #         save_path = os.path.join('/home/gy/SE/X2CT/3DGAN/save_models/multiView_CTGAN/VerSe_apex/real_G', image_name)
        #         save_image(imn[i], save_path)
        # im = torch.squeeze(fake_input, dim=0).detach()
        # for n in range(im.shape[0]):
        #     imn = tensor_to_image(im[n])
        #     for i in range(imn.shape[0]):
        #         image_name = 'fake_input_G_' + str(n) + '_' + str(i) + '.png'
        #         save_path = os.path.join('/home/gy/SE/X2CT/3DGAN/save_models/multiView_CTGAN/VerSe_apex/fake_G', image_name)
        #         save_image(imn[i], save_path)

        self.loss_G, D_fake_pred = self.backward_G_basic(self.netD, fake_input, self.criterionGAN, self.opt.gan_lambda)
        '''transformer'''
        if self.transformer_flag is True:
            self.loss_G_t, D_fake_pred_t = self.backward_G_basic(self.netD, fake_input_t, self.criterionGAN, self.opt.gan_lambda)


        # identity loss
        if idt_lambda > 0:
            # focus area weight assignment
            if self.opt.idt_reduction == 'none' and self.opt.idt_weight > 0:
                idt_low, idt_high = self.opt.idt_weight_range
                idt_weight = self.opt.idt_weight
                loss_idt = self.criterionIdt(self.G_fake_D, self.G_real_D)
                mask = (self.G_real_D > idt_low) & (self.G_real_D < idt_high)
                loss_idt[mask] = loss_idt[mask] * idt_weight
                self.loss_idt = loss_idt.mean() * idt_lambda
            else:
                self.loss_idt = self.criterionIdt(self.G_fake_D, self.G_real_D) * idt_lambda


        D_real_pred = self.netD(real_input)

        # feature metric loss
        if fea_m_lambda > 0:
            self.loss_fea_m = self.feature_metric_loss(D_fake_pred, D_real_pred, loss_weight=fea_m_lambda, num_D=self.opt.num_D, feat_weights=4.0 / (self.opt.n_layers_D + 1), criterionFea=self.criterionFea)

        # map loss
        if map_m_lambda > 0:
            self.loss_map_m = 0.
            for direction in self.multi_view:
                self.loss_map_m += self.criterionMap(
                    getattr(self, 'G_Map_{}_fake'.format(direction)),
                    getattr(self, 'G_Map_{}_real'.format(direction))) * map_m_lambda
            self.loss_map_m = self.loss_map_m / len(self.multi_view)


        # auxiliary loss
        if self.auxiliary_loss:
            self.loss_auxiliary = (self.criterionAuxiliary(self.G_fake_D1, self.G_real_D1) +
                                   self.criterionAuxiliary(self.G_fake_D2, self.G_real_D2)) * \
                                  self.opt.auxiliary_lambda

        # focal frequency loss
        if self.ffl_flag is True:
            self.loss_ffl = 0
            # ffl_2D
            for direction in self.multi_view:
                self.loss_ffl += self.criterionFFL(
                    getattr(self, 'G_Map_{}_fake'.format(direction)),
                    getattr(self, 'G_Map_{}_real'.format(direction))) * 10.
            self.loss_ffl = self.loss_ffl / len(self.multi_view)
            # ffl_3D
            # self.loss_ffl += self.criterionFFL_3D(self.G_fake_D, self.G_real_D)

        # image gradient difference loss
        if self.igdl_flag is True:
            self.loss_igdl = 0
            # igdl_2D
            for direction in self.multi_view:
                self.loss_igdl += self.criterionIGDL(
                    getattr(self, 'G_Map_{}_fake'.format(direction)),
                    getattr(self, 'G_Map_{}_real'.format(direction))) * 10.
            self.loss_igdl = self.loss_igdl / len(self.multi_view)
            # igdl_3D
            # self.loss_igdl = self.criterionIGDL_3D(self.G_fake_D, self.G_real_D)

        # ssim loss
        if self.ssim_flag is True:
            self.loss_ssim = 0
            # ssim_2D
            # for direction in self.multi_view:
            #     self.loss_ssim += 1 - SSIM(
            #         getattr(self, 'G_Map_{}_fake'.format(direction)),
            #         getattr(self, 'G_Map_{}_real'.format(direction)), data_range=1.0, multichannel=True) * 0.01
            # self.loss_ssim = self.loss_ssim / len(self.multi_view)
            # ssim_3D
            fake_ssim = torch.squeeze(self.G_fake_D, dim=1).cpu().detach().numpy()
            real_ssim = torch.squeeze(self.G_real_D, dim=1).cpu().detach().numpy()
            self.loss_ssim = 1 - ((Structural_Similarity(fake_ssim, real_ssim, PIXEL_MAX=1.0, size_average=True)[-1]) * 0.01) #[ssim_d.mean(), ssim_h.mean(), ssim_w.mean(), ssim_avg.mean()]
            self.loss_ssim = torch.tensor(self.loss_ssim).cuda()
            del fake_ssim
            del real_ssim

        # psnr loss
        if self.psnr_flag is True:
            self.loss_psnr = 0
            # psnr_2D
            # for direction in self.multi_view:
            #     self.loss_psnr += 1 - Metrics.Peak_Signal_to_Noise_Rate(getattr(self, 'G_Map_{}_fake'.format(direction)),
            #         getattr(self, 'G_Map_{}_real'.format(direction)), PIXEL_MAX=1.0) * 0.01
            # self.loss_psnr = self.loss_psnr / len(self.multi_view)
            # psnr_3D
            fake_psnr = torch.squeeze(self.G_fake_D, dim=1).cpu().detach().numpy()
            real_psnr = torch.squeeze(self.G_real_D, dim=1).cpu().detach().numpy()
            self.loss_psnr = 1 - (Peak_Signal_to_Noise_Rate_3D(fake_psnr, real_psnr, PIXEL_MAX=1.0) * 0.01)
            self.loss_psnr = torch.tensor(self.loss_psnr).cuda()
            del fake_psnr
            del real_psnr

        # 0.0 must be add to loss_total, otherwise loss_G will
        # be changed when loss_total_G change
        '''transformer'''
        if self.transformer_flag is True:
            # loss_G_t_temp = self.loss_G_t
            # loss_G_t_temp = torch.tensor(0).cuda()
            self.loss_G_t_temp = self.loss_G_t.clone().detach() * 0.2
            # self.loss_total_G = self.loss_G*0.5 + self.loss_G_t*0.5 + 0.0
            self.loss_total_G = self.loss_G + 0.0
            self.loss_total_G += self.loss_G_t_temp

        else:
            self.loss_total_G = self.loss_G + 0.0
        if idt_lambda > 0:
            self.loss_total_G += self.loss_idt
            # apex lossG.backward()
            # with amp.scale_loss(self.loss_idt, self.optimizer_G, loss_id=2) as scaled_lossG_idt:
            #     scaled_lossG_idt.backward()
        if fea_m_lambda > 0:
            self.loss_total_G += self.loss_fea_m
        if map_m_lambda > 0:
            self.loss_total_G += self.loss_map_m
            # apex lossG.backward()
            # with amp.scale_loss(self.loss_map_m, self.optimizer_G, loss_id=3) as scaled_lossG_map_m:
            #     scaled_lossG_map_m.backward()
        if self.auxiliary_loss:
            self.loss_total_G += self.loss_auxiliary
        if self.ffl_flag is True:
            self.loss_total_G += self.loss_ffl
        if self.igdl_flag is True:
            self.loss_total_G += self.loss_igdl
        if self.ssim_flag is True:
            self.loss_total_G += self.loss_ssim
        if self.psnr_flag is True:
            self.loss_total_G += self.loss_psnr

        # apex lossG.backward()
        # with amp.scale_loss(self.loss_total_G, self.optimizer_G, loss_id=0) as scaled_lossG:
        #     scaled_lossG.backward()

        self.loss_total_G.backward()
        '''transformer'''
        if self.transformer_flag is True:
            self.loss_G_t.backward()

    def optimize_parameters(self):
        # forward
        self()
        self.set_requires_grad([self.netD], False)
        self.optimizer_G.zero_grad()
        '''transformer'''
        if self.transformer_flag is True:
            self.optimizer_G_t.zero_grad()
        self.backward_G()
        # for name, parms in self.named_parameters():
        #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad, ' -->grad_value:', parms.grad)
        self.optimizer_G.step()
        '''transformer'''
        if self.transformer_flag is True:
            self.optimizer_G_t.step()

        self.set_requires_grad([self.netD], True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        # for name, parms in self.named_parameters():
        #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad, ' -->grad_value:', parms.grad)
        self.optimizer_D.step()

    def optimize_D(self):
        # forward
        self()
        self.set_requires_grad([self.netD], True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()


