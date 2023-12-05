# ------------------------------------------------------------------------------
# Copyright (c) Tencent
# Licensed under the GPLv3 License.
# Created by Kai Ma (makai0324@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import functools

import torch

from lib.model.nets.generator.encoder_decoder_utils import *

from ..utils import get_norm_layer

from memonger import SublinearSequential
import cv2
import os

from CoTr.position_encoding import build_position_encoding
from CoTr.DeformableTrans import DeformableTransformer

from STLNet import STL

from ASPP import ASPP, ASPP_3D

from conformer import ConvBlock_3D, FCUUp3D


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


def UNetLike_DownStep5(input_shape, encoder_input_channels, decoder_output_channels, decoder_out_activation, encoder_norm_layer, decoder_norm_layer, upsample_mode, decoder_feature_out=False):
    # 64, 32, 16, 8, 4
    encoder_block_list = [6, 12, 24, 16, 6]
    decoder_block_list = [1, 2, 2, 2, 2, 0]
    growth_rate = 32
    encoder_channel_list = [64]
    decoder_channel_list = [16, 16, 32, 64, 128, 256]
    decoder_begin_size = input_shape // pow(2, len(encoder_block_list))
    return UNetLike_DenseDimensionNet(encoder_input_channels, decoder_output_channels, decoder_begin_size, encoder_block_list, decoder_block_list, growth_rate, encoder_channel_list, decoder_channel_list, decoder_out_activation, encoder_norm_layer, decoder_norm_layer, upsample_mode, decoder_feature_out)


def UNetLike_DownStep5_3(input_shape, encoder_input_channels, decoder_output_channels, decoder_out_activation, encoder_norm_layer, decoder_norm_layer, upsample_mode, decoder_feature_out=False):
    # 64, 32, 16, 8, 4
    encoder_block_list = [6, 12, 32, 32, 12]
    decoder_block_list = [3, 3, 3, 3, 3, 1]
    growth_rate = 32
    encoder_channel_list = [64]
    decoder_channel_list = [16, 32, 64, 64, 128, 256]
    decoder_begin_size = input_shape // pow(2, len(encoder_block_list))
    return UNetLike_DenseDimensionNet(encoder_input_channels, decoder_output_channels, decoder_begin_size, encoder_block_list, decoder_block_list, growth_rate, encoder_channel_list, decoder_channel_list, decoder_out_activation, encoder_norm_layer, decoder_norm_layer, upsample_mode, decoder_feature_out)


'''our UNetLike_DownStep5_View3'''
# def UNetLike_DownStep5_View3(input_shape, encoder_input_channels, decoder_output_channels, decoder_out_activation, encoder_norm_layer, decoder_norm_layer, upsample_mode, decoder_feature_out=False):
#     # 64, 32, 16, 8, 4
#     encoder_block_list = [6, 12, 24, 16, 6]
#     decoder_block_list = [1, 2, 2, 2, 2, 0]
#     growth_rate = 32
#     encoder_channel_list = [64]
#     decoder_channel_list = [16, 16, 32, 64, 128, 256]
#     decoder_begin_size = input_shape // pow(2, len(encoder_block_list))
#     return UNetLike_DenseDimensionNet(encoder_input_channels, decoder_output_channels, decoder_begin_size, encoder_block_list, decoder_block_list, growth_rate, encoder_channel_list, decoder_channel_list, decoder_out_activation, encoder_norm_layer, decoder_norm_layer, upsample_mode, decoder_feature_out)


class UNetLike_DenseDimensionNet(nn.Module):
    def __init__(self, encoder_input_channels, decoder_output_channels, decoder_begin_size, encoder_block_list, decoder_block_list, growth_rate, encoder_channel_list, decoder_channel_list, decoder_out_activation, encoder_norm_layer=nn.BatchNorm2d, decoder_norm_layer=nn.BatchNorm3d, upsample_mode='nearest', decoder_feature_out=False):
        super(UNetLike_DenseDimensionNet, self).__init__()

        self.decoder_channel_list = decoder_channel_list
        self.decoder_block_list = decoder_block_list
        self.n_downsampling = len(encoder_block_list)
        self.decoder_begin_size = decoder_begin_size
        self.decoder_feature_out = decoder_feature_out
        activation = nn.ReLU(True)
        bn_size = 4

        ##############
        # Encoder
        ##############
        if type(encoder_norm_layer) == functools.partial:
            use_bias = encoder_norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = encoder_norm_layer == nn.InstanceNorm2d

        encoder_layers0 = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(encoder_input_channels, encoder_channel_list[0], kernel_size=7, padding=0, bias=use_bias),
            # nn.ReflectionPad2d(1), # our light
            # nn.Conv2d(encoder_input_channels, encoder_channel_list[0], kernel_size=3, padding=0, stride=1, bias=use_bias),# our light

            encoder_norm_layer(encoder_channel_list[0]),
            activation
        ]
        self.encoder_layer = nn.Sequential(*encoder_layers0)
        # self.encoder_layer = SublinearSequential(*encoder_layers0)

        num_input_channels = encoder_channel_list[0]
        for index, channel in enumerate(encoder_block_list):
            # pooling
            down_layers = [
                encoder_norm_layer(num_input_channels),
                activation,
                nn.Conv2d(num_input_channels, num_input_channels, kernel_size=3, stride=2, padding=1, bias=use_bias),
                # our light
                # nn.Conv2d(num_input_channels, num_input_channels, kernel_size=3, stride=2, padding=3 // 2, groups=num_input_channels, bias=False),
                # nn.BatchNorm2d(num_input_channels),
                # nn.ReLU(inplace=True),
                # nn.Conv2d(num_input_channels, num_input_channels, kernel_size=1, stride=1, padding=0, bias=False),
            ]
            down_layers += [
                Dense_2DBlock(encoder_block_list[index], num_input_channels, bn_size, growth_rate, encoder_norm_layer, activation, use_bias),
            ]
            num_input_channels = num_input_channels + encoder_block_list[index] * growth_rate

            # feature maps are compressed into 1 after the lastest downsample layers
            if index == (self.n_downsampling-1):
                down_layers += [
                    nn.AdaptiveAvgPool2d(1)
                ]
            else:
                num_out_channels = num_input_channels // 2
                down_layers += [
                    encoder_norm_layer(num_input_channels),
                    activation,
                    nn.Conv2d(num_input_channels, num_out_channels, kernel_size=1, stride=1, padding=0, bias=use_bias),
                ]
                num_input_channels = num_out_channels
            encoder_channel_list.append(num_input_channels)
            setattr(self, 'encoder_layer' + str(index), nn.Sequential(*down_layers))
            # setattr(self, 'encoder_layer' + str(index), SublinearSequential(*down_layers))

        ##############
        # Linker
        ##############
        if type(decoder_norm_layer) == functools.partial:
            use_bias = decoder_norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = decoder_norm_layer == nn.InstanceNorm3d

        # linker FC
        # apply fc to link 2d and 3d
        self.base_link = nn.Sequential(*[
            nn.Linear(encoder_channel_list[-1], decoder_begin_size**3*decoder_channel_list[-1]),
            nn.Dropout(0.5),
            activation
        ])
        # self.base_link = SublinearSequential(*[
        #     nn.Linear(encoder_channel_list[-1], decoder_begin_size ** 3 * decoder_channel_list[-1]),
        #     nn.Dropout(0.5),
        #     activation
        # ])

        for index, channel in enumerate(encoder_channel_list[:-1]):
            in_channels = channel
            out_channels = decoder_channel_list[index]
            link_layers = [
                Dimension_UpsampleCutBlock(in_channels, out_channels, encoder_norm_layer, decoder_norm_layer, activation, use_bias)
            ]
            setattr(self, 'linker_layer' + str(index), nn.Sequential(*link_layers))
            # setattr(self, 'linker_layer' + str(index), SublinearSequential(*link_layers))

        ##############
        # Decoder
        ##############
        for index, channel in enumerate(decoder_channel_list[:-1]):
            out_channels = channel
            in_channels = decoder_channel_list[index+1]
            decoder_layers = []
            decoder_compress_layers = []
            if index != (len(decoder_channel_list) - 2):
                decoder_compress_layers += [
                    nn.Conv3d(in_channels * 2, in_channels, kernel_size=3, padding=1, bias=use_bias),
                    # our light
                    # # nn.Conv3d(in_channels * 2, in_channels * 2, kernel_size=1, stride=1, padding=0, bias=False),
                    # # nn.BatchNorm3d(in_channels * 2),
                    # # nn.ReLU(inplace=True),
                    # nn.Conv3d(in_channels * 2, in_channels * 2, kernel_size=3, stride=1, padding=3 // 2,
                    #           groups=in_channels * 2, bias=False),
                    # nn.BatchNorm3d(in_channels * 2),
                    # nn.ReLU(inplace=True),
                    # nn.Conv3d(in_channels * 2, in_channels, kernel_size=1, stride=1, padding=0, bias=False),

                    decoder_norm_layer(in_channels),
                    activation
                ]
                for _ in range(decoder_block_list[index+1]):
                    decoder_layers += [
                        nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1, bias=use_bias),
                        # our light
                        # nn.Conv3d(in_channels, in_channels * 3, kernel_size=1, stride=1, padding=0, bias=False),
                        # nn.BatchNorm3d(in_channels * 3),
                        # nn.Conv3d(in_channels * 3, in_channels * 3, kernel_size=3, stride=1, padding=3 // 2,
                        #           groups=in_channels * 3, bias=False),
                        # nn.BatchNorm3d(in_channels * 3),
                        # nn.Conv3d(in_channels * 3, in_channels, kernel_size=1, stride=1, padding=0, bias=False),

                        decoder_norm_layer(in_channels),
                        activation
                    ]
            decoder_layers += [
                Upsample_3DUnit(3, in_channels, out_channels, decoder_norm_layer, scale_factor=2, upsample_mode=upsample_mode, activation=activation, use_bias=use_bias)
            ]
            # If decoder_feature_out is True, compressed feature after upsampling and concatenation
            # can be obtained.
            if decoder_feature_out:
                setattr(self, 'decoder_compress_layer' + str(index), nn.Sequential(*decoder_compress_layers))   # No SublinearSequential
                setattr(self, 'decoder_layer' + str(index), nn.Sequential(*decoder_layers))
                # setattr(self, 'decoder_layer' + str(index), SublinearSequential(*decoder_layers))
            else:
                setattr(self, 'decoder_layer' + str(index), nn.Sequential(*(decoder_compress_layers+decoder_layers)))   # No SublinearSequential
        # last decode
        decoder_layers = []
        decoder_compress_layers = [
            nn.Conv3d(decoder_channel_list[0] * 2, decoder_channel_list[0], kernel_size=3, padding=1, bias=use_bias),
            decoder_norm_layer(decoder_channel_list[0]),
            activation
        ]
        for _ in range(decoder_block_list[0]):
            decoder_layers += [
                nn.Conv3d(decoder_channel_list[0], decoder_channel_list[0], kernel_size=3, padding=1, bias=use_bias),
                decoder_norm_layer(decoder_channel_list[0]),
                activation
            ]
        if decoder_feature_out:
            setattr(self, 'decoder_compress_layer' + str(-1), nn.Sequential(*decoder_compress_layers))  # No SublinearSequential
            setattr(self, 'decoder_layer' + str(-1), nn.Sequential(*decoder_layers))
            # setattr(self, 'decoder_layer' + str(-1), SublinearSequential(*decoder_layers))
        else:
            setattr(self, 'decoder_layer' + str(-1), nn.Sequential(*(decoder_compress_layers + decoder_layers)))    # No SublinearSequential

        self.decoder_layer = nn.Sequential(*[
            nn.Conv3d(decoder_channel_list[0], decoder_output_channels, kernel_size=7, padding=3, bias=use_bias),
            decoder_out_activation()
        ])
        # self.decoder_layer = SublinearSequential(*[
        #     nn.Conv3d(decoder_channel_list[0], decoder_output_channels, kernel_size=7, padding=3, bias=use_bias),
        #     decoder_out_activation()
        # ])

    def forward(self, input):
        '''our_View3'''
        # if self.input.shape[2] == 2:
        #     input = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=3, padding=1, stride=2)(input)

        encoder_feature = self.encoder_layer(input)
        next_input = encoder_feature
        for i in range(self.n_downsampling):
            setattr(self, 'feature_linker' + str(i), getattr(self, 'linker_layer' + str(i))(next_input))
            next_input = getattr(self, 'encoder_layer'+str(i))(next_input)

        next_input = self.base_link(next_input.view(next_input.size(0), -1))
        next_input = next_input.view(next_input.size(0), self.decoder_channel_list[-1], self.decoder_begin_size, self.decoder_begin_size, self.decoder_begin_size)

        for i in range(self.n_downsampling - 1, -2, -1):
            if i == (self.n_downsampling - 1):
                if self.decoder_feature_out:
                    next_input = getattr(self, 'decoder_layer' + str(i))(getattr(self, 'decoder_compress_layer' + str(i))(next_input))
                else:
                    next_input = getattr(self, 'decoder_layer' + str(i))(next_input)

            else:
                if self.decoder_feature_out:
                    next_input = getattr(self, 'decoder_layer' + str(i))(getattr(self, 'decoder_compress_layer' + str(i))(torch.cat((next_input, getattr(self, 'feature_linker'+str(i+1))), dim=1)))
                else:
                    next_input = getattr(self, 'decoder_layer' + str(i))(torch.cat((next_input, getattr(self, 'feature_linker'+str(i+1))), dim=1))

        return self.decoder_layer(next_input)


class MultiView_UNetLike_DenseDimensionNet(nn.Module):
    def __init__(self, view1Model, view2Model, view1Order, view2Order, backToSub, decoder_output_channels, decoder_out_activation, decoder_block_list=None, decoder_norm_layer=nn.BatchNorm3d, upsample_mode='nearest', transformer_flag=False, stl_flag=False, conformer_flag=False):
        super(MultiView_UNetLike_DenseDimensionNet, self).__init__()
        # dropout
        self.dropout = nn.Dropout(p=0.5)

        # stl
        self.stl = STL(in_channel=64)

        # aspp
        self.aspp = ASPP(in_channel=512, depth=512)
        # self.aspp_512 = ASPP(in_channel=512, depth=512)
        # self.aspp_64 = ASPP(in_channel=64, depth=64)
        # self.aspp_128 = ASPP(in_channel=128, depth=128)
        # self.aspp_256 = ASPP(in_channel=256, depth=256)
        self.aspp_3d = ASPP_3D(in_channel=16, depth=16)

        self.view1Model = view1Model
        self.view2Model = view2Model

        '''our downsample'''
        # in: B,1,256,256. out: B,64,128,128
        # downsample2d_layers0 = [
        #     # nn.Conv2d(1, 16, kernel_size=1, stride=1, padding=0, bias=False),
        #     # nn.InstanceNorm2d(16),
        #     # nn.ReflectionPad2d(1),
        #     # nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=0, groups=16, bias=False),
        #     # nn.InstanceNorm2d(16),
        #     # nn.Conv2d(16, 64, kernel_size=1, stride=1, padding=0, bias=False),
        #     # nn.InstanceNorm2d(64),
        #     # nn.ReLU(inplace=True),
        #
        #     nn.ReflectionPad2d(1),
        #     nn.Conv2d(1, 16, kernel_size=3, padding=0, stride=1, bias=nn.InstanceNorm2d),
        #     nn.InstanceNorm2d(16),
        #     nn.ReLU(inplace=True),
        #     nn.ReflectionPad2d(1),
        #     nn.Conv2d(16, 64, kernel_size=3, padding=0, stride=2, bias=nn.InstanceNorm2d),
        #     nn.InstanceNorm2d(64),
        #     nn.ReLU(inplace=True),
        # ]
        # self.downsample2d_layer = nn.Sequential(*downsample2d_layers0)
        # self.downsample2d_layer = SublinearSequential(*downsample2d_layers0)
        '''our decoder_layer'''
        # in: B,16,128,128,128. out: B,1,256,256,256
        # decoder3d_layers = [
        #     nn.ConvTranspose3d(16, 16, kernel_size=3, padding=1, stride=2, output_padding=1),
        #     nn.InstanceNorm3d(16),
        #     nn.ReLU(True),
        #     nn.Conv3d(16, 1, kernel_size=3, padding=1, stride=1),
        #     # nn.Conv3d(16, 1, kernel_size=7, padding=1, stride=3),
        #     nn.InstanceNorm3d(1),
        #     nn.ReLU(True),
        # ]
        # decoder3d_layers_avg = [
        #     nn.ConvTranspose3d(16, 16, kernel_size=3, padding=1, stride=2, output_padding=1),
        #     nn.InstanceNorm3d(16),
        #     nn.ReLU(True),
        #     nn.Conv3d(16, 1, kernel_size=3, padding=1, stride=1),
        #     # nn.Conv3d(16, 1, kernel_size=7, padding=1, stride=3),
        #     nn.InstanceNorm3d(1),
        #     nn.ReLU(True),
        # ]
        # self.decoder3d_layer = nn.Sequential(*decoder3d_layers)
        # # self.decoder3d_layer = SublinearSequential(*decoder3d_layers)
        # self.decoder3d_layer_avg = nn.Sequential(*decoder3d_layers_avg)
        # # self.decoder3d_layer_avg = SublinearSequential(*decoder3d_layers_avg)
        '''our View3Model'''
        # self.view3Model = UNetLike_DownStep5_View3(input_shape=256, encoder_input_channels=2,
        #                                            decoder_output_channels=1, decoder_out_activation=nn.ReLU,
        #                                            encoder_norm_layer=get_norm_layer(norm_type='instance'),
        #                                            decoder_norm_layer=get_norm_layer(norm_type='batch'),
        #                                            upsample_mode='transposed', decoder_feature_out=True)

        self.view1Order = view1Order
        self.view2Order = view2Order
        self.backToSub = backToSub
        self.n_downsampling = view2Model.n_downsampling
        self.decoder_channel_list = view2Model.decoder_channel_list
        if decoder_block_list is None:
            self.decoder_block_list = view2Model.decoder_block_list
        else:
            self.decoder_block_list = decoder_block_list

        activation = nn.ReLU(True)
        if type(decoder_norm_layer) == functools.partial:
            use_bias = decoder_norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = decoder_norm_layer == nn.InstanceNorm3d
        ##############
        # Decoder
        ##############
        for index, channel in enumerate(self.decoder_channel_list[:-1]):
            out_channels = channel
            in_channels = self.decoder_channel_list[index + 1]
            decoder_layers = []
            decoder_compress_layers = []
            if index != (len(self.decoder_channel_list) - 2):
                decoder_compress_layers += [
                    nn.Conv3d(in_channels * 2, in_channels, kernel_size=3, padding=1, bias=use_bias),
                    decoder_norm_layer(in_channels),
                    activation
                ]
                for _ in range(self.decoder_block_list[index+1]):
                    decoder_layers += [
                        nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1, bias=use_bias),
                        decoder_norm_layer(in_channels),
                        activation
                    ]
            decoder_layers += [
                Upsample_3DUnit(3, in_channels, out_channels, decoder_norm_layer, scale_factor=2, upsample_mode=upsample_mode,
                                activation=activation, use_bias=use_bias)
            ]

            setattr(self, 'decoder_layer' + str(index), nn.Sequential(*(decoder_compress_layers + decoder_layers)))
            # setattr(self, 'decoder_layer' + str(index), SublinearSequential(*(decoder_compress_layers + decoder_layers)))
        # last decode
        decoder_layers = []
        decoder_compress_layers = [
            nn.Conv3d(self.decoder_channel_list[0] * 2, self.decoder_channel_list[0], kernel_size=3, padding=1, bias=use_bias),
            decoder_norm_layer(self.decoder_channel_list[0]),
            activation
        ]
        for _ in range(self.decoder_block_list[0]):
            decoder_layers += [
                nn.Conv3d(self.decoder_channel_list[0], self.decoder_channel_list[0], kernel_size=3, padding=1, bias=use_bias),
                decoder_norm_layer(self.decoder_channel_list[0]),
                activation
            ]
        setattr(self, 'decoder_layer' + str(-1), nn.Sequential(*(decoder_compress_layers + decoder_layers)))    # No SublinearSequential
        self.decoder_layer = nn.Sequential(*[
            nn.Conv3d(self.decoder_channel_list[0], decoder_output_channels, kernel_size=7, padding=3, bias=use_bias),
            # dropout
            # nn.Dropout(p=0.5),
            decoder_out_activation()
        ])
        # self.decoder_layer = SublinearSequential(*[
        #     nn.Conv3d(self.decoder_channel_list[0], decoder_output_channels, kernel_size=7, padding=3, bias=use_bias),
        #     decoder_out_activation()
        # ])

        self.transposed_layer = Transposed_And_Add(view1Order, view2Order)
        # our view3
        # self.transposed_layer_view3 = Transposed_And_Add_View3(view1Order, view2Order, [0, 1, 2, 3, 4])

        # transformer
        self.transformer_flag = transformer_flag
        if transformer_flag:
            self.position_embed = build_position_encoding(mode='v2', hidden_dim=128)
            self.encoder_Detrans = DeformableTransformer(d_model=128, dim_feedforward=512, dropout=0.1, activation='gelu', num_feature_levels=3, nhead=4, num_encoder_layers=3, enc_n_points=2)
            self.deconv64 = nn.Conv3d(in_channels=128, out_channels=64, kernel_size=(1,1,1))
            self.deconv32 = nn.Conv3d(in_channels=128, out_channels=32, kernel_size=(1,1,1))

        self.stl_flag = stl_flag
        if stl_flag:
            self.stl_conv = nn.Conv2d(in_channels=832, out_channels=64, kernel_size=(1, 1)) # in=768+64=832

        self.conformer_flag = conformer_flag
        if conformer_flag:
            self.position_embed = build_position_encoding(mode='v2', hidden_dim=128)
            self.encoder_Detrans = DeformableTransformer(d_model=128, dim_feedforward=512, dropout=0.1, activation='gelu', num_feature_levels=3, nhead=4, num_encoder_layers=3, enc_n_points=2)
            self.deconv64 = nn.Conv3d(in_channels=128, out_channels=64, kernel_size=(1, 1, 1))
            self.deconv32 = nn.Conv3d(in_channels=128, out_channels=32, kernel_size=(1, 1, 1))
            self.fusion_block4 = ConvBlock_3D(inplanes=128, outplanes=128, groups=1)
            self.fusion_block3 = ConvBlock_3D(inplanes=64, outplanes=64, groups=1)
            self.fusion_block2 = ConvBlock_3D(inplanes=32, outplanes=32, groups=1)
            # self.expand_block4 = FCUUp3D(inplanes=128, outplanes=128)
            # self.expand_block3 = FCUUp3D(inplanes=64, outplanes=64)
            # self.expand_block2 = FCUUp3D(inplanes=32, outplanes=32)

    def posi_mask(self, x):
        x_fea = []
        x_posemb = []
        masks = []
        for lvl, fea in enumerate(x):
            if lvl > 1:
                x_fea.append(fea)
                x_posemb.append(self.position_embed(fea))
                masks.append(torch.zeros((fea.shape[0], fea.shape[2], fea.shape[3], fea.shape[4]), dtype=torch.bool).cuda())
        # x_fea = x
        # x_posemb = self.position_embed(x)
        # masks = torch.zeros((x.shape[0], x.shape[2], x.shape[3], x.shape[4]), dtype=torch.bool).cuda()
        return x_fea, masks, x_posemb


    def forward(self, input):
        # # ptflops
        # input = [input, input]
        # only support two views
        assert len(input) == 2

        '''our downsample'''
        # input_downsample1 = self.downsample2d_layer(input[0])
        # input_downsample2 = self.downsample2d_layer(input[1])

        # View 1 encoding process
        # view1_encoder_feature = self.view1Model.encoder_layer(input_downsample1)
        view1_encoder_feature = self.view1Model.encoder_layer(input[0])
        #apex
        # if torch.any(torch.isnan(view1_encoder_feature)) or torch.any(torch.isinf(view1_encoder_feature)):
        #     print('nan')
        #     for parameters in self.view1Model.parameters():
        #         print(parameters)
        # apex test
        # im = torch.squeeze(view1_encoder_feature, dim=0).detach()
        # im = im.permute(2, 0, 1)
        # im = tensor_to_image(im)
        # for i in range(im.shape[0]):
        #     image_name = 'view1_encoder_feature' + '_' + str(i) + '.png'
        #     save_path = os.path.join('/home/gy/SE/X2CT/3DGAN/save_models/multiView_CTGAN/VerSe_apex/view1_encoder_feature', image_name)
        #     save_image(im[i], save_path)
        '''STL'''
        if self.stl_flag:
            stl_ouput = self.stl(view1_encoder_feature)
            view1_encoder_feature = self.stl_conv(torch.cat((stl_ouput, view1_encoder_feature), dim=1))

        view1_next_input = view1_encoder_feature
        '''conformer'''
        # B = input[0].shape[0]
        # cls_tokens = self.cls_token.expand(B, -1, -1)
        # x_t = self.trans_patch_conv(view1_encoder_feature).flatten(2).transpose(1, 2)
        # x_t = torch.cat([cls_tokens, x_t], dim=1)
        # x_t = self.trans_1(x_t)
        for i in range(self.view1Model.n_downsampling):
            setattr(self.view1Model, 'feature_linker' + str(i), getattr(self.view1Model, 'linker_layer' + str(i))(view1_next_input))
            view1_next_input = getattr(self.view1Model, 'encoder_layer'+str(i))(view1_next_input)
            # ASPP
            if view1_next_input.shape[1] == 512 and view1_next_input.shape[2] == 8:
                view1_next_input = self.aspp(view1_next_input)
            # if view1_next_input.shape[1] == 512:
            #         view1_next_input = self.aspp_512(view1_next_input)
            # if view1_next_input.shape[1] == 256:
            #         view1_next_input = self.aspp_256(view1_next_input)
            # if view1_next_input.shape[1] == 128:
            #         view1_next_input = self.aspp_128(view1_next_input)
            # if view1_next_input.shape[1] == 64:
            #         view1_next_input = self.aspp_64(view1_next_input)

            # apex
            # if torch.any(torch.isnan(view1_next_input)) or torch.any(torch.isinf(view1_next_input)):
            #     print('nan')
            #     for parameters in self.view1Model.parameters():
            #         print(parameters)
            # apex test
            # im = torch.squeeze(view1_next_input, dim=0).detach()
            # im = tensor_to_image(im)
            # for j in range(im.shape[0]):
            #     image_name = 'view1_next_input' + '_' + str(i) + '_' + str(j) + '.png'
            #     save_path = os.path.join('/home/gy/SE/X2CT/3DGAN/save_models/multiView_CTGAN/VerSe_apex/view1_next_input', image_name)
            #     save_image(im[j], save_path)

        # View 2 encoding process
        # view2_encoder_feature = self.view2Model.encoder_layer(input_downsample2)
        view2_encoder_feature = self.view2Model.encoder_layer(input[1])
        # apex
        # if torch.any(torch.isnan(view2_encoder_feature)) or torch.any(torch.isinf(view2_encoder_feature)):
        #     print('nan')
        #     for parameters in self.view2Model.parameters():
        #         print(parameters)
        '''STL'''
        if self.stl_flag:
            stl_ouput = self.stl(view2_encoder_feature)
            view2_encoder_feature = self.stl_conv(torch.cat((stl_ouput, view2_encoder_feature), dim=1))

        view2_next_input = view2_encoder_feature
        for i in range(self.view2Model.n_downsampling):
            setattr(self.view2Model, 'feature_linker' + str(i), getattr(self.view2Model, 'linker_layer' + str(i))(view2_next_input))
            view2_next_input = getattr(self.view2Model, 'encoder_layer' + str(i))(view2_next_input)
            # ASPP
            if view2_next_input.shape[1] == 512 and view2_next_input.shape[2] == 8:
                view2_next_input = self.aspp(view2_next_input)
            # if view2_next_input.shape[1] == 512:
            #     view2_next_input = self.aspp_512(view2_next_input)
            # if view2_next_input.shape[1] == 256:
            #     view2_next_input = self.aspp_256(view2_next_input)
            # if view2_next_input.shape[1] == 128:
            #     view2_next_input = self.aspp_128(view2_next_input)
            # if view2_next_input.shape[1] == 64:
            #     view2_next_input = self.aspp_64(view2_next_input)

            # apex
            # if torch.any(torch.isnan(view2_next_input)) or torch.any(torch.isinf(view2_next_input)):
            #     print('nan')
            #     for parameters in self.view2Model.parameters():
            #         print(parameters)

        '''our View 3(1+2) encoding process'''
        # input_downsample3 = torch.cat(input_downsample1, input_downsample2, dim=2)
        # view3_encoder_feature = self.view3Model.encoder_layer(input_downsample3)
        # view3_next_input = view3_encoder_feature
        # for i in range(self.view3Model.n_downsampling):
        #     setattr(self.view3Model, 'feature_linker' + str(i), getattr(self.view3Model, 'linker_layer' + str(i))(view3_next_input))
        #     view3_next_input = getattr(self.view3Model, 'encoder_layer' + str(i))(view3_next_input)

        # View 1 decoding process Part1
        view1_next_input = self.view1Model.base_link(view1_next_input.view(view1_next_input.size(0), -1))
        # apex
        # if torch.any(torch.isnan(view1_next_input)) or torch.any(torch.isinf(view1_next_input)):
        #     print('nan')
        view1_next_input = view1_next_input.view(view1_next_input.size(0), self.view1Model.decoder_channel_list[-1], self.view1Model.decoder_begin_size,
                                                 self.view1Model.decoder_begin_size, self.view1Model.decoder_begin_size)
        # apex
        # if torch.any(torch.isnan(view1_next_input)) or torch.any(torch.isinf(view1_next_input)):
        #     print('nan')
        # View 2 decoding process Part1
        view2_next_input = self.view2Model.base_link(view2_next_input.view(view2_next_input.size(0), -1))
        # apex
        # if torch.any(torch.isnan(view2_next_input)) or torch.any(torch.isinf(view2_next_input)):
        #     print('nan')
        view2_next_input = view2_next_input.view(view2_next_input.size(0), self.view2Model.decoder_channel_list[-1], self.view2Model.decoder_begin_size,
                                                 self.view2Model.decoder_begin_size, self.view2Model.decoder_begin_size)
        # apex
        # if torch.any(torch.isnan(view2_next_input)) or torch.any(torch.isinf(view2_next_input)):
        #     print('nan')
        ''' our View 3 decoding process Part1 '''
        # view3_next_input = self.view3Model.base_link(view3_next_input.view(view3_next_input.size(0), -1))
        # view3_next_input = view3_next_input.view(view3_next_input.size(0), self.view3Model.decoder_channel_list[-1],
        #                                          self.view3Model.decoder_begin_size,
        #                                          self.view3Model.decoder_begin_size, self.view3Model.decoder_begin_size)

        '''transformer'''
        if self.transformer_flag:
            x_convs1 = []
            x_convs2 = []
            for i in range(5):
                x_convs1.append(getattr(self.view1Model, 'feature_linker' + str(i)))
                x_convs2.append(getattr(self.view2Model, 'feature_linker' + str(i)))
            x_fea, masks, x_posemb = self.posi_mask(x_convs1)
            x_trans = self.encoder_Detrans(x_fea, masks, x_posemb)
            skip1 = x_trans[:, 36864::].transpose(-1, -2).view(getattr(self.view1Model, 'feature_linker' + str(4)).shape)
            setattr(self.view1Model, 'feature_linker' + str(4), skip1)
            # b = getattr(self.view1Model, 'feature_linker' + str(3)).shape
            skip2 = self.deconv64(x_trans[:, 32768:36864].transpose(-1, -2).view(torch.Size([1,128,16,16,16])))
            setattr(self.view1Model, 'feature_linker' + str(3), skip2)
            # c = getattr(self.view1Model, 'feature_linker' + str(2)).shape
            skip3 = self.deconv32(x_trans[:, :32768].transpose(-1, -2).view(torch.Size([1,128,32,32,32])))
            setattr(self.view1Model, 'feature_linker' + str(2), skip3)

            x_fea, masks, x_posemb = self.posi_mask(x_convs2)
            x_trans = self.encoder_Detrans(x_fea, masks, x_posemb)
            skip1 = x_trans[:, 36864::].transpose(-1, -2).view(getattr(self.view2Model, 'feature_linker' + str(4)).shape)
            setattr(self.view2Model, 'feature_linker' + str(4), skip1)
            # skip2 = x_trans[:, 32768:36864].transpose(-1, -2)[:, :64, :].view(getattr(self.view2Model, 'feature_linker' + str(3)).shape)
            skip2 = self.deconv64(x_trans[:, 32768:36864].transpose(-1, -2).view(torch.Size([1, 128, 16, 16, 16])))
            setattr(self.view2Model, 'feature_linker' + str(3), skip2)
            # skip3 = x_trans[:, :32768].transpose(-1, -2)[:, :32, :].view(getattr(self.view2Model, 'feature_linker' + str(2)).shape)
            skip3 = self.deconv32(x_trans[:, :32768].transpose(-1, -2).view(torch.Size([1, 128, 32, 32, 32])))
            setattr(self.view2Model, 'feature_linker' + str(2), skip3)

        '''conformer'''
        if self.conformer_flag:
            B = getattr(self.view1Model, 'feature_linker' + str(4)).shape[0]
            x_convs1 = []
            x_convs2 = []
            for i in range(5):
                x_convs1.append(getattr(self.view1Model, 'feature_linker' + str(i)))
                x_convs2.append(getattr(self.view2Model, 'feature_linker' + str(i)))
            x_fea, masks, x_posemb = self.posi_mask(x_convs1)
            x_trans = self.encoder_Detrans(x_fea, masks, x_posemb)
            skip1 = x_trans[:, 36864:].transpose(-1, -2).view(getattr(self.view1Model, 'feature_linker' + str(4)).shape)
            # skip1 = self.expand_block4(skip1)
            skip1 = self.fusion_block4(x_convs1[4], skip1, return_x_2=False)
            setattr(self.view1Model, 'feature_linker' + str(4), skip1)
            # b = getattr(self.view1Model, 'feature_linker' + str(3)).shape
            skip2 = self.deconv64(x_trans[:, 32768:36864].transpose(-1, -2).view(torch.Size([B, 128, 16, 16, 16])))
            # skip2 = self.expand_block3(skip2)
            skip2 = self.fusion_block3(x_convs1[3], skip2, return_x_2=False)
            setattr(self.view1Model, 'feature_linker' + str(3), skip2)
            # c = getattr(self.view1Model, 'feature_linker' + str(2)).shape
            skip3 = self.deconv32(x_trans[:, :32768].transpose(-1, -2).view(torch.Size([B, 128, 32, 32, 32])))
            # skip3 = self.expand_block2(skip3)
            skip3 = self.fusion_block2(x_convs1[2], skip3, return_x_2=False)
            setattr(self.view1Model, 'feature_linker' + str(2), skip3)

            x_fea, masks, x_posemb = self.posi_mask(x_convs2)
            x_trans = self.encoder_Detrans(x_fea, masks, x_posemb)
            skip1 = x_trans[:, 36864::].transpose(-1, -2).view(getattr(self.view2Model, 'feature_linker' + str(4)).shape)
            # skip1 = self.expand_block4(skip1)
            skip1 = self.fusion_block4(x_convs1[4], skip1, return_x_2=False)
            setattr(self.view2Model, 'feature_linker' + str(4), skip1)
            # skip2 = x_trans[:, 32768:36864].transpose(-1, -2)[:, :64, :].view(getattr(self.view2Model, 'feature_linker' + str(3)).shape)
            skip2 = self.deconv64(x_trans[:, 32768:36864].transpose(-1, -2).view(torch.Size([B, 128, 16, 16, 16])))
            # skip2 = self.expand_block3(skip2)
            skip2 = self.fusion_block3(x_convs1[3], skip2, return_x_2=False)
            setattr(self.view2Model, 'feature_linker' + str(3), skip2)
            # skip3 = x_trans[:, :32768].transpose(-1, -2)[:, :32, :].view(getattr(self.view2Model, 'feature_linker' + str(2)).shape)
            skip3 = self.deconv32(x_trans[:, :32768].transpose(-1, -2).view(torch.Size([B, 128, 32, 32, 32])))
            # skip3 = self.expand_block2(skip3)
            skip3 = self.fusion_block2(x_convs1[2], skip3, return_x_2=False)
            setattr(self.view2Model, 'feature_linker' + str(2), skip3)

        view_next_input = None
        # View 1 and 2 decoding process Part2
        for i in range(self.n_downsampling - 1, -2, -1):
            if i == (self.n_downsampling - 1):
                view1_next_input = getattr(self.view1Model, 'decoder_compress_layer' + str(i))(view1_next_input)
                # apex
                # if torch.any(torch.isnan(view1_next_input)) or torch.any(torch.isinf(view1_next_input)):
                #     print('nan')
                view2_next_input = getattr(self.view2Model, 'decoder_compress_layer' + str(i))(view2_next_input)
                # apex
                # if torch.any(torch.isnan(view2_next_input)) or torch.any(torch.isinf(view2_next_input)):
                #     print('nan')
                '''our View3'''
                # view3_next_input = getattr(self.view3Model, 'decoder_compress_layer' + str(i))(view3_next_input)
                ########### MultiView Fusion
                # Method One: Fused feature back to sub-branch
                if self.backToSub:
                    '''our'''
                    # view_avg = self.transposed_layer_view3(view1_next_input, view2_next_input, view3_next_input) / 3
                    view_avg = self.transposed_layer(view1_next_input, view2_next_input) / 2
                    # apex
                    # if torch.any(torch.isnan(view_avg)) or torch.any(torch.isinf(view_avg)):
                    #     print('nan')
                    view1_next_input = view_avg.permute(*self.view1Order)
                    # apex
                    # if torch.any(torch.isnan(view1_next_input)) or torch.any(torch.isinf(view1_next_input)):
                    #     print('nan')
                    view2_next_input = view_avg.permute(*self.view2Order)
                    # apex
                    # if torch.any(torch.isnan(view2_next_input)) or torch.any(torch.isinf(view2_next_input)):
                    #     print('nan')
                    view_next_input = getattr(self, 'decoder_layer' + str(i))(view_avg)
                    # apex
                    # if torch.any(torch.isnan(view_next_input)) or torch.any(torch.isinf(view_next_input)):
                    #     print('nan')
                # Method Two: Fused feature only used in main-branch
                else:
                    view_avg = self.transposed_layer(view1_next_input, view2_next_input) / 2
                    view_next_input = getattr(self, 'decoder_layer' + str(i))(view_avg)
                ###########
                view1_next_input = getattr(self.view1Model, 'decoder_layer' + str(i))(view1_next_input)
                # apex
                # if torch.any(torch.isnan(view1_next_input)) or torch.any(torch.isinf(view1_next_input)):
                #     print('nan')
                view2_next_input = getattr(self.view2Model, 'decoder_layer' + str(i))(view2_next_input)
                # apex
                # if torch.any(torch.isnan(view2_next_input)) or torch.any(torch.isinf(view2_next_input)):
                #     print('nan')
            else:
                view1_next_input = getattr(self.view1Model, 'decoder_compress_layer' + str(i))(torch.cat((view1_next_input, getattr(self.view1Model, 'feature_linker' + str(i + 1))), dim=1))
                # apex
                # if torch.any(torch.isnan(view1_next_input)) or torch.any(torch.isinf(view1_next_input)):
                #     print('nan')
                # apex test
                # im = torch.squeeze(view1_next_input, dim=0).detach()
                # for n in range(im.shape[0]):
                #     imn = tensor_to_image(im[n])
                #     for m in range(imn.shape[0]):
                #         image_name = 'view1_next_input_' + str(i) + '_' + str(n) + '_' + str(m) + '.png'
                #         save_path = os.path.join('/home/gy/SE/X2CT/3DGAN/save_models/multiView_CTGAN/VerSe_apex/decoder_compress_layer', image_name)
                #         save_image(imn[m], save_path)
                view2_next_input = getattr(self.view2Model, 'decoder_compress_layer' + str(i))(torch.cat((view2_next_input, getattr(self.view2Model, 'feature_linker' + str(i + 1))), dim=1))
                # apex
                # if torch.any(torch.isnan(view2_next_input)) or torch.any(torch.isinf(view2_next_input)):
                #     print('nan')
                ########### MultiView Fusion
                # Method One: Fused feature back to sub-branch
                if self.backToSub:
                    '''our'''
                    # view_avg = self.transposed_layer_view3(view1_next_input, view2_next_input, view3_next_input) / 3
                    view_avg = self.transposed_layer(view1_next_input, view2_next_input) / 2
                    # apex
                    # if torch.any(torch.isnan(view_avg)) or torch.any(torch.isinf(view_avg)):
                    #     print('nan')
                    # apex test
                    # im = torch.squeeze(view_avg, dim=0).detach()
                    # for n in range(im.shape[0]):
                    #     imn = tensor_to_image(im[n])
                    #     for m in range(imn.shape[0]):
                    #         image_name = 'view_avg_' + str(i) + '_' + str(n) + '_' + str(m) + '.png'
                    #         save_path = os.path.join('/home/gy/SE/X2CT/3DGAN/save_models/multiView_CTGAN/VerSe_apex/backToSub', image_name)
                    #         save_image(imn[m], save_path)
                    view1_next_input = view_avg.permute(*self.view1Order)
                    view2_next_input = view_avg.permute(*self.view2Order)
                    view_next_input = getattr(self, 'decoder_layer' + str(i))(torch.cat((view_avg, view_next_input), dim=1))
                    # apex
                    # if torch.any(torch.isnan(view_next_input)) or torch.any(torch.isinf(view_next_input)):
                    #     print('nan')
                    # apex test
                    # im = torch.squeeze(view_next_input, dim=0).detach()
                    # for n in range(im.shape[0]):
                    #     imn = tensor_to_image(im[n])
                    #     for m in range(imn.shape[0]):
                    #         image_name = 'view_next_input_' + str(i) + '_' + str(n) + '_' + str(m) + '.png'
                    #         save_path = os.path.join('/home/gy/SE/X2CT/3DGAN/save_models/multiView_CTGAN/VerSe_apex/backToSub', image_name)
                    #         save_image(imn[m], save_path)
                # Method Two: Fused feature only used in main-branch
                else:
                    view_avg = self.transposed_layer(view1_next_input, view2_next_input) / 2
                    view_next_input = getattr(self, 'decoder_layer' + str(i))(torch.cat((view_avg, view_next_input), dim=1))
                ###########
                view1_next_input = getattr(self.view1Model, 'decoder_layer' + str(i))(view1_next_input)
                # apex
                # if torch.any(torch.isnan(view1_next_input)) or torch.any(torch.isinf(view1_next_input)):
                #     print('nan')
                view2_next_input = getattr(self.view2Model, 'decoder_layer' + str(i))(view2_next_input)
                # apex
                # if torch.any(torch.isnan(view2_next_input)) or torch.any(torch.isinf(view2_next_input)):
                #     print('nan')

        '''our upsample'''
        # # print('view1_next_input_shape: ', view1_next_input.size)
        # # print('view_next_input_shape: ', view_next_input.size)
        # fake1 = self.decoder3d_layer(view1_next_input)
        # fake2 = self.decoder3d_layer(view2_next_input)
        # fake = self.decoder3d_layer_avg(view_next_input)
        # return fake1, fake2, fake

        # apex
        # a = self.view1Model.decoder_layer(view1_next_input)
        # if torch.any(torch.isnan(a)) or torch.any(torch.isinf(a)):
        #     print('nan')
        # b = self.view2Model.decoder_layer(view2_next_input)
        # if torch.any(torch.isnan(b)) or torch.any(torch.isinf(b)):
        #     print('nan')
        # c = self.decoder_layer(view_next_input)
        # if torch.any(torch.isnan(c)) or torch.any(torch.isinf(c)):
        #     print('nan')
        # return a,b,c

        # ASPP_3D
        # view1_next_input = self.aspp_3d(view1_next_input)
        # view2_next_input = self.aspp_3d(view2_next_input)
        # view_next_input = self.aspp_3d(view_next_input)

        # dropout
        # view1_next_input = self.dropout(view1_next_input)
        # view2_next_input = self.dropout(view2_next_input)
        # view_next_input = self.dropout(view_next_input)

        return self.view1Model.decoder_layer(view1_next_input), self.view2Model.decoder_layer(view2_next_input), self.decoder_layer(view_next_input)