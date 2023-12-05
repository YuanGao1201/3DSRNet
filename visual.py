# ------------------------------------------------------------------------------
# Copyright (c) Tencent
# Licensed under the GPLv3 License.
# Created by Kai Ma (makai0324@gmail.com)
# ------------------------------------------------------------------------------

import argparse
from lib.config.config import cfg_from_yaml, cfg, merge_dict_and_yaml, print_easy_dict
from lib.dataset.factory import get_dataset
from lib.model.factory import get_model
from lib.utils import html
from lib.utils.visualizer import tensor_back_to_unnormalization, save_images, tensor_back_to_unMinMax
# from lib.utils.metrics_np import MAE, MSE, Peak_Signal_to_Noise_Rate, Structural_Similarity, Cosine_Similarity
from lib.utils import ct as CT
import copy
import tqdm
import torch
import numpy as np
import os
from thop import profile
from ptflops import get_model_complexity_info


def parse_args():
    parse = argparse.ArgumentParser(description='CTGAN')
    parse.add_argument('--data', type=str,
                       # default='LIDC256',
                       # default='VerSe_SS',
                       # default='VerSe_apex',
                       # default='VerSe_apex_transformer',
                       # default='VerSe_apex_transformer_twin',
                       # default='VerSe_spine_transformer_twin',
                       # default='VerSe_stl',
                       # default='VerSe_aspp',
                       default='VerSe_128_base',
                       # default='VerSe_128_base_SmoothL1Loss',
                       # default='VerSe_128_base_psnr',
                       # default='VerSe_128_base_psnr3d',
                       # default='VerSe_128_base_ssim',
                       # default='VerSe_aspp_ssim',
                       # default='VerSe_aspp_ffl_3D',
                       # default='VerSe_aspp_finetune',
                       # default='VerSe_aspp_igdl_3D',
                       # default='VerSe_aspp2',
                       # default='VerSe_igdl',
                       # default='VerSe_apex_rotate15_random',
                       # default='VerSe_apex_transformer_rotate15',
                       # default='VerSe_apex_256',
                       # default='VerSe_RF_noffl',
                       # default='VerSe_RF_crop128',
                       # default='VerSe',
                       # default='VerSe_ffl',
                       # default='VerSe_aspp_conformer_igdl2D_ffl2D',
                       # default='VerSe_aspp_conformer_resize128',
                       # default='VerSe_aspp_conformer',
                       dest='data',
                       help='input data ')
    parse.add_argument('--tag', type=str,
                       default='d2_multiview2500',
                       dest='tag',
                       help='distinct from other try')
    parse.add_argument('--dataroot', type=str,
                       # default='/media/gy/Data/2019-CVPR-X2CT/LIDC-HDF5-256',
                       default='/media/gy/Data/VerSe/',
                       # default='/media/gy/Data/VerSe/ctpro_rotate15_random_crop128/',
                       # default='/media/gy/Data/VerSe/ctpro_rotate15_crop128/',
                       # default='/home/gy/VerSe/',
                       dest='dataroot',
                       help='input data root')
    parse.add_argument('--dataset', type=str,
                       default='test',
                       dest='dataset',
                       help='Train or test or valid')
    parse.add_argument('--datasetfile', type=str,
                       # default='./data/test-LIDC.txt',
                       # default='./data/verse_test.txt',
                       # default='./data/verse_RF128_test.txt',
                       default='./data/real_test.txt',
                       dest='datasetfile',
                       help='Train or test or valid file path')
    parse.add_argument('--ymlpath', type=str,
                       default='./experiment/multiview2500/d2_multiview2500.yml',
                       dest='ymlpath',
                       help='config have been modified')
    parse.add_argument('--gpu', type=str,
                       # default='0,1',
                       default='0',
                       dest='gpuid',
                       help='gpu is split by ,')
    parse.add_argument('--dataset_class', type=str,
                       default='align_ct_xray_views_std',
                       dest='dataset_class',
                       help='Dataset class should select from unalign /')
    parse.add_argument('--model_class', type=str,
                       default='MultiViewCTGAN',
                       dest='model_class',
                       help='Model class should select from cyclegan / ')
    parse.add_argument('--check_point', type=str,
                       default='100',
                       dest='check_point',
                       help='which epoch to load? ')
    parse.add_argument('--latest', action='store_true', dest='latest',
                       help='set to latest to use latest cached model')
    parse.add_argument('--verbose', action='store_true', dest='verbose',
                       help='if specified, print more debugging information')
    parse.add_argument('--load_path', type=str,
                       # default='/media/gy/Data/VerSe/X2CT/save_models/multiView_CTGAN/VerSe_igdl/d2_multiview2500/checkpoint',
                       # default='/media/gy/Data/VerSe/X2CT/save_models/multiView_CTGAN/VerSe_aspp2/d2_multiview2500/checkpoint',
                       # default='/media/gy/Data/VerSe/X2CT/save_models/multiView_CTGAN/VerSe_aspp_igdl_3D/d2_multiview2500/checkpoint',
                       # default='/media/gy/Data/VerSe/X2CT/save_models/multiView_CTGAN/VerSe_aspp_finetune/d2_multiview2500/checkpoint',
                       # default='/media/gy/Data/VerSe/X2CT/save_models/multiView_CTGAN/VerSe_aspp_ffl_3D/d2_multiview2500/checkpoint',
                       # default='/media/gy/Data/VerSe/X2CT/save_models/multiView_CTGAN/VerSe_aspp/d2_multiview2500/checkpoint',
                       default='/media/gy/Data/VerSe/X2CT/save_models/multiView_CTGAN/VerSe_128_base/d2_multiview2500/checkpoint',
                       # default='/media/gy/Data/VerSe/X2CT/save_models/multiView_CTGAN/VerSe_stl/d2_multiview2500/checkpoint',
                       # default='/media/gy/Data/VerSe/X2CT/save_models/multiView_CTGAN/VerSe_spine_transformer_twin/d2_multiview2500/checkpoint',
                       # default='/media/gy/Data/VerSe/X2CT/save_models/multiView_CTGAN/VerSe_apex_transformer_twin/d2_multiview2500/checkpoint',
                       # default='/home/gy/SE/X2CT/3DGAN/save_models/multiView_CTGAN/VerSe_apex_rotate15_random/d2_multiview2500/checkpoint',
                       # default='/media/gy/Data/VerSe/X2CT/save_models/multiView_CTGAN/VerSe_apex_transformer/d2_multiview2500/checkpoint',
                       # default='/media/gy/Data/VerSe/X2CT/save_models/multiView_CTGAN/VerSe_apex_256/d2_multiview2500/checkpoint',
                       # default='/home/gy/X2CT/3DGAN/save_models/multiView_CTGAN/VerSe_apex/d2_multiview2500/checkpoint',
                       # default='/media/gy/Data/VerSe/X2CT/save_models/multiView_CTGAN/VerSe_aspp_conformer_igdl2D_ffl2D/d2_multiview2500/checkpoint',
                       # default='/media/gy/Data/VerSe/X2CT/save_models/multiView_CTGAN/VerSe_aspp_conformer_resize128/d2_multiview2500/checkpoint',
                       # default='/media/gy/Data/VerSe/X2CT/save_models/multiView_CTGAN/VerSe_aspp_conformer/d2_multiview2500/checkpoint',
                       # default='/media/gy/Data/VerSe/X2CT/save_models/multiView_CTGAN/VerSe_128_base_SmoothL1Loss/d2_multiview2500/checkpoint',
                       # default='/media/gy/Data/VerSe/X2CT/save_models/multiView_CTGAN/VerSe_128_base_psnr/d2_multiview2500/checkpoint',
                       # default='/media/gy/Data/VerSe/X2CT/save_models/multiView_CTGAN/VerSe_128_base_psnr3d/d2_multiview2500/checkpoint',
                       # default='/media/gy/Data/VerSe/X2CT/save_models/multiView_CTGAN/VerSe_128_base_ssim/d2_multiview2500/checkpoint',
                       # default='/media/gy/Data/VerSe/X2CT/save_models/multiView_CTGAN/VerSe_aspp_ssim/d2_multiview2500/checkpoint',
                       dest='load_path',
                       help='if load_path is not None, model will load from load_path')
    parse.add_argument('--how_many', type=int, dest='how_many',
                       default=45,
                       help='if specified, print more debugging information')
    parse.add_argument('--resultdir', type=str,
                       default='/media/gy/Data/VerSe/X2CT/visual',
                       # default='./multiview',
                       # default='./multiview-real',
                       dest='resultdir',
                       help='dir to save result')
    args = parse.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    # check gpu
    if args.gpuid == '':
        args.gpu_ids = []
    else:
        if torch.cuda.is_available():
            split_gpu = str(args.gpuid).split(',')
            args.gpu_ids = [int(i) for i in split_gpu]
        else:
            print('There is no gpu!')
            exit(0)

    # check point
    if args.check_point is None:
        args.epoch_count = 1
    else:
        args.epoch_count = int(args.check_point)

    # merge config with yaml
    if args.ymlpath is not None:
        cfg_from_yaml(args.ymlpath)
    # merge config with argparse
    opt = copy.deepcopy(cfg)
    opt = merge_dict_and_yaml(args.__dict__, opt)
    print_easy_dict(opt)

    opt.serial_batches = True

    # add data_augmentation
    datasetClass, _, dataTestClass, collateClass = get_dataset(opt.dataset_class)
    opt.data_augmentation = dataTestClass

    # get dataset
    dataset = datasetClass(opt)
    print('DataSet is {}'.format(dataset.name))
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(opt.nThreads),
        collate_fn=collateClass)

    dataset_size = len(dataloader)
    print('#Test images = %d' % dataset_size)

    # get model
    gan_model = get_model(opt.model_class)()
    print('Model --{}-- will be Used'.format(gan_model.name))

    # set to test
    gan_model.eval()

    gan_model.init_process(opt)
    total_steps, epoch_count = gan_model.setup(opt)

    # must set to test Mode again, due to  omission of assigning mode to network layers
    # model.training is test, but BN.training is training
    if opt.verbose:
        print('## Model Mode: {}'.format('Training' if gan_model.training else 'Testing'))
        for i, v in gan_model.named_modules():
            print(i, v.training)

    if 'batch' in opt.norm_G:
        gan_model.eval()
    elif 'instance' in opt.norm_G:
        gan_model.eval()
        # instance norm in training mode is better
        for name, m in gan_model.named_modules():
            if m.__class__.__name__.startswith('InstanceNorm'):
                m.train()
    else:
        raise NotImplementedError()

    if opt.verbose:
        print('## Change to Model Mode: {}'.format('Training' if gan_model.training else 'Testing'))
        for i, v in gan_model.named_modules():
            print(i, v.training)

    web_dir = os.path.join(opt.resultdir, opt.data, '%s_%s' % (opt.dataset, opt.check_point))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.data, opt.dataset, opt.check_point))
    ctVisual = CT.CTVisual()

    # FLOPs-thop
    # input0 = torch.randn(1, 1, 128, 128).cuda()
    # input1 = torch.randn(1, 1, 128, 128).cuda()
    # input = [input0, input1]
    # macs, params = profile(gan_model.netG, inputs=(input,))
    # print("MACs:", macs, ", FLOPs: ", macs*2, ", params:", params)

    # FLOPs-ptflops
    # macs, params = get_model_complexity_info(gan_model.netG, (1, 128, 128), as_strings=True,
    #                                          print_per_layer_stat=True, verbose=True)
    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    avg_dict = dict()
    for epoch_i, data in tqdm.tqdm(enumerate(dataloader)):

        gan_model.set_input(data)
        gan_model.test()

        visuals = gan_model.get_current_visuals()
        img_path = gan_model.get_image_paths()

        if epoch_i <= opt.how_many:
            #
            # Image HTML
            #
            save_images(webpage, visuals, img_path, gan_model.get_normalization_list(), max_image=50)
            #
            # CT Source
            #
            generate_CT = visuals['G_fake'].data.clone().cpu().numpy()
            real_CT = visuals['G_real'].data.clone().cpu().numpy()
            # To NDHW
            if 'std' in opt.dataset_class or 'baseline' in opt.dataset_class:
                generate_CT_transpose = generate_CT
                real_CT_transpose = real_CT
            else:
                generate_CT_transpose = np.transpose(generate_CT, (0, 2, 1, 3))
                real_CT_transpose = np.transpose(real_CT, (0, 2, 1, 3))
            # Inveser Deepth
            generate_CT_transpose = generate_CT_transpose[:, ::-1, :, :]
            real_CT_transpose = real_CT_transpose[:, ::-1, :, :]
            # To [0, 1]
            generate_CT_transpose = tensor_back_to_unnormalization(generate_CT_transpose, opt.CT_MEAN_STD[0],
                                                                   opt.CT_MEAN_STD[1])
            real_CT_transpose = tensor_back_to_unnormalization(real_CT_transpose, opt.CT_MEAN_STD[0],
                                                               opt.CT_MEAN_STD[1])
            # Clip generate_CT
            generate_CT_transpose = np.clip(generate_CT_transpose, 0, 1)

            # #
            # # Evaluate Part
            # #
            # mae = MAE(real_CT_transpose, generate_CT_transpose, size_average=False)
            # mse = MSE(real_CT_transpose, generate_CT_transpose, size_average=False)
            # cosinesimilarity = Cosine_Similarity(real_CT_transpose, generate_CT_transpose, size_average=False)
            # psnr = Peak_Signal_to_Noise_Rate(real_CT_transpose, generate_CT_transpose, size_average=False, PIXEL_MAX=1.0)
            # ssim = Structural_Similarity(real_CT_transpose, generate_CT_transpose, size_average=False, PIXEL_MAX=1.0)
            #
            # metrics_list = [('MAE', mae), ('MSE', mse), ('CosineSimilarity', cosinesimilarity), ('PSNR-1', psnr[0]),
            #                 ('PSNR-2', psnr[1]), ('PSNR-3', psnr[2]), ('PSNR-avg', psnr[3]),
            #                 ('SSIM-1', ssim[0]), ('SSIM-2', ssim[1]), ('SSIM-3', ssim[2]), ('SSIM-avg', ssim[3])]

            # To HU coordinate
            generate_CT_transpose = tensor_back_to_unMinMax(generate_CT_transpose, opt.CT_MIN_MAX[0],
                                                            opt.CT_MIN_MAX[1]).astype(np.int32) - 1024
            real_CT_transpose = tensor_back_to_unMinMax(real_CT_transpose, opt.CT_MIN_MAX[0], opt.CT_MIN_MAX[1]).astype(
                np.int32) - 1024
            # Save
            # name1 = os.path.splitext(os.path.basename(img_path[0][0]))[0]
            # name2 = os.path.split(os.path.dirname(img_path[0][0]))[-1]
            # name = name2 + '_' + name1
            name = os.path.split(os.path.dirname(img_path[0][0][0]))[-1]
            image_root = os.path.join(web_dir, 'CT', name)
            if not os.path.exists(image_root):
                os.makedirs(image_root)
            # save_path = os.path.join(image_root, 'fake_ct.mha')
            save_path = os.path.join(image_root, 'fake_ct.nii.gz')
            ctVisual.save(generate_CT_transpose.squeeze(0), spacing=(1.0, 1.0, 1.0), origin=(0, 0, 0), path=save_path)
            # save_path = os.path.join(image_root, 'real_ct.mha')
            save_path = os.path.join(image_root, 'real_ct.nii.gz')
            ctVisual.save(real_CT_transpose.squeeze(0), spacing=(1.0, 1.0, 1.0), origin=(0, 0, 0), path=save_path)

        else:
            break
        del visuals, img_path
    webpage.save()
