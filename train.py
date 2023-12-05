# ------------------------------------------------------------------------------
# Copyright (c) Tencent
# Licensed under the GPLv3 License.
# Created by Kai Ma (makai0324@gmail.com)
# ------------------------------------------------------------------------------

import argparse
from lib.config.config import cfg_from_yaml, cfg, merge_dict_and_yaml, print_easy_dict
from lib.dataset.factory import get_dataset
from lib.model.factory import get_model
import copy
import torch
import time
import os
from torch import nn
# import torch.nn.utils.prune as prune
import torch.nn.functional as F
from apex import amp


def parse_args():
    parse = argparse.ArgumentParser(description='CTGAN')
    parse.add_argument('--data', type=str,
                       # default='LIDC256',
                       # default='VerSe_RF_noffl_256',
                       # default='VerSe_RF_crop',
                       # default='VerSe_apex',
                       # default='VerSe_apex_rotate15_random',
                       # default='VerSe_apex_transformer',
                       # default='VerSe_apex_transformer_twin',
                       # default='VerSe_spine_transformer_twin',
                       # default='VerSe_stl',
                       # default='VerSe_igdl',
                       # default='VerSe_aspp_igdl_3D',
                       # default='VerSe_aspp',
                       # default='VerSe_aspp_finetune',
                       # default='VerSe_aspp2',
                       # default='VerSe_apex_transformer_ffl',
                       # default='VerSe_apex_gai',
                       # default='VerSe_RF_crop128',
                       # default='VerSe_SS',
                       # default='VerSe_ffl',
                       # default='VerSe_aspp_ffl_3D',
                       # default='VerSe_128_base',
                       # default='VerSe_128_base_SmoothL1Loss',
                       # default='VerSe_128_base_psnr',
                       # default='VerSe_128_base_psnr3d',
                       # default='VerSe_128_base_ssim',
                       default='VerSe_aspp_ssim',
                       # default='VerSe_aspp_transformer05_twin_igdl2D_ffl2D',
                       # default='VerSe_aspp_conformer_igdl2D_ffl2D',
                       # default='VerSe_aspp_conformer_resize128',
                       # default='VerSe_aspp_conformer',
                       # default='VerSe_conformer',
                       dest='data',
                       help='input data ')
    parse.add_argument('--tag', type=str,
                       default='d2_multiview2500',
                       # default='d2_singleview2500',
                       dest='tag',
                       help='distinct from other try')
    parse.add_argument('--dataroot', type=str,
                       # default='/media/gy/Data/2019-CVPR-X2CT/LIDC-HDF5-256',
                       # default='/media/gy/Data/VerSe/ctpro_rotate15_random_crop128/',
                       default='/media/gy/Data/VerSe/',
                       # default='/home/gy/VerSe/',
                       dest='dataroot',
                       help='input data root')
    parse.add_argument('--dataset', type=str,
                       default='train',
                       dest='dataset',
                       help='Train or test or valid')
    parse.add_argument('--valid_dataset', type=str, default='test', dest='valid_dataset',
                       help='Train or test or valid')
    parse.add_argument('--datasetfile', type=str,
                       # default='./data/train-LIDC.txt',
                       # default='./data/verse_RF128_train.txt',
                       # default='./data/verse_RF128_test.txt',
                       default='./data/verse_train.txt',
                       # default='./data/verse_RF128+spine_train.txt',
                       # default='./data/c_test.txt',
                       # default='./data/verse_RF_crop_train.txt',
                       dest='datasetfile',
                       help='Train or test or valid file path')
    parse.add_argument('--valid_datasetfile', type=str,
                       # default='./data/test-LIDC.txt',
                       # default='./data/verse_RF128_test.txt',
                       default='./data/verse_test.txt',
                       # default='./data/verse_RF128+spine_test.txt',
                       # default='./data/verse_RF_crop_test.txt',
                       dest='valid_datasetfile',
                       help='Train or test or valid file path')
    parse.add_argument('--ymlpath', type=str,
                       default='./experiment/multiview2500/d2_multiview2500.yml',
                       # default='./experiment/singleview2500/d2_singleview2500.yml',
                       dest='ymlpath',
                       help='config have been modified')
    parse.add_argument('--gpu', type=str,
                       default='0',
                       # default='1,0',
                       # default='2',
                       dest='gpuid',
                       help='gpu is split by ,')
    parse.add_argument('--dataset_class', type=str,
                       default='align_ct_xray_views_std',
                       # default='align_ct_xray_std',
                       dest='dataset_class',
                       help='Dataset class should select from align /')
    parse.add_argument('--model_class', type=str,
                       default='MultiViewCTGAN',
                       # default='SingleViewCTGAN',
                       dest='model_class',
                       help='Model class should select from simpleGan / ')
    parse.add_argument('--check_point', type=str,
                       default=None,
                       # default='50',
                       dest='check_point',
                       help='which epoch to load? ')
    parse.add_argument('--load_path', type=str,
                       default=None,
                       # default='/home/gy/SE/X2CT/3DGAN/save_models/multiView_CTGAN/VerSe_apex/d2_multiview2500/checkpoint',
                       dest='load_path',
                       help='if load_path is not None, model will load from load_path')
    parse.add_argument('--latest', action='store_true', dest='latest',
                       help='set to latest to use latest cached model')
    parse.add_argument('--verbose', action='store_true', dest='verbose',
                       help='if specified, print more debugging information')
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
        args.epoch_count = int(args.check_point) + 1

    # merge config with yaml
    if args.ymlpath is not None:
        cfg_from_yaml(args.ymlpath)
    # merge config with argparse
    opt = copy.deepcopy(cfg)
    opt = merge_dict_and_yaml(args.__dict__, opt)
    print_easy_dict(opt)

    # add data_augmentation
    datasetClass, augmentationClass, dataTestClass, collateClass = get_dataset(opt.dataset_class)
    opt.data_augmentation = augmentationClass

    # valid dataset
    if args.valid_dataset is not None:
        valid_opt = copy.deepcopy(opt)
        valid_opt.data_augmentation = dataTestClass
        valid_opt.datasetfile = opt.valid_datasetfile


        valid_dataset = datasetClass(valid_opt)
        print('Valid DataSet is {}'.format(valid_dataset.name))
        valid_dataloader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=int(valid_opt.nThreads),
            collate_fn=collateClass)
        valid_dataset_size = len(valid_dataloader)
        print('#validation images = %d' % valid_dataset_size)
    else:
        valid_dataloader = None

    # get dataset
    dataset = datasetClass(opt)
    print('DataSet is {}'.format(dataset.name))
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=int(opt.nThreads),
        collate_fn=collateClass)

    dataset_size = len(dataloader)
    print('#training images = %d' % dataset_size)

    # get model
    gan_model = get_model(opt.model_class)()
    print('Model --{}-- will be Used'.format(gan_model.name))
    gan_model.init_process(opt)
    total_steps, epoch_count = gan_model.setup(opt)

    # prune
    # for name, module in gan_model.named_modules():
    #     # prune 20% of connections in all 2D-conv layers
    #     if isinstance(module, torch.nn.Conv2d):
    #         prune.random_unstructured(module, name='weight', amount=0.2)
    #         # 将所有卷积层的权重减去 20%
    #     # prune 40% of connections in all linear layers
    #     elif isinstance(module, torch.nn.Linear):
    #         prune.l1_unstructured(module, name='weight', amount=0.4)
    #         # 将所有全连接层的权重减去 40%
    #     # prune 30% of connections in all linear layers
    #     elif isinstance(module, torch.nn.Conv3d):
    #         prune.random_unstructured(module, name='weight', amount=0.3)
    #         # 将所有Conv3d的权重减去 30%
    # # print(dict(gan_model.named_buffers()).keys())  # to verify that all masks exist

    # set to train
    gan_model.train()

    # visualizer
    from lib.utils.visualizer import Visualizer
    visualizer = Visualizer(log_dir=os.path.join(gan_model.save_root, 'train_log'))

    total_steps = total_steps

    # train discriminator more
    dataloader_iter_for_discriminator = iter(dataloader)

    # train
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()

        for epoch_i, data in enumerate(dataloader):

            iter_start_time = time.time()

            total_steps += 1

            gan_model.set_input(data)
            t0 = time.time()
            gan_model.optimize_parameters()
            t1 = time.time()

            # if total_steps == 1:
            #   visualizer.add_graph(model=gan_model, input=gan_model.forward())

            # # visual gradient
            if opt.verbose and total_steps % opt.print_freq == 0:
              for name, para in gan_model.named_parameters():
                visualizer.add_histogram('Grad_' + name, para.grad.data.clone().cpu().numpy(), step=total_steps)
                visualizer.add_histogram('Weight_' + name, para.data.clone().cpu().numpy(), step=total_steps)
              for name in gan_model.model_names:
                net = getattr(gan_model, 'net' + name)
                if hasattr(net, 'output_dict'):
                  for name, out in net.output_dict.items():
                    visualizer.add_histogram(name, out.numpy(), step=total_steps)

            # loss
            loss_dict = gan_model.get_current_losses()
            visualizer.add_scalars('Train_Loss', loss_dict, step=total_steps)
            total_loss = visualizer.add_total_scalar('Total loss', loss_dict, step=total_steps)
            visualizer.add_average_scalers('Epoch Loss', loss_dict, step=total_steps, write=False)
            visualizer.add_average_scalar('Epoch total Loss', total_loss)

            # metrics
            # metrics_dict = gan_model.get_current_metrics()
            # visualizer.add_scalars('Train_Metrics', metrics_dict, step=total_steps)
            # visualizer.add_average_scalers('Epoch Metrics', metrics_dict, step=total_steps, write=False)

            if total_steps % opt.print_freq == 0:
                print('total step: {} timer: {:.4f} sec.'.format(total_steps, t1 - t0))
                print('epoch {}/{}, step{}:{} || total loss:{:.4f}'.format(epoch, opt.niter + opt.niter_decay,
                                                                           epoch_i, dataset_size, total_loss))
                print('||'.join(['{}: {:.4f}'.format(k, v) for k, v in loss_dict.items()]))
                # print('||'.join(['{}: {:.4f}'.format(k, v) for k, v in metrics_dict.items()]))
                print('')

            if total_steps % opt.print_img_freq == 0:
              visualizer.add_image('Image', gan_model.get_current_visuals(), gan_model.get_normalization_list(), total_steps)

            '''
            WGAN
            '''
            if (opt.critic_times - 1) > 0:
                for critic_i in range(opt.critic_times - 1):
                    try:
                        data = next(dataloader_iter_for_discriminator)
                        gan_model.set_input(data)
                        gan_model.optimize_D()
                    except:
                        dataloader_iter_for_discriminator = iter(dataloader)
            del(loss_dict)

        # # save model every epoch
        # print('saving the latest model (epoch %d, total_steps %d)' %
        #       (epoch, total_steps))
        # gan_model.save_networks(epoch, total_steps, True)

        # save model several epoch
        if epoch % opt.save_epoch_freq == 0 and epoch >= opt.begin_save_epoch:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            gan_model.save_networks(epoch, total_steps)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        ##########
        # For speed
        ##########
        # visualizer.add_image('Image_Epoch', gan_model.get_current_visuals(), gan_model.get_normalization_list(), epoch)
        # visualizer.add_average_scalers('Epoch Loss', None, step=epoch, write=True)
        # visualizer.add_average_scalar('Epoch total Loss', None, step=epoch, write=True)

        # visualizer.add_average_scalers('Epoch Metrics', None, step=epoch, write=True)

        # visualizer.add_scalar('Learning rate', gan_model.optimizers[0].param_groups[0]['lr'], epoch)
        gan_model.update_learning_rate(epoch)

        # # Test
        # if args.valid_dataset is not None:
        #   if epoch % opt.save_epoch_freq == 0 or epoch==1:
        #     gan_model.eval()
        #     iter_valid_dataloader = iter(valid_dataloader)
        #     for v_i in range(len(valid_dataloader)):
        #       data = next(iter_valid_dataloader)
        #       gan_model.set_input(data)
        #       gan_model.test()
        #
        #       if v_i < opt.howmany_in_train:
        #         visualizer.add_image('Test_Image', gan_model.get_current_visuals(), gan_model.get_normalization_list(), epoch*10+v_i, max_image=25)
        #
        #       # metrics
        #       metrics_dict = gan_model.get_current_metrics()
        #       visualizer.add_average_scalers('Epoch Test_Metrics', metrics_dict, step=total_steps, write=False)
        #     visualizer.add_average_scalers('Epoch Test_Metrics', None, step=epoch, write=True)
        #
        #     gan_model.train()
