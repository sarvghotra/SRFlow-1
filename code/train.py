# Copyright (c) 2020 Huawei Technologies Co., Ltd.
# Licensed under CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International) (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#
# The code is released for academic research use only. For commercial use, please contact Huawei Technologies Co., Ltd.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file contains content licensed by https://github.com/xinntao/BasicSR/blob/master/LICENSE/LICENSE

import os
from os.path import basename
import math
import argparse
import random
import logging
import sys
from collections import OrderedDict

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler

import options.options as option
from utils import util
from data import create_dataloader, create_dataset
from models import create_model
from utils.timer import Timer, TickTock
from utils.util import get_resume_paths


def getEnv(name): import os; return True if name in os.environ.keys() else False


def init_dist(backend='nccl', **kwargs):
    ''' initialization for distributed training'''
    # if mp.get_start_method(allow_none=True) is None:
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn')
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)


def main():
    def SR_validation():
        pass

    #### options
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to option YMAL file.')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)

    #### distributed training settings
    if args.launcher == 'none':
        opt['dist'] = False
        rank = -1
        print('Disabled distributed training.')
    else:
        opt['dist'] = True
        init_dist()
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()

    #opt['dist'] = False
    #rank = -1
    #print('Disabled distributed training.')

    #### loading resume state if exists
    if opt['path'].get('resume_state', None):
        #resume_state_path, _ = get_resume_paths(opt)

        # distributed resuming: all load into default GPU
        device_id = torch.cuda.current_device()
        resume_state = torch.load(opt['path']['resume_state'],
                                    map_location=lambda storage, loc: storage.cuda(device_id))
        option.check_resume(opt, resume_state['iter'])  # check resume options
    else:
        resume_state = None

    #### mkdir and loggers
    if rank <= 0:  # normal training (rank -1) OR distributed training (rank 0)
        if resume_state is None:
            #util.mkdir_and_rename(
            #    opt['path']['experiments_root'])  # rename experiment folder if exists
            util.mkdirs((path for key, path in opt['path'].items() if not key == 'experiments_root'
                         and 'pretrain_model' not in key and 'resume' not in key
                         and 'strict_load' not in key))

        # config loggers. Before it, the log will not work
        util.setup_logger('base', opt['path']['log'], 'train_' + opt['name'], level=logging.INFO,
                          screen=True, tofile=True)
        util.setup_logger('val', opt['path']['log'], 'val_' + opt['name'], level=logging.INFO,
                          screen=True, tofile=True)
        logger = logging.getLogger('base')
        logger.info(option.dict2str(opt))

        # tensorboard logger
        if opt.get('use_tb_logger', False) and 'debug' not in opt['name']:
            version = float(torch.__version__[0:3])
            if version >= 1.1:  # PyTorch 1.1
                from torch.utils.tensorboard import SummaryWriter
            else:
                logger.info(
                    'You are using PyTorch {}. Tensorboard will use [tensorboardX]'.format(version))
                from tensorboardX import SummaryWriter
            '''
            conf_name = basename(args.opt).replace(".yml", "")
            exp_dir = opt['path']['experiments_root']
            log_dir_train = os.path.join(exp_dir, 'tb', conf_name, 'train')
            log_dir_valid = os.path.join(exp_dir, 'tb', conf_name, 'valid')
            tb_logger_train = SummaryWriter(log_dir=log_dir_train)
            tb_logger_valid = SummaryWriter(log_dir=log_dir_valid)
            '''
            tb_logger = SummaryWriter(log_dir=os.path.join(opt['path']['tb_log_dir'], opt['name']))
    else:
        util.setup_logger('base', opt['path']['log'], 'train', level=logging.INFO, screen=True)
        logger = logging.getLogger('base')

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    #### random seed
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    if rank <= 0:
        logger.info('Random seed: {}'.format(seed))
    util.set_random_seed(seed)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    #### create train and val dataloader
    dataset_ratio = 200  # enlarge the size of each epoch
    val_loaders = {}
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = create_dataset(dataset_opt)
            print('Dataset created')
            train_size = int(math.ceil(len(train_set) / dataset_opt['batch_size']))

            #total_epochs = int(math.ceil(total_iters / train_size))
            total_epochs = int(opt['train']['epoch'])
            opt['train']['niter'] = total_epochs * train_size
            total_iters = int(opt['train']['niter'])
            if opt['dist']:
                train_sampler = DistributedSampler(train_set)
            else:
                train_sampler = None
            train_loader = create_dataloader(train_set, dataset_opt, opt, train_sampler)
            if rank <= 0:
                logger.info('Number of train images: {:,d}, iters: {:,d}'.format(
                    len(train_set), train_size))
                logger.info('Total epochs needed: {:d} for iters {:,d}'.format(
                    total_epochs, total_iters))
        else:
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt, opt, None)
            val_loaders[dataset_opt['name']] = val_loader
            if rank <= 0:
                logger.info('Number of val images in [{:s}]: {:d}'.format(
                    dataset_opt['name'], len(val_set)))

    assert train_loader is not None

    # relative learning rate
    if 'train' in opt:
        niter = opt['train']['niter']
        if 'T_period_rel' in opt['train']:
            opt['train']['T_period'] = [int(x * niter) for x in opt['train']['T_period_rel']]
        if 'restarts_rel' in opt['train']:
            opt['train']['restarts'] = [int(x * niter) for x in opt['train']['restarts_rel']]
        if 'lr_steps_rel' in opt['train']:
            opt['train']['lr_steps'] = [int(x * niter) for x in opt['train']['lr_steps_rel']]
        if 'lr_steps_inverse_rel' in opt['train']:
            opt['train']['lr_steps_inverse'] = [int(x * niter) for x in opt['train']['lr_steps_inverse_rel']]

    #### create model
    current_step = 0 if resume_state is None else resume_state['iter']
    model = create_model(opt, current_step)

    #### resume training
    if resume_state:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            resume_state['epoch'], resume_state['iter']))

        start_epoch = resume_state['epoch']
        current_step = resume_state['iter']
        model.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        current_step = 0
        start_epoch = 0

    #### training
    timer = Timer()
    logger.info('Start training from epoch: {:d}, iter: {:d}'.format(start_epoch, current_step))
    timerData = TickTock()

    for epoch in range(start_epoch, total_epochs + 1):
        if opt['dist']:
            train_sampler.set_epoch(epoch)

        timerData.tick()
        for _, train_data in enumerate(train_loader):
            timerData.tock()
            current_step += 1
            if current_step > total_iters:
                break

            #### training
            model.feed_data(train_data)

            #### update learning rate
            model.update_learning_rate(current_step, warmup_iter=opt['train']['warmup_iter'])

            try:
                nll = model.optimize_parameters(current_step)
            except RuntimeError as e:
                print("Skipping ERROR caught in nll = model.optimize_parameters(current_step): ")
                print(e)

            if nll is None:
                nll = 0

            #### log
            def eta(t_iter):
                return (t_iter * (opt['train']['niter'] - current_step)) / 3600

            if current_step % opt['logger']['print_freq'] == 0 and rank <= 0:
                avg_time = timer.get_average_and_reset()
                avg_data_time = timerData.get_average_and_reset()
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}, t:{:.2e}, td:{:.2e}, eta:{:.2e}, nll:{:.3e}> '.format(
                    epoch, current_step, model.get_current_learning_rate(), avg_time, avg_data_time,
                    eta(avg_time), nll)
                #print(message)

                timer.tick()
                if opt['use_tb_logger'] and 'debug' not in opt['name']:
                    tb_logger.add_scalar('train/nll', nll, current_step)
                    tb_logger.add_scalar('train/lr', model.get_current_learning_rate(), current_step)
                    tb_logger.add_scalar('time/iteration', timer.get_last_iteration(), current_step)
                    tb_logger.add_scalar('time/data', timerData.get_last_iteration(), current_step)
                    tb_logger.add_scalar('time/eta', eta(timer.get_last_iteration()), current_step)
                    for k, v in model.get_current_log().items():
                        tb_logger.add_scalar(k, v, current_step)

                if rank <= 0:
                    logger.info(message)
                    sys.stdout.flush()

            # validation
            if current_step % opt['train']['val_freq'] == 0 and rank <= 0:
                logger.info('Saving models and training states.')
                model.save(current_step)
                model.save_training_state(epoch, current_step)

                for val_name, val_loader in val_loaders.items():

                    def compute_metric_SRFlow_validation():
                        avg_psnr = 0.0
                        idx = 0
                        nlls = []
                        for val_data in val_loader:
                            idx += 1
                            model.feed_data(val_data)
                            nll = model.test()

                            if nll is None:
                                nll = 0

                            nlls.append(nll)
                            visuals = model.get_current_visuals()
                            n_samples = 1   #model.n_sample
                            sr_img = None

                            if hasattr(model, 'heats'):
                                sr_img = util.tensor2img(visuals['SR', model.heats[-1], n_samples])  # uint8
                            else:
                                sr_img = util.tensor2img(visuals['SR'])  # uint8
                            assert sr_img is not None

                            # calculate PSNR
                            gt_img = util.tensor2img(visuals['GT'])  # uint8
                            crop_size = opt['scale']
                            gt_img = gt_img / 255.
                            sr_img = sr_img / 255.
                            cropped_sr_img = sr_img[crop_size:-crop_size, crop_size:-crop_size, :]
                            cropped_gt_img = gt_img[crop_size:-crop_size, crop_size:-crop_size, :]
                            avg_psnr += util.calculate_psnr(cropped_sr_img * 255, cropped_gt_img * 255)

                        avg_psnr = avg_psnr / idx
                        avg_nll = sum(nlls) / len(nlls)

                        # log
                        logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
                        #logger_val = logging.getLogger('val')  # validation logger
                        logger.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}'.format(
                            epoch, current_step, avg_psnr))

                        # tensorboard logger
                        if opt['use_tb_logger'] and 'debug' not in opt['name']:
                            tb_logger.add_scalar('psnr/{}'.format(val_name), avg_psnr, current_step)
                            tb_logger.add_scalar('nll/{}'.format(val_name), avg_nll, current_step)

                    def save_imgs_SRFlow_validation():
                        n_samples = 1
                        for val_data in val_loader:
                            visuals = OrderedDict()
                            for heat in opt['val']['heats']:
                                sr_t = model.get_sr(lq=val_data['LQ'], heat=heat)[0]
                                sr = torch.clamp(sr_t, 0, 1).detach().float().cpu()
                                # Assuming that there is only one sample per LR
                                # TODO: Add more samples per heat?
                                visuals[('SR', heat, 0)] = sr

                            visuals['LQ'] = val_data['LQ'].detach()[0].float().cpu()

                            sr_img = None
                            # Save SR images for reference
                            img_name = os.path.splitext(os.path.basename(val_data['LQ_path'][0]))[0]
                            img_dir = os.path.join(opt['datasets'][val_name]['output_val_images'],
                                                    val_name,
                                                    img_name,
                                                    str(current_step))
                            util.mkdir(img_dir)

                            if hasattr(model, 'heats'):
                                for heat in model.heats:
                                    for i in range(n_samples):
                                        sr_img = util.tensor2img(visuals['SR', heat, i])  # uint8
                                        save_img_path = os.path.join(img_dir,
                                                                    '{:s}_{:09d}_h{:03d}_s{:d}.png'.format(img_name,
                                                                                                            current_step,
                                                                                                            int(heat * 100), i))
                                        util.save_img(sr_img, save_img_path)

                            else:
                                sr_img = util.tensor2img(visuals['SR'])  # uint8
                                save_img_path = os.path.join(img_dir,
                                                            '{:s}_{:d}.png'.format(img_name, current_step))
                                util.save_img(sr_img, save_img_path)

                    def SRFlow_validation():
                        if opt['datasets'][val_name]['compute_metrics']:
                            compute_metric_SRFlow_validation()
                        else:
                            save_imgs_SRFlow_validation()

                    def SR_validation():
                        avg_psnr = 0.0
                        idx = 0
                        for val_data in val_loader:
                            idx += 1
                            model.feed_data(val_data)
                            model.test()

                            visuals = model.get_current_visuals()
                            sr_img = util.tensor2img(visuals['SR'])  # uint8
                            gt_img = util.tensor2img(visuals['GT'])  # uint8
                            lq_img = util.tensor2img(visuals['LQ'])  # uint8

                            # Save SR images for reference
                            if opt['datasets'][val_name]['save_output_img']:
                                img_name = os.path.splitext(os.path.basename(val_data['LQ_path'][0]))[0]
                                img_dir = os.path.join(opt['datasets'][val_name]['output_val_images'], val_name, img_name)
                                util.mkdir(img_dir)
                                save_img_path = os.path.join(img_dir,
                                                            '{:s}_{:d}.png'.format(img_name, current_step))
                                util.save_img(sr_img, save_img_path)

                            # calculate LPIPS and PSNR
                            if opt['datasets'][val_name]['compute_metrics']:
                                crop_size = opt['scale']
                                #lpips_score = util.calc_lpips(lpips_metric_model, visuals['GT'], visuals['SR'])
                                gt_img = gt_img / 255.
                                sr_img = sr_img / 255.
                                cropped_sr_img = sr_img[crop_size:-crop_size, crop_size:-crop_size, :]
                                cropped_gt_img = gt_img[crop_size:-crop_size, crop_size:-crop_size, :]
                                avg_psnr += util.calculate_psnr(cropped_sr_img * 255, cropped_gt_img * 255)

                        if opt['datasets'][val_name]['compute_metrics']:
                            avg_psnr = avg_psnr / idx
                            # log
                            logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
                            #logger_val = logging.getLogger('val')  # validation logger
                            logger.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}'.format(
                                epoch, current_step, avg_psnr))

                            # tensorboard logger
                            if opt['use_tb_logger'] and 'debug' not in opt['name']:
                                tb_logger.add_scalar('psnr/{}'.format(val_name), avg_psnr, current_step)

                    if opt['model'] == "SR":
                        SR_validation()
                    elif opt['model'] == "SRFlow":
                        SRFlow_validation()

                    tb_logger.flush()
                    tb_logger.flush()

            timerData.tick()

    with open(os.path.join(opt['path']['root'], "TRAIN_DONE"), 'w') as f:
        f.write("TRAIN_DONE")

    if rank <= 0:
        logger.info('Saving the final model.')
        model.save('latest')

        logger.info('End of training.')


if __name__ == '__main__':
    main()
