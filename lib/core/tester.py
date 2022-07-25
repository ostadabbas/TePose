# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import time
import torch
import shutil
import logging
import numpy as np
import os.path as osp
from progress.bar import Bar

from lib.core.config import BASE_DATA_DIR
from lib.utils.utils import move_dict_to_device, AverageMeter, check_data_pararell
from scipy.stats import bernoulli
from lib.data_utils._kp_utils import convert_kps

from lib.utils.eval_utils import (
    compute_accel,
    compute_error_accel,
    compute_error_verts,
    batch_compute_similarity_transform_torch,
)

logger = logging.getLogger(__name__)


class Tester():
    def __init__(
            self,
            data_loaders,
            generator,
            motion_discriminator,
            gen_optimizer,
            dis_motion_optimizer,
            dis_motion_update_steps,
            end_epoch,
            criterion,
            start_epoch=0,
            lr_scheduler=None,
            motion_lr_scheduler=None,
            device=None,
            writer=None,
            debug=False,
            debug_freq=1000,
            logdir='output',
            resume=None,
            performance_type='min',
            num_iters_per_epoch=1000,
            seqlen = 6,
            vidlen = 1600,
            update_theta_rate = 0.9,
    ):

        # exclude motion discriminator
        self.train_2d_loader, self.train_3d_loader, _, self.valid_loader = data_loaders
        #self.train_2d_loader, self.train_3d_loader, self.valid_loader = data_loaders

        # exclude motion discriminator
        # self.disc_motion_iter = iter(self.disc_motion_loader)

        self.train_2d_iter = self.train_3d_iter = None

        if self.train_2d_loader:
            self.train_2d_iter = iter(self.train_2d_loader)

        if self.train_3d_loader:
            self.train_3d_iter = iter(self.train_3d_loader)

        # Models and optimizers
        self.generator = generator
        self.gen_optimizer = gen_optimizer

        self.motion_discriminator = motion_discriminator
        self.dis_motion_optimizer = dis_motion_optimizer

        # Training parameters
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.criterion = criterion
        self.lr_scheduler = lr_scheduler
        self.motion_lr_scheduler = motion_lr_scheduler
        self.device = device
        self.writer = writer
        self.debug = debug
        self.debug_freq = debug_freq
        self.logdir = logdir

        self.dis_motion_update_steps = dis_motion_update_steps

        self.performance_type = performance_type
        self.train_global_step = 0
        self.valid_global_step = 0
        self.epoch = 0
        self.best_performance = float('inf') if performance_type == 'min' else -float('inf')

        self.evaluation_accumulators = dict.fromkeys(['pred_j3d', 'target_j3d', 'target_theta', 'pred_verts', 'pred_j3d_tsr', 'target_j3d_tsr', 'vidlen_each'])

        self.num_iters_per_epoch = num_iters_per_epoch
        self.seqlen = seqlen
        self.vidlen = vidlen
        self.update_theta_rate = update_theta_rate
        self.repeat_num = 2

        if self.writer is None:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=self.logdir)

        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if len(resume) == 0 or resume[-20:]=='model_best_1.pth.tar':
            # load models for second stage training
            self.load_model_stage_2(resume)
        else:
            # Resume from a pretrained model
            self.resume_pretrained(resume)


    def validate(self):
        self.generator.eval()

        start = time.time()

        summary_string = ''

        if self.evaluation_accumulators is not None:
            for k,v in self.evaluation_accumulators.items():
                self.evaluation_accumulators[k] = []

        J_regressor = torch.from_numpy(np.load(osp.join(BASE_DATA_DIR, 'J_regressor_h36m.npy'))).float()

        for i, target in enumerate(self.valid_loader):
            bar = Bar('Validation (%d/%d)'%(i+1,len(self.valid_loader)), fill='#', max=20)
            move_dict_to_device(target, self.device)
            self.evaluation_accumulators['target_j3d_tsr'].append(target['kp_3d'])
            self.evaluation_accumulators['vidlen_each'].append(target['vidlen_each'])

            for j in range(target['kp_2d'].shape[1]-self.seqlen+1):
                if j==0:
                    if i==0:
                        theta_input = torch.zeros((len(self.valid_loader)*target['kp_2d'].shape[0], target['kp_2d'].shape[1], 85), device=self.device).float().cuda()
                    theta_input[target['index'].view(-1).long(), :self.seqlen-1,:] = target['theta_pseu'][:, :self.seqlen-1,:]
                    pred_j3d_tsr = torch.zeros((target['kp_2d'].shape[0], target['kp_2d'].shape[1], target['kp_3d'].shape[2], 3), device=self.device).float().cuda()

                # <=============
                with torch.no_grad():

                    inp = torch.zeros((target['kp_2d'].shape[0] ,self.seqlen, 2048+85), device=self.device).float().cuda()
                    inp[:, :, :2048] = target['features'][:, j:j+self.seqlen,:]
                    #inp = target['features'][:, j:j+self.seqlen,:]
                    inp[:, :self.seqlen-1, 2048:] = theta_input[target['index'].view(-1).long(), j:j+self.seqlen-1,:]

                    preds = self.generator(inp, J_regressor=J_regressor)
    
                    # convert to 14 keypoint format for evaluation
                    n_kp = preds[-1]['kp_3d'].shape[-2]
                    pred_j3d = preds[-1]['kp_3d'].view(-1, n_kp, 3)
                    #pred_j3d = torch.from_numpy(convert_kps(pred_j3d.cpu().numpy(), src='spin', dst='mpii3d_test')).float().cuda().to(self.device)
                    #n_kp = pred_j3d.shape[-2]
                    target_j3d = target['kp_3d'][:,j+self.seqlen-1,:,:].view(-1, n_kp, 3)
                    pred_verts = preds[-1]['verts'].view(-1, 6890, 3)
                    target_theta = target['theta'][:,j+self.seqlen-1,:].view(-1, 85)
                    theta_input[target['index'].view(-1).long(), j+self.seqlen-1, :] = preds[-1]['theta']
    
                    self.evaluation_accumulators['pred_verts'].append(pred_verts[j < (target['vidlen_each'].view(-1)-self.seqlen+1)])
                    self.evaluation_accumulators['target_theta'].append(target_theta[j < (target['vidlen_each'].view(-1)-self.seqlen+1)])
                    self.evaluation_accumulators['pred_j3d'].append(pred_j3d[j < (target['vidlen_each'].view(-1)-self.seqlen+1)])
                    self.evaluation_accumulators['target_j3d'].append(target_j3d[j < (target['vidlen_each'].view(-1)-self.seqlen+1)])
                # =============>
    
                    pred_j3d_tsr[:, j+self.seqlen-1, :, :] = pred_j3d

                # =============>

                    vidlen_tmp = target['kp_2d'].shape[1]
                    if j%(int(target['kp_2d'].shape[1]/20))==0:
                        batch_time = time.time() - start
                        summary_string = f'({int(j/int(vidlen_tmp/20))}/20) | batch: {batch_time * 10.0:.4}ms | ' \
                             f'Total: {bar.elapsed_td} | ETA: {bar.eta_td:}'

                        bar.suffix = summary_string
                        bar.next()

            self.evaluation_accumulators['pred_j3d_tsr'].append(pred_j3d_tsr)
            bar.finish()

            logger.info(summary_string)

    def test(self):

        self.validate()
        performance = self.evaluate()
        self.writer.close()


    def save_model(self, performance, epoch):
        save_dict = {
            'epoch': epoch,
            'gen_state_dict': self.generator.state_dict(),
            'performance': self.best_performance,
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'gen_optimizer': self.gen_optimizer.state_dict(),
            'disc_motion_state_dict': self.motion_discriminator.state_dict(),
            'disc_motion_optimizer': self.dis_motion_optimizer.state_dict(),
        }

        filename = osp.join(self.logdir, 'checkpoint.pth.tar')
        torch.save(save_dict, filename)

        if self.performance_type == 'min':
            is_best = performance < self.best_performance
        else:
            is_best = performance > self.best_performance

        if is_best:
            logger.info('Best performance achived, saving it!')
            self.best_performance = performance
            shutil.copyfile(filename, osp.join(self.logdir, 'model_best.pth.tar'))

            with open(osp.join(self.logdir, 'best.txt'), 'w') as f:
                f.write(str(float(performance)))

    def load_model_stage_2(self, model_path):
        if len(model_path)==0:
            model_path = osp.join(self.logdir, 'model_best_1.pth.tar')
        if osp.isfile(model_path):
            checkpoint = torch.load(model_path)
            self.generator.load_state_dict(checkpoint['gen_state_dict'])
            self.best_performance = checkpoint['performance']

            if 'disc_motion_optimizer' in checkpoint.keys():
                self.motion_discriminator.load_state_dict(checkpoint['disc_motion_state_dict'])

            logger.info(f"=> loaded checkpoint '{model_path}' "
                  f"(epoch {self.start_epoch}, performance {self.best_performance})")
        else:
            logger.info(f"=> no checkpoint found at '{model_path}'")

    def resume_pretrained(self, model_path):
        if osp.isfile(model_path):
            checkpoint = torch.load(model_path)
            self.start_epoch = checkpoint['epoch']
            self.generator.load_state_dict(checkpoint['gen_state_dict'])
            self.gen_optimizer.load_state_dict(checkpoint['gen_optimizer'])
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            self.best_performance = checkpoint['performance']

            logger.info(f"=> loaded checkpoint '{model_path}' "
                  f"(epoch {self.start_epoch}, performance {self.best_performance})")
        else:
            logger.info(f"=> no checkpoint found at '{model_path}'")

    def evaluate(self):

        for k, v in self.evaluation_accumulators.items():
            self.evaluation_accumulators[k] = torch.cat(v, dim=0)

        pred_j3ds = self.evaluation_accumulators['pred_j3d']
        target_j3ds = self.evaluation_accumulators['target_j3d']

        #pred_j3ds = torch.from_numpy(pred_j3ds).float()
        #target_j3ds = torch.from_numpy(target_j3ds).float()

        print(f'Evaluating on {pred_j3ds.shape[0]} number of poses...')
        pred_pelvis = (pred_j3ds[:,[2],:] + pred_j3ds[:,[3],:]) / 2.0
        target_pelvis = (target_j3ds[:,[2],:] + target_j3ds[:,[3],:]) / 2.0
        #pred_pelvis = pred_j3ds[:, [-3], :]
        #target_pelvis = target_j3ds[:, [-3], :]

        pred_j3ds -= pred_pelvis
        target_j3ds -= target_pelvis
        del pred_pelvis, target_pelvis

        errors = torch.sqrt(((pred_j3ds - target_j3ds) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
        S1_hat = batch_compute_similarity_transform_torch(pred_j3ds.to(self.device), target_j3ds.to(self.device))
        errors_pa = torch.sqrt(((S1_hat - target_j3ds) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()

        del pred_j3ds, target_j3ds

        m2mm = 1000

        vidlen_each = self.evaluation_accumulators['vidlen_each']
        pred_j3ds_tsr = self.evaluation_accumulators['pred_j3d_tsr']
        target_j3ds_tsr = self.evaluation_accumulators['target_j3d_tsr']

        pred_pelvis_tsr = (pred_j3ds_tsr[:,:,[2],:] + pred_j3ds_tsr[:,:,[3],:]) / 2.0
        target_pelvis_tsr = (target_j3ds_tsr[:,[2],:] + target_j3ds_tsr[:,[3],:]) / 2.0
        #pred_pelvis_tsr = pred_j3ds_tsr[:,:,[-3],:]
        #target_pelvis_tsr = target_j3ds_tsr[:,[-3],:]

        pred_j3ds_tsr -= pred_pelvis_tsr
        target_j3ds_tsr -= target_pelvis_tsr
        del pred_pelvis_tsr, target_pelvis_tsr

        accel = compute_accel(pred_j3ds_tsr.cpu(), vidlen_each.cpu(), self.seqlen) * m2mm
        accel_err = compute_error_accel(joints_pred=pred_j3ds_tsr.cpu(), joints_gt=target_j3ds_tsr.cpu(), vidlen_each=vidlen_each.cpu(), seqlen=self.seqlen) * m2mm
        del pred_j3ds_tsr, target_j3ds_tsr

        pred_verts = self.evaluation_accumulators['pred_verts']
        target_theta = self.evaluation_accumulators['target_theta']

        pve = np.mean(compute_error_verts(target_theta=target_theta, pred_verts=pred_verts, device=self.device)) * m2mm

        mpjpe = np.mean(errors) * m2mm
        pa_mpjpe = np.mean(errors_pa) * m2mm

        eval_dict = {
            'mpjpe': mpjpe,
            'pa-mpjpe': pa_mpjpe,
            'accel': accel,
            'pve': pve,
            'accel_err': accel_err
        }

        log_str = f'Epoch {self.epoch+1}, '
        log_str += ' '.join([f'{k.upper()}: {v:.4f},'for k,v in eval_dict.items()])
        logger.info(log_str)

        for k,v in eval_dict.items():
            self.writer.add_scalar(f'error/{k}', v, global_step=self.epoch)

        # return accel_err
        return pa_mpjpe
