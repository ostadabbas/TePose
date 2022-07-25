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

from lib.utils.eval_utils import (
    compute_accel,
    compute_error_accel,
    compute_error_verts,
    batch_compute_similarity_transform_torch,
)

logger = logging.getLogger(__name__)


class Trainer():
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

        self.train_2d_loader, self.train_3d_loader, self.disc_motion_loader, self.valid_loader = data_loaders

        self.disc_motion_iter = iter(self.disc_motion_loader)

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

        if len(resume) != 0:
            # Resume from a pretrained model
            self.resume_pretrained(resume)

    def train(self):
        # Single epoch training routine

        losses = AverageMeter()
        kp_2d_loss = AverageMeter()
        kp_3d_loss = AverageMeter()
        # accel_loss = AverageMeter()

        timer = {
            'batch': 0,
        }
        X_update_theta = bernoulli(self.update_theta_rate)

        self.generator.train()
        self.motion_discriminator.train()

        start = time.time()

        summary_string = ''

        bar = Bar(f'Epoch {self.epoch + 1}/{self.end_epoch}', fill='#', max=int(len(self.train_3d_loader)/8))

        target_2d = None
        for i in range(int(len(self.train_3d_loader)/8)):
            if self.train_3d_iter:
                try:
                    target_3d = next(self.train_3d_iter)
                except StopIteration:
                    self.train_3d_iter = iter(self.train_3d_loader)
                    target_3d = next(self.train_3d_iter)
                move_dict_to_device(target_3d, self.device)
            if self.train_2d_iter:
                try:
                    target_2d = next(self.train_2d_iter)
                except StopIteration:
                    self.train_2d_iter = iter(self.train_2d_loader)
                    target_2d = next(self.train_2d_iter)
                move_dict_to_device(target_2d, self.device)

            for j in range(self.num_iters_per_epoch):

                real_motion_samples = None
                try:
                    real_motion_samples = next(self.disc_motion_iter)
                    if real_motion_samples['theta'].shape[0]<target_2d['kp_2d'].shape[0]+target_3d['kp_2d'].shape[0]:
                        real_motion_samples = next(self.disc_motion_iter)
                except StopIteration:
                    self.disc_motion_iter = iter(self.disc_motion_loader)
                    real_motion_samples = next(self.disc_motion_iter)
                    if real_motion_samples['theta'].shape[0] < target_2d['kp_2d'].shape[0]+target_3d['kp_2d'].shape[0]:
                        real_motion_samples = next(self.disc_motion_iter)

                move_dict_to_device(real_motion_samples, self.device)
                
                if j==0:
                    theta_input = torch.zeros((target_2d['kp_2d'].shape[0]+target_3d['kp_2d'].shape[0], self.seqlen-1, 85), device=self.device).float().cuda()
                    theta_input[:target_2d['kp_2d'].shape[0],:,:] = target_2d['theta_pseu'][:,:,:self.seqlen -1, :].view(target_2d['kp_2d'].shape[0]*2, -1)[target_2d['switch_id'][:, :, j + self.seqlen - 1].view(-1).type(torch.bool)].view(target_2d['kp_2d'].shape[0],self.seqlen-1,85).cuda()
                    theta_input[target_2d['kp_2d'].shape[0]:target_2d['kp_2d'].shape[0] + target_3d['kp_2d'].shape[0],:,:] = target_3d['theta_pseu'][:, :self.seqlen-1,:].cuda()

                index_update_theta = X_update_theta.rvs(target_2d['kp_2d'].shape[0]+ target_3d['kp_2d'].shape[0])

                switch_2d = 1 - (target_2d['switch_id'][:,0,j+self.seqlen-1] - target_2d['switch_id'][:,0,max(j+self.seqlen-2,self.seqlen-1)]).abs().view(-1).cpu().numpy()
                index_update_theta[:target_2d['kp_2d'].shape[0]] = index_update_theta[:target_2d['kp_2d'].shape[0]] * switch_2d

                inp = torch.zeros((target_2d['kp_2d'].shape[0] + target_3d['kp_2d'].shape[0],self.seqlen, 2048+85), device=self.device).float().cuda()
                inp[:target_2d['kp_2d'].shape[0], :, :2048] = target_2d['features'][:, :, j:j+self.seqlen,:].view(target_2d['kp_2d'].shape[0]*2, -1)[target_2d['switch_id'][:, :, j + self.seqlen - 1].view(-1).type(torch.bool)].view(target_2d['kp_2d'].shape[0],self.seqlen,2048).cuda()
                inp[target_2d['kp_2d'].shape[0]:target_2d['kp_2d'].shape[0] + target_3d['kp_2d'].shape[0], :, :2048] = target_3d['features'][:, j:j+self.seqlen,:].cuda()

                inp[index_update_theta==1, :self.seqlen-1, 2048:] = theta_input[index_update_theta==1, :,:].cuda()
                inp[index_update_theta==0, :self.seqlen-1, 2048:] = torch.cat((target_2d['theta_pseu'][:,:,j:j+self.seqlen-1,:].view(target_2d['kp_2d'].shape[0]*2, -1)[target_2d['switch_id'][:, :, j + self.seqlen -1].view(-1).type(torch.bool)].view(target_2d['kp_2d'].shape[0], self.seqlen-1, 85)[index_update_theta[:target_2d['kp_2d'].shape[0]]==0], target_3d['theta_pseu'][index_update_theta[target_2d['kp_2d'].shape[0]:target_2d['kp_2d'].shape[0] + target_3d['kp_2d'].shape[0]]==0, j:j+self.seqlen-1, :]), dim=0).cuda()

                start_tmp = time.time()
                theta_input[index_update_theta==0, :,:] = torch.cat((target_2d['theta_pseu'][:,:,j:j+self.seqlen-1,:].view(target_2d['kp_2d'].shape[0]*2, -1)[target_2d['switch_id'][:, :, j + self.seqlen - 1].view(-1).type(torch.bool)].view(target_2d['kp_2d'].shape[0], self.seqlen-1, 85)[index_update_theta[:target_2d['kp_2d'].shape[0]]==0], target_3d['theta_pseu'][index_update_theta[target_2d['kp_2d'].shape[0]:target_2d['kp_2d'].shape[0] + target_3d['kp_2d'].shape[0]]==0, j:j+self.seqlen-1, :]), dim=0).cuda()

                vidlen_each = torch.cat((target_2d['vidlen_each'], target_3d['vidlen_each']), dim=0)
                inp = inp[j < (vidlen_each.view(-1)-self.seqlen+1),:,:]

                if inp.shape[0]>0:

                    preds = self.generator(inp, is_train=True)

                    target_2d_ = target_2d.copy()
                    target_2d_['kp_2d'] =  torch.unsqueeze(target_2d['kp_2d'][j < (target_2d['vidlen_each'].view(-1)-self.seqlen+1),j+self.seqlen-1,:,:],1).repeat(1, self.repeat_num, 1, 1)
                
                    target_3d_ = target_3d.copy()
                    target_3d_['kp_2d'] =  torch.unsqueeze(target_3d['kp_2d'][j < (target_3d['vidlen_each'].view(-1)-self.seqlen+1),j+self.seqlen-1,:,:],1).repeat(1, self.repeat_num, 1, 1)
                    target_3d_['kp_3d'] =  torch.unsqueeze(target_3d['kp_3d'][j < (target_3d['vidlen_each'].view(-1)-self.seqlen+1),j+self.seqlen-1,:,:],1).repeat(1, self.repeat_num, 1, 1)
                    target_3d_['theta'] =  torch.unsqueeze(target_3d['theta'][j < (target_3d['vidlen_each'].view(-1)-self.seqlen+1),j+self.seqlen-1,:],1).repeat(1, self.repeat_num, 1)
                    target_3d_['w_3d'] =  torch.unsqueeze(target_3d['w_3d'][j < (target_3d['vidlen_each'].view(-1)-self.seqlen+1),j+self.seqlen-1],1).repeat(1, self.repeat_num)
                    target_3d_['w_smpl'] =  torch.unsqueeze(target_3d['w_smpl'][j < (target_3d['vidlen_each'].view(-1)-self.seqlen+1),j+self.seqlen-1],1).repeat(1, self.repeat_num)

                   #move_dict_to_device(target_2d_, self.device)
                   #move_dict_to_device(target_3d_, self.device)

                    real_motion_samples['theta'] = real_motion_samples['theta'][:len(vidlen_each.view(-1)), :, :]
                    real_motion_samples['theta'] = real_motion_samples['theta'][j < (vidlen_each.view(-1)-self.seqlen+1),:,:]

                    #total_previous_thetas = torch.cat((target_2d['theta_pseu'][:,:,j:j+self.seqlen-1,:].view(target_2d['kp_2d'].shape[0]*2, -1)[target_2d['switch_id'][:, :, j + self.seqlen -1].view(-1).type(torch.bool)].view(target_2d['kp_2d'].shape[0], self.seqlen-1, 85), target_3d['theta'][:, j:j+self.seqlen-1, :]), dim=0).cuda()
                    total_previous_thetas = theta_input.detach()
                    total_previous_thetas = total_previous_thetas[j < (vidlen_each.view(-1)-self.seqlen+1),:,:]

                    gen_loss, motion_dis_loss, loss_dict = self.criterion(
                        generator_outputs=preds,
                        data_2d=target_2d_,
                        data_3d=target_3d_,
                        pre_mosh=total_previous_thetas,
                        data_motion_mosh=real_motion_samples,
                        motion_discriminator=self.motion_discriminator,
                    )

                    # <======= Backprop generator and discriminator
                    self.gen_optimizer.zero_grad()
                    gen_loss.backward()
                    self.gen_optimizer.step()

                    if (j % self.dis_motion_update_steps == 0) & (motion_dis_loss.item()!=0):
                        self.dis_motion_optimizer.zero_grad()
                        motion_dis_loss.backward()
                        self.dis_motion_optimizer.step()
                    # =======>

                    theta_input[j < (vidlen_each.view(-1)-self.seqlen+1),:self.seqlen-2,:] = theta_input[j < (vidlen_each.view(-1)-self.seqlen+1),1:self.seqlen-1,:]

                    theta_input[j < (vidlen_each.view(-1)-self.seqlen+1),self.seqlen-2,:] = preds[-1]['theta'][:, :].mean(dim=1).detach()

                    # <======= Log training info
                    total_loss = gen_loss

                    losses.update(total_loss.item(), inp.size(0))
                    kp_2d_loss.update(loss_dict['loss_kp_2d'].item(), inp.size(0))
                    kp_3d_loss.update(loss_dict['loss_kp_3d'].item(), inp.size(0))

            timer['batch'] = time.time() - start
            start = time.time()

            summary_string = f'({i + 1}/{int(len(self.train_3d_loader)/8+1)}) | Total: {bar.elapsed_td} | ' \
                             f'ETA: {bar.eta_td:} | loss: {losses.avg:.2f} | 2d: {kp_2d_loss.avg:.2f} ' \
                             f'| 3d: {kp_3d_loss.avg:.2f} '

            for k, v in loss_dict.items():
                summary_string += f' | {k}: {v:.3f}'
                self.writer.add_scalar('train_loss/'+k, v, global_step=self.train_global_step)

            for k,v in timer.items():
                summary_string += f' | {k}: {v:.2f}'

            self.writer.add_scalar('train_loss/loss', total_loss.item(), global_step=self.train_global_step)

            if self.debug:
                print('==== Visualize ====')
                from lib.utils.vis import batch_visualize_vid_preds
                video = target_3d['video']
                dataset = 'spin'
                vid_tensor = batch_visualize_vid_preds(video, preds[-1], target_3d.copy(),
                                                       vis_hmr=False, dataset=dataset)
                self.writer.add_video('train-video', vid_tensor, global_step=self.train_global_step, fps=10)

            self.train_global_step += 1
            bar.suffix = summary_string
            bar.next()

            if torch.isnan(total_loss):
                print(total_loss)
                #exit('Nan value in loss, exiting!...')
            # =======>

        bar.finish()

        logger.info(summary_string)

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

    def fit(self):

        for epoch in range(self.start_epoch, self.end_epoch):
            self.epoch = epoch
            self.train()
            self.validate()
            performance = self.evaluate()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step(performance)

            if self.motion_lr_scheduler is not None:
                self.motion_lr_scheduler.step(performance)

            # log the learning rate
            for param_group in self.gen_optimizer.param_groups:
                #print(f'Learning rate {param_group["lr"]}')
                logger.info(f'Learning rate {param_group["lr"]}')
                self.writer.add_scalar('lr/gen_lr', param_group['lr'], global_step=self.epoch)

            for param_group in self.dis_motion_optimizer.param_groups:
                print(f'Learning rate {param_group["lr"]}')
                self.writer.add_scalar('lr/dis_lr', param_group['lr'], global_step=self.epoch)

            logger.info(f'Epoch {epoch+1} performance: {performance:.4f}')

            self.save_model(performance, epoch)

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

    def resume_pretrained(self, model_path):
        if osp.isfile(model_path):
            checkpoint = torch.load(model_path)
            self.start_epoch = checkpoint['epoch']+1
            self.generator.load_state_dict(checkpoint['gen_state_dict'])
            self.gen_optimizer.load_state_dict(checkpoint['gen_optimizer'])
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            self.best_performance = checkpoint['performance']

            if 'disc_motion_optimizer' in checkpoint.keys():
                self.motion_discriminator.load_state_dict(checkpoint['disc_motion_state_dict'])
                self.dis_motion_optimizer.load_state_dict(checkpoint['disc_motion_optimizer'])

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
