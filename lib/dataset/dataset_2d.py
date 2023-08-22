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

import os
import torch
import random
import logging
import numpy as np
import os.path as osp
import joblib

from torch.utils.data import Dataset

from lib.core.config import TePose_DB_DIR
from lib.data_utils._kp_utils import convert_kps
from lib.data_utils._img_utils import normalize_2d_kp, transfrom_keypoints, split_into_chunks, combine_into_chunks, get_single_image_crop


logger = logging.getLogger(__name__)


class Dataset2D(Dataset):
    def __init__(self, load_opt, seqlen, vidlen, overlap=0., folder=None, dataset_name=None, debug=False):

        self.load_opt = load_opt
        self.set = 'train'
        self.folder = folder
        self.dataset_name = dataset_name
        self.seqlen = seqlen
        self.vidlen = vidlen
        self.mid_frame = -1 #int(seqlen/2)
        self.stride = int(seqlen * (1-overlap) + 0.5)
        self.debug = debug
        self.db = self.load_db()
        self.vid_indices = combine_into_chunks(self.db['vid_name'], self.seqlen, self.vidlen)

    def __len__(self):
        return len(self.vid_indices)

    def __getitem__(self, index):
        return self.get_single_item(index)

    def load_db(self):

        db_file = osp.join(TePose_DB_DIR, f'{self.dataset_name}_{set}_db.pt')
        if self.set == 'train':
            if self.load_opt == 'repr_wpw_h36m_mpii3d_model':
                if self.dataset_name == 'posetrack':
                    db_file = osp.join(TePose_DB_DIR, f'{self.dataset_name}_{self.set}_db.pt')
                    psetheta_file = osp.join(TePose_DB_DIR, f'{self.dataset_name}_{self.set}_pseudotheta.pt')

            elif self.load_opt == 'repr_wopw_3dpw_model':
                if self.dataset_name == 'posetrack':
                    db_file = osp.join(TePose_DB_DIR, f'{self.dataset_name}_{self.set}_occ_db.pt')
                    psetheta_file = osp.join(TePose_DB_DIR, f'{self.dataset_name}_{self.set}_occ_pseudotheta.pt')

            elif self.load_opt == 'repr_wopw_mpii3d_model':
                if self.dataset_name == 'posetrack':
                    db_file = osp.join(TePose_DB_DIR, f'{self.dataset_name}_{self.set}_db.pt')
                    psetheta_file = osp.join(TePose_DB_DIR, f'{self.dataset_name}_{self.set}_pseudotheta.pt')

            # if self.dataset_name == 'pennaction':
            #     db_file = osp.join(TePose_DB_DIR, f'{self.dataset_name}_{set}_scale12_db.pt')

        if osp.isfile(db_file):
            db = joblib.load(db_file)
        else:
            raise ValueError(f'{db_file} do not exists')

        if osp.isfile(psetheta_file):
            self.psetheta = joblib.load(psetheta_file)
        else:
            raise ValueError(f'{psetheta_file} do not exists')

        print(f'Loaded {self.dataset_name} dataset from {db_file}')
        return db

    def get_sequence(self, start_index, end_index, data):
        if start_index != end_index:
            return data[start_index:end_index+1]
        else:
            return data[start_index:start_index+1].repeat(self.seqlen, axis=0)

    def get_single_item(self, index):
        start_end_index = self.vid_indices[index]

        kp_2d = []
        bbox = []
        input = np.zeros((2, self.vidlen, 2048), dtype=np.float16)
        theta_tensor_pseu = np.zeros((2, self.vidlen, 85), dtype=np.float16)
        switch_id = torch.zeros(2, self.vidlen).float()
        switch_id[0,:] = 1
        switch = 0
        len_tmp = 0
        for k in range(len(start_end_index)):
            input[switch, len_tmp:len_tmp+start_end_index[k][1]-start_end_index[k][0]+1,:] = self.get_sequence(start_end_index[k][0], start_end_index[k][1], self.db['features'])

            theta_tensor_pseu[switch, len_tmp:len_tmp+start_end_index[k][1]-start_end_index[k][0]+1,3:] = self.get_sequence(start_end_index[k][0], start_end_index[k][1], self.psetheta[:,3:])
            theta_tensor_pseu[switch, len_tmp:len_tmp+start_end_index[k][1]-start_end_index[k][0]+1,0] = 1.

            switch_id[switch, len_tmp+self.seqlen-1:len_tmp+start_end_index[k][1]-start_end_index[k][0]+1] = 1
            switch_id[1-switch, len_tmp+self.seqlen-1:len_tmp+start_end_index[k][1]-start_end_index[k][0]+1] = 0
            switch = 1-switch
            len_tmp = len_tmp + start_end_index[k][1]-start_end_index[k][0] - self.seqlen + 2
            if k==0:
                kp_2d_tmp = self.get_sequence(start_end_index[k][0], start_end_index[k][1], self.db['joints2D'])
                kp_2d.append(kp_2d_tmp)
                bbox_tmp = self.get_sequence(start_end_index[k][0], start_end_index[k][1], self.db['bbox'])
                bbox.append(bbox_tmp)
            else:
                kp_2d_tmp = self.get_sequence(start_end_index[k][0]+self.seqlen-1, start_end_index[k][1], self.db['joints2D'])
                kp_2d.append(kp_2d_tmp)
                bbox_tmp = self.get_sequence(start_end_index[k][0]+self.seqlen-1, start_end_index[k][1], self.db['bbox'])
                bbox.append(bbox_tmp)

        kp_2d = np.concatenate(kp_2d, axis=0)
        bbox = np.concatenate(bbox, axis=0)
        if self.dataset_name != 'posetrack':
            kp_2d = convert_kps(kp_2d, src=self.dataset_name, dst='spin')
        kp_2d_tensor = np.ones((self.vidlen, 49, 3), dtype=np.float16)

        for idx in range(kp_2d.shape[0]):
            # crop image and transform 2d keypoints
            kp_2d[idx,:,:2], trans = transfrom_keypoints(
                kp_2d=kp_2d[idx,:,:2],
                center_x=bbox[idx,0],
                center_y=bbox[idx,1],
                width=bbox[idx,2],
                height=bbox[idx,3],
                patch_width=224,
                patch_height=224,
                do_augment=False,
            )
            kp_2d[idx,:,:2] = normalize_2d_kp(kp_2d[idx,:,:2], 224)
            kp_2d_tensor[idx] = kp_2d[idx]

        vidlen_each = torch.zeros(1)
        vidlen_each[0] = np.float(len_tmp + self.seqlen -1)

        target = {
            'features': torch.from_numpy(input).float(),
            'theta_pseu': torch.from_numpy(theta_tensor_pseu).float(), 
            'kp_2d': torch.from_numpy(kp_2d_tensor).float(), # 2D keypoints transformed according to bbox cropping
            'switch_id': switch_id,
            'vidlen_each': vidlen_each.float(),
            # 'instance_id': instance_id,
        }

        if self.debug:

            vid_name = self.db['vid_name'][start_index]

            if self.dataset_name == 'pennaction':
                vid_folder = "frames"
                vid_name = vid_name.split('/')[-1].split('.')[0]
                img_id = "img_name"
            elif self.dataset_name == 'posetrack':
                vid_folder = osp.join('images', vid_name.split('/')[-2])
                vid_name = vid_name.split('/')[-1].split('.')[0]
                img_id = "img_name"
            else:
                vid_name = '_'.join(vid_name.split('_')[:-1])
                vid_folder = 'imageFiles'
                img_id= 'frame_id'
            f = osp.join(self.folder, vid_folder, vid_name)
            video_file_list = [osp.join(f, x) for x in sorted(os.listdir(f)) if x.endswith('.jpg')]
            frame_idxs = self.get_sequence(start_index, end_index, self.db[img_id])
            if self.dataset_name == 'pennaction' or self.dataset_name == 'posetrack':
                video = frame_idxs
            else:
                video = [video_file_list[i] for i in frame_idxs]

            video = torch.cat(
                [get_single_image_crop(image, bbox).unsqueeze(0) for image, bbox in zip(video, bbox)], dim=0
            )

            target['video'] = video

        return target


