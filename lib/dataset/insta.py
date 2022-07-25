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

import h5py
import torch
import logging
import numpy as np
import os.path as osp
import joblib

from torch.utils.data import Dataset
from lib.core.config import TePose_DB_DIR
from lib.data_utils._kp_utils import convert_kps
from lib.data_utils._img_utils import normalize_2d_kp, split_into_chunks, combine_into_chunks

logger = logging.getLogger(__name__)

class Insta(Dataset):
    def __init__(self, load_opt, seqlen, vidlen, overlap=0., debug=False):
        self.seqlen = seqlen
        self.vidlen = vidlen
        self.h5_file = osp.join(TePose_DB_DIR, 'insta_train_db.h5')
        psetheta_file = osp.join(TePose_DB_DIR, 'insta_train_pseudotheta.pt')

        self.theta = joblib.load(psetheta_file)
        with h5py.File(self.h5_file, 'r') as db:
            self.db = db
            self.vid_indices = combine_into_chunks(self.db['vid_name'], self.seqlen, self.vidlen)

        print(f'InstaVariety number of dataset objects {self.__len__()}')

    def __len__(self):
        return len(self.vid_indices)

    def __getitem__(self, index):
        return self.get_single_item(index)

    def get_sequence(self, start_index, end_index, data):
        if start_index != end_index:
            return data[start_index:end_index+1]
        else:
            return data[start_index:start_index+1].repeat(self.seqlen, axis=0)

    def get_single_item(self, index):
        start_end_index = self.vid_indices[index]

        with h5py.File(self.h5_file, 'r') as db:
            self.db = db

            kp_2d = []
            input = np.zeros((2, self.vidlen, 2048), dtype=np.float16)
            theta_tensor_pseu = np.zeros((2, self.vidlen, 85), dtype=np.float16)
            switch_id = torch.zeros(2, self.vidlen).float()
            switch_id[0,:] = 1
            switch = 0
            len_tmp = 0
            for k in range(len(start_end_index)):
                input[switch, len_tmp:len_tmp+start_end_index[k][1]-start_end_index[k][0]+1,:] = self.get_sequence(start_end_index[k][0], start_end_index[k][1], self.db['features'])

                theta_tensor_pseu[switch, len_tmp:len_tmp+start_end_index[k][1]-start_end_index[k][0]+1,3:] = self.get_sequence(start_end_index[k][0], start_end_index[k][1], self.theta[:,3:])
                theta_tensor_pseu[switch, len_tmp:len_tmp+start_end_index[k][1]-start_end_index[k][0]+1,0] = 1.

                switch_id[switch, len_tmp+self.seqlen-1:len_tmp+start_end_index[k][1]-start_end_index[k][0]+1] = 1
                switch_id[1-switch, len_tmp+self.seqlen-1:len_tmp+start_end_index[k][1]-start_end_index[k][0]+1] = 0
                switch = 1-switch
                len_tmp = len_tmp + start_end_index[k][1]-start_end_index[k][0] - self.seqlen + 2
                if k==0:
                    kp_2d_tmp = self.get_sequence(start_end_index[k][0], start_end_index[k][1], self.db['joints2D'])
                    kp_2d.append(kp_2d_tmp)
                else:
                    kp_2d_tmp = self.get_sequence(start_end_index[k][0]+self.seqlen-1, start_end_index[k][1], self.db['joints2D'])
                    kp_2d.append(kp_2d_tmp)

            kp_2d = np.concatenate(kp_2d, axis=0)
            kp_2d = convert_kps(kp_2d, src='insta', dst='spin')
            kp_2d_tensor = np.ones((self.vidlen, 49, 3), dtype=np.float16)

            #vid_name = self.get_sequence(start_index, end_index, self.db['vid_name'])
            #frame_id = self.get_sequence(start_index, end_index, self.db['frame_id']).astype(str)
            #instance_id = np.array([v.decode('ascii') + f for v, f in zip(vid_name, frame_id)])

        for idx in range(kp_2d.shape[0]):
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
            # 'instance_id': instance_id
        }

        return target
