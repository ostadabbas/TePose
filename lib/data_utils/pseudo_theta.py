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
import os.path as osp
#os.environ['PYOPENGL_PLATFORM'] = 'egl'

import torch
import joblib
import h5py
import shutil
import colorsys
import argparse
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from lib.core.config import TCMR_DB_DIR

from lib.models.vibe import VIBE

from lib.utils.demo_utils import download_ckpt

MIN_NUM_FRAMES = 25


def main(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    file_name = args.file_name
    db_file = osp.join(TCMR_DB_DIR, file_name + '_db.pt')
            
    # ========= Define VIBE model ========= #
    model = VIBE(
        n_layers=2,
        seqlen=16,
        hidden_size=1024,
        add_linear=True,
        use_residual=True,
    ).to(device)

    # ========= Load pretrained weights ========= #
    pretrained_file = download_ckpt(use_3dpw=True)
    ckpt = torch.load(pretrained_file)
    print(f'Performance of pretrained model on 3DPW: {ckpt["performance"]}')
    ckpt = ckpt['gen_state_dict']
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    print(f'Loaded pretrained weights from \"{pretrained_file}\"')

    # ========= Load features ========= #
    if osp.isfile(db_file):
        db = joblib.load(db_file)
    elif file_name == 'insta_train':
        db = h5py.File(osp.join(TCMR_DB_DIR, file_name + '_db.h5'), 'r')
    else:
        raise ValueError(f'{db_file} do not exists')

    print(f'Loaded dataset from {db_file}')

    video_names, group = np.unique(db['vid_name'], return_index=True)
    perm = np.argsort(group)
    video_names, group = video_names[perm], group[perm]

    indices = np.split(np.arange(0, db['vid_name'].shape[0]), group[1:])

    thetas = []
    with torch.no_grad():
        for idx in tqdm(range(len(video_names))):
            indexes = indices[idx]

            input = torch.from_numpy(db['features'][indexes[0]:indexes[-1]+1]).float()

            for k in range(input.shape[0]//args.vibe_batch_size):
                batch = input[args.vibe_batch_size*k:min(args.vibe_batch_size*(k+1), input.shape[0])]
                batch = batch.unsqueeze(0)
                batch = batch.to(device)
                batch_size, seqlen = batch.shape[:2]
                output = model(batch)[-1]
                thetas.append(output['theta'].reshape(batch_size * seqlen, -1))
            if np.mod(input.shape[0],args.vibe_batch_size) != 0:
                k = input.shape[0]//args.vibe_batch_size
                batch = input[max(0,input.shape[0] - args.vibe_batch_size):]
                batch = batch.unsqueeze(0)
                batch = batch.to(device)
                batch_size, seqlen = batch.shape[:2]
                output = model(batch)[-1]
                thetas.append(output['theta'].reshape(batch_size * seqlen, -1)[k*args.vibe_batch_size-input.shape[0]:])

        thetas = torch.cat(thetas, dim=0).cpu().numpy()

    db_file = osp.join(TCMR_DB_DIR, file_name + '_pseudotheta.pt')
    joblib.dump(thetas, db_file)

    print('================= END =================')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--vibe_batch_size', type=int, default=450,
                        help='batch size of VIBE')

    parser.add_argument('--file_name', type=str, default='mpii3d_train_scale12_occ', choices=['mpii3d_train_scale12_occ', 'mpii3d_train_scale12', 'mpii3d_train_scale12_new_occ', 'mpii3d_train_scale1', 'mpii3d_val_scale12', '3dpw_val', 'posetrack_train', 'posetrack_train_occ', 'insta_train', 'pennaction_train_scale12', '3dpw_test', '3dpw_test_all', '3dpw_train_occ', '3dpw_train', 'h36m_train_25fps_tight', 'h36m_train_25fps', 'h36m_train_25fps_occ', 'h36m_test_25fps', 'h36m_test_front_25fps_tight'],
                        help='file name for input database')

    args = parser.parse_args()

    main(args)
