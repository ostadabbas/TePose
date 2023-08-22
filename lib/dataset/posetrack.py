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

from lib.dataset import Dataset2D
from lib.core.config import POSETRACK_DIR


class PoseTrack(Dataset2D):
    def __init__(self, load_opt, seqlen, vidlen, overlap=0.75, folder=None, debug=False):
        db_name = 'posetrack'
        super(PoseTrack, self).__init__(
            load_opt=load_opt,
            seqlen = seqlen,
            vidlen = vidlen,
            folder=POSETRACK_DIR,
            dataset_name=db_name,
            debug=debug,
            overlap=overlap,
        )
        print(f'{db_name} - number of dataset objects {self.__len__()}')
