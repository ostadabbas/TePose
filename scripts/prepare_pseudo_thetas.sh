#!/usr/bin/env bash
export PYTHONPATH="./:$PYTHONPATH"

python lib/data_utils/pseudo_theta.py --file_name 'mpii3d_train_scale12_occ'

python lib/data_utils/pseudo_theta.py --file_name 'mpii3d_train_scale12'

python lib/data_utils/pseudo_theta.py --file_name 'mpii3d_train_scale12_new_occ'

python lib/data_utils/pseudo_theta.py --file_name 'mpii3d_train_scale1'

python lib/data_utils/pseudo_theta.py --file_name 'mpii3d_val_scale12'

python lib/data_utils/pseudo_theta.py --file_name '3dpw_val'

python lib/data_utils/pseudo_theta.py --file_name 'posetrack_train'

python lib/data_utils/pseudo_theta.py --file_name 'posetrack_train_occ'

python lib/data_utils/pseudo_theta.py --file_name 'insta_train'

#python lib/data_utils/pseudo_theta.py --file_name 'pennaction_train_scale12'

python lib/data_utils/pseudo_theta.py --file_name '3dpw_test'

python lib/data_utils/pseudo_theta.py --file_name '3dpw_test_all'

python lib/data_utils/pseudo_theta.py --file_name '3dpw_train_occ'

python lib/data_utils/pseudo_theta.py --file_name '3dpw_train'

python lib/data_utils/pseudo_theta.py --file_name 'h36m_train_25fps_tight'

python lib/data_utils/pseudo_theta.py --file_name 'h36m_train_25fps'

python lib/data_utils/pseudo_theta.py --file_name 'h36m_train_25fps_occ'

python lib/data_utils/pseudo_theta.py --file_name 'h36m_test_front_25fps_tight'

python lib/data_utils/pseudo_theta.py --file_name 'h36m_test_25fps'

