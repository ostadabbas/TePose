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
from lib.core.config import BASE_DATA_DIR
from lib.models.smpl import SMPL, SMPL_MODEL_DIR
#os.environ['PYOPENGL_PLATFORM'] = 'egl'

import cv2
import time
import torch
import joblib
import shutil
import colorsys
import argparse
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm
from multi_person_tracker import MPT
from torch.utils.data import DataLoader
from lib.dataset.inference import Inference

from lib.models.tepose import TePose
from lib.models.vibe import VIBE
from lib.utils.demo_utils import download_ckpt
from lib.utils.renderer import Renderer
from lib.utils.smooth_pose import smooth_pose
from lib.dataset.dataset_demo import CropDataset, FeatureDataset
from lib.utils.demo_utils import (
    download_youtube_clip,
    convert_crop_cam_to_orig_img,
    convert_crop_coords_to_orig_img,
    prepare_rendering_results,
    video_to_images,
    images_to_video,
)

MIN_NUM_FRAMES = 25
random.seed(1)
torch.manual_seed(1)
np.random.seed(1)


def main(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


    """ Prepare input video (images) """
    video_file = args.vid_file
    if video_file.startswith('https://www.youtube.com'):
        print(f"Donwloading YouTube video \'{video_file}\'")
        video_file = download_youtube_clip(video_file, '/tmp')
        if video_file is None:
            exit('Youtube url is not valid!')
        print(f"YouTube Video has been downloaded to {video_file}...")

    if not os.path.isfile(video_file):
        exit(f"Input video \'{video_file}\' does not exist!")

    output_path = osp.join('./output/demo_output', os.path.basename(video_file).replace('.mp4', ''))
    Path(output_path).mkdir(parents=True, exist_ok=True)
    image_folder, num_frames, img_shape = video_to_images(video_file, return_info=True)

    print(f"Input video number of frames {num_frames}\n")
    orig_height, orig_width = img_shape[:2]


    """ Run tracking """
    total_time = time.time()
    bbox_scale = 1.2
    # run multi object tracker
    mot = MPT(
        device=device,
        batch_size=args.tracker_batch_size,
        display=args.display,
        detector_type=args.detector,
        output_format='dict',
        yolo_img_size=args.yolo_img_size,
    )
    tracking_results = mot(image_folder)

    # remove tracklets if num_frames is less than MIN_NUM_FRAMES
    for person_id in list(tracking_results.keys()):
        if tracking_results[person_id]['frames'].shape[0] < MIN_NUM_FRAMES:
            del tracking_results[person_id]


    """ Get VIBE model """
    model_vibe = VIBE(
        n_layers=2,
        batch_size=450,
        seqlen=16,
        hidden_size=1024,
        pretrained='/mnt/e/Sony/TePose-master/data/base_data/spin_model_checkpoint.pth.tar',
        add_linear=True,
        bidirectional=False,
        use_residual=True,
    ).to(device)

    # Get feature_extractor
    from lib.models.spin import hmr
    hmr = hmr().to(device)
    #checkpoint = torch.load(osp.join(BASE_DATA_DIR, 'spin_model_checkpoint.pth.tar'))
    checkpoint = torch.load('/mnt/e/Sony/TePose-master/data/base_data/spin_model_checkpoint.pth.tar')
    hmr.load_state_dict(checkpoint['model'], strict=False)
    hmr.eval()

    # ========= Load pretrained weights for VIBE ========= #
    pretrained_file = download_ckpt(use_3dpw=False)
    ckpt = torch.load(pretrained_file)
    print(f'Performance of pretrained model on 3DPW: {ckpt["performance"]}')
    ckpt = ckpt['gen_state_dict']
    model_vibe.load_state_dict(ckpt, strict=False)
    model_vibe.eval()
    print(f'Loaded pretrained weights from \"{pretrained_file}\"')


    """ Get TePose model """
    seq_len = 6
    model = TePose(
        seqlen=6,
        n_layers=2,
        hidden_size=1024
    ).to(device)

    # Load pretrained weights
    pretrained_file = args.model
    ckpt = torch.load(pretrained_file)
    print(f"Load pretrained weights from \'{pretrained_file}\'")
    ckpt = ckpt['gen_state_dict']
    model.load_state_dict(ckpt, strict=False)

    # Change mesh gender
    gender = args.gender  # 'neutral', 'male', 'female'
    model.regressor.smpl = SMPL(
        SMPL_MODEL_DIR,
        batch_size=64,
        create_transl=False,
        gender=gender
    ).cuda()

    model.eval()



    """ Run TePose on each person """
    print("\nRunning TePose on each person tracklet...")
    tepose_time = time.time()
    tepose_results = {}
    for person_id in tqdm(list(tracking_results.keys())):
        bboxes = joints2d = None
        bboxes = tracking_results[person_id]['bbox']
        frames = tracking_results[person_id]['frames']

        # Prepare static image features
        dataset = CropDataset(
            image_folder=image_folder,
            frames=frames,
            bboxes=bboxes,
            joints2d=joints2d,
            scale=bbox_scale,
        )

        bboxes = dataset.bboxes
        frames = dataset.frames
        has_keypoints = True if joints2d is not None else False

        crop_dataloader = DataLoader(dataset, batch_size=32, num_workers=8)

        with torch.no_grad():
            feature_list = []
            for i, batch in enumerate(crop_dataloader):
                if has_keypoints:
                    batch, nj2d = batch
                    norm_joints2d.append(nj2d.numpy().reshape(-1, 21, 3))

                batch = batch.to(device)
                feature = hmr.feature_extractor(batch.reshape(-1,3,224,224))
                feature_list.append(feature.cpu())

            del batch

            feature_list = torch.cat(feature_list, dim=0)

        # Encode temporal features and estimate 3D human mesh
        dataset = FeatureDataset(
            image_folder=image_folder,
            frames=frames,
            seq_len=num_frames,
        )
        dataset.feature_list = feature_list

        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)
        with torch.no_grad():
            pred_cam, pred_verts, pred_pose, pred_betas, pred_joints3d, smpl_joints2d, norm_joints2d = [], [], [], [], [], [], []

            for i, batch in enumerate(dataloader):

                if has_keypoints:
                    batch, nj2d = batch
                    norm_joints2d.append(nj2d.numpy().reshape(-1, 21, 3))

                batch = batch.to(device)

                #seqlen_, nc, h, w = batch.shape

                #feature = hmr.feature_extractor(batch.reshape(-1, nc, h, w))

                #feature = feature.reshape(1, seqlen_, -1)

                #batch = batch.unsqueeze(0)

                batch_size, seqlen = batch.shape[:2]
                output = model_vibe(batch)[-1]

                pred_cam.append(output['theta'][0,:seq_len-1, :3].clone())
                pred_verts.append(output['verts'][0,:seq_len-1].clone())
                pred_pose.append(output['theta'][0,:seq_len-1, 3:75].clone())
                pred_betas.append(output['theta'][0,:seq_len-1, 75:].clone())
                pred_joints3d.append(output['kp_3d'][0,:seq_len-1].clone())
                smpl_joints2d.append(output['kp_2d'].reshape(batch_size * seqlen, -1, 2)[:seq_len-1].clone())
                theta_input = output['theta'][:,:seq_len-1, :].detach().clone()
                for k in range(num_frames-seq_len+1):
                    inp = torch.zeros((1,seq_len, 2048+85), device=device).float().cuda()
                    inp[0,:,:2048] = batch[:,k:k+seq_len,:].clone()
                    inp[:,:seq_len-1,2048:] = theta_input.clone().cuda()
                    output = model(inp)[0]

                    pred_cam.append(output['theta'][:, :3].clone())
                    pred_verts.append(output['verts'].clone())
                    pred_pose.append(output['theta'][:, 3:75].clone())
                    pred_betas.append(output['theta'][:, 75:].clone())
                    pred_joints3d.append(output['kp_3d'].clone())
                    smpl_joints2d.append(output['kp_2d'].clone().reshape(1, -1, 2))
                    
                    theta_input[:,:seq_len-2,:] = theta_input[:,1:seq_len-1,:].clone()
                    theta_input[:,seq_len-2,:] = output['theta'].clone().detach()


            pred_cam = torch.cat(pred_cam, dim=0)
            pred_verts = torch.cat(pred_verts, dim=0)
            pred_pose = torch.cat(pred_pose, dim=0)
            pred_betas = torch.cat(pred_betas, dim=0)
            pred_joints3d = torch.cat(pred_joints3d, dim=0)
            smpl_joints2d = torch.cat(smpl_joints2d, dim=0)

            del batch

        # ========= [Optional] run Temporal SMPLify to refine the results ========= #
        if args.run_smplify and args.tracking_method == 'pose':
            norm_joints2d = np.concatenate(norm_joints2d, axis=0)
            norm_joints2d = convert_kps(norm_joints2d, src='staf', dst='spin')
            norm_joints2d = torch.from_numpy(norm_joints2d).float().to(device)

            # Run Temporal SMPLify
            update, new_opt_vertices, new_opt_cam, new_opt_pose, new_opt_betas, \
            new_opt_joints3d, new_opt_joint_loss, opt_joint_loss = smplify_runner(
                pred_rotmat=pred_pose,
                pred_betas=pred_betas,
                pred_cam=pred_cam,
                j2d=norm_joints2d,
                device=device,
                batch_size=norm_joints2d.shape[0],
                pose2aa=False,
            )

            # update the parameters after refinement
            print(f'Update ratio after Temporal SMPLify: {update.sum()} / {norm_joints2d.shape[0]}')
            pred_verts = pred_verts.cpu()
            pred_cam = pred_cam.cpu()
            pred_pose = pred_pose.cpu()
            pred_betas = pred_betas.cpu()
            pred_joints3d = pred_joints3d.cpu()
            pred_verts[update] = new_opt_vertices[update]
            pred_cam[update] = new_opt_cam[update]
            pred_pose[update] = new_opt_pose[update]
            pred_betas[update] = new_opt_betas[update]
            pred_joints3d[update] = new_opt_joints3d[update]

        elif args.run_smplify and args.tracking_method == 'bbox':
            print('[WARNING] You need to enable pose tracking to run Temporal SMPLify algorithm!')
            print('[WARNING] Continuing without running Temporal SMPLify!..')

        # ========= Save results to a pickle file ========= #
        pred_cam = pred_cam.cpu().numpy()
        pred_verts = pred_verts.cpu().numpy()
        pred_pose = pred_pose.cpu().numpy()
        pred_betas = pred_betas.cpu().numpy()
        pred_joints3d = pred_joints3d.cpu().numpy()
        smpl_joints2d = smpl_joints2d.cpu().numpy()

        # Runs 1 Euro Filter to smooth out the results
        if args.smooth:
            min_cutoff = args.smooth_min_cutoff # 0.004
            beta = args.smooth_beta # 1.5
            print(f'Running smoothing on person {person_id}, min_cutoff: {min_cutoff}, beta: {beta}')
            pred_verts, pred_pose, pred_joints3d = smooth_pose(pred_pose, pred_betas,
                                                               min_cutoff=min_cutoff, beta=beta)

        bboxes[seq_len-1:, 2:] = bboxes[seq_len-1:, 2:] * bbox_scale
        if args.render_plain:
            pred_cam[:,0], pred_cam[:,1:] = 1, 0  # np.array([[1, 0, 0]])


        orig_cam = convert_crop_cam_to_orig_img(
            cam=pred_cam,
            bbox=bboxes,
            img_width=orig_width,
            img_height=orig_height
        )

        joints2d_img_coord = convert_crop_coords_to_orig_img(
            bbox=bboxes,
            keypoints=smpl_joints2d,
            crop_size=224,
        )

        output_dict = {
            'pred_cam': pred_cam,
            'orig_cam': orig_cam,
            'verts': pred_verts,
            'pose': pred_pose,
            'betas': pred_betas,
            'joints3d': pred_joints3d,
            'joints2d': joints2d,
            'joints2d_img_coord': joints2d_img_coord,
            'bboxes': bboxes,
            'frame_ids': frames,
        }

        tepose_results[person_id] = output_dict

    del model

    end = time.time()
    fps = num_frames / (end - tepose_time)
    print(f'TePose FPS: {fps:.2f}')
    total_time = time.time() - total_time
    print(f'Total time spent: {total_time:.2f} seconds (including model loading time).')
    print(f'Total FPS (including model loading time): {num_frames / total_time:.2f}.')

    if args.save_pkl:
        print(f"Saving output results to \'{os.path.join(output_path, 'tepose_output.pkl')}\'.")
        joblib.dump(tepose_results, os.path.join(output_path, "tepose_output.pkl"))

    """ Render results as a single video """
    renderer = Renderer(resolution=(orig_width, orig_height), orig_img=True, wireframe=args.wireframe)

    output_img_folder = f'{image_folder}_output'
    input_img_folder = f'{image_folder}_input'
    os.makedirs(output_img_folder, exist_ok=True)
    os.makedirs(input_img_folder, exist_ok=True)

    print(f"\nRendering output video, writing frames to {output_img_folder}")
    # prepare results for rendering
    frame_results = prepare_rendering_results(tepose_results, num_frames)
    mesh_color = {k: colorsys.hsv_to_rgb(np.random.rand(), 0.5, 1.0) for k in tepose_results.keys()}

    image_file_names = sorted([
        os.path.join(image_folder, x)
        for x in os.listdir(image_folder)
        if x.endswith('.png') or x.endswith('.jpg')
    ])

    for frame_idx in tqdm(range(len(image_file_names))):
        img_fname = image_file_names[frame_idx]
        img = cv2.imread(img_fname)
        input_img = img.copy()
        if args.render_plain:
            img[:] = 0

        if args.sideview:
            side_img = np.zeros_like(img)

        for person_id, person_data in frame_results[frame_idx].items():
            frame_verts = person_data['verts']
            frame_cam = person_data['cam']

            mesh_filename = None
            if args.save_obj:
                mesh_folder = os.path.join(output_path, 'meshes', f'{person_id:04d}')
                Path(mesh_folder).mkdir(parents=True, exist_ok=True)
                mesh_filename = os.path.join(mesh_folder, f'{frame_idx:06d}.obj')

            mc = mesh_color[person_id]

            img = renderer.render(
                img,
                frame_verts,
                cam=frame_cam,
                color=mc,
                mesh_filename=mesh_filename,
            )
            if args.sideview:
                side_img = renderer.render(
                    side_img,
                    frame_verts,
                    cam=frame_cam,
                    color=mc,
                    angle=270,
                    axis=[0,1,0],
                )

        if args.sideview:
            img = np.concatenate([img, side_img], axis=1)

        # save output frames
        cv2.imwrite(os.path.join(output_img_folder, f'{frame_idx:06d}.jpg'), img)
        cv2.imwrite(os.path.join(input_img_folder, f'{frame_idx:06d}.jpg'), input_img)

        if args.display:
            cv2.imshow('Video', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    if args.display:
        cv2.destroyAllWindows()

    """ Save rendered video """
    vid_name = os.path.basename(video_file)
    save_name = f'{vid_name.replace(".mp4", "")}_TePose_output2.mp4'
    save_path = os.path.join(output_path, save_name)

    images_to_video(img_folder=output_img_folder, output_vid_file=save_path)
    images_to_video(img_folder=input_img_folder, output_vid_file=os.path.join(output_path, vid_name))
    print(f"Saving result video to {os.path.abspath(save_path)}")
    shutil.rmtree(output_img_folder)
    shutil.rmtree(input_img_folder)
    shutil.rmtree(image_folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--vid_file', type=str, default='data/chouzhen/squatdown3.mp4', help='input video path or youtube link')

    parser.add_argument('--tracking_method', type=str, default='bbox', choices=['bbox', 'pose'],
                        help='tracking method to calculate the tracklet of a subject from the input video')

    parser.add_argument('--model', type=str, default='model/tepose_wpw_3dpw_test.pth.tar', help='path to pretrained model weight')

    parser.add_argument('--detector', type=str, default='yolo', choices=['yolo', 'maskrcnn'],
                        help='object detector to be used for bbox tracking')

    parser.add_argument('--yolo_img_size', type=int, default=416,
                        help='input image size for yolo detector')

    parser.add_argument('--tracker_batch_size', type=int, default=12,
                        help='batch size of object detector used for bbox tracking')

    parser.add_argument('--display', action='store_true',
                        help='visualize the results of each step during demo')

    parser.add_argument('--save_pkl', action='store_true',
                        help='save results to a pkl file')

    parser.add_argument('--save_obj', action='store_true',
                        help='save results as .obj files.')

    parser.add_argument('--run_smplify', action='store_true',
                        help='run smplify for refining the results, you need pose tracking to enable it')

    parser.add_argument('--gender', type=str, default='male',
                        help='set gender of people from (neutral, male, female)')

    parser.add_argument('--wireframe', action='store_true',
                        help='render all meshes as wireframes.')

    parser.add_argument('--sideview', action='store_true',
                        help='render meshes from alternate viewpoint.')

    parser.add_argument('--render_plain', action='store_true',
                        help='render meshes on plain background')

    parser.add_argument('--smooth', action='store_true',
                        help='smooth the results to prevent jitter')

    parser.add_argument('--smooth_min_cutoff', type=float, default=0.004,
                        help='one euro filter min cutoff. '
                             'Decreasing the minimum cutoff frequency decreases slow speed jitter')

    parser.add_argument('--smooth_beta', type=float, default=0.7,
                        help='one euro filter beta. '
                             'Increasing the speed coefficient(beta) decreases speed lag.')

    parser.add_argument('--gpu', type=int, default='1', help='gpu num')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    main(args)
