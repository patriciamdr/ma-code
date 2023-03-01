import os
import torch.utils.data as data
import numpy as np

from common.camera import normalize_screen_coordinates
from common.generator_unrealcv import GeneratorUnrealCV
import sys
sys.path.append(os.getcwd())
from demo.lib.hrnet.lib.utils.transforms import get_affine_transform, affine_transform


class Fusion(data.Dataset):
    def __init__(self, opt, dataset, train=False, keypoint_file='CPN'):
        self.data_type = opt.dataset
        self.train = train
        self.keypoint_file = keypoint_file

        self.action_filter = None if opt.actions == '*' else opt.actions.split(',')
        self.downsample = opt.downsample
        self.subset = opt.subset
        self.stride = opt.stride
        self.crop_uv = opt.crop_uv
        self.test_aug = opt.test_augmentation
        self.pad = opt.pad

        self.n_joints = dataset.skeleton().num_joints()
        self.cropped_image_size = opt.cropped_image_size
        self.orig_image_size = dataset.orig_image_size
        self.scale_factor = opt.scale_factor
        self.rotation_factor = opt.rotation_factor
        # self.sigma = opt.heatmap_sigma
        # self.heatmap_size = opt.heatmap_res_out

        self.kps_left, self.kps_right = dataset.skeleton().joints_left(), dataset.skeleton().joints_right()

        self.idx_dct, self.poses_2d_gt, self.poses_scores, self.poses_vis, self.poses_3d, self.camera_params, self.keypoints = \
            self.fetch(dataset, train)
        if self.train:
            self.generator = GeneratorUnrealCV(opt.batch_size // opt.stride, self.camera_params, self.poses_3d, self.keypoints,
                                               self.poses_scores, self.poses_vis, self.poses_2d_gt, self.stride,
                                               pad=self.pad, augment=opt.data_augmentation, reverse_aug=opt.reverse_augmentation,
                                               kps_left=self.kps_left, kps_right=self.kps_right, out_all=opt.train_out_all)
            print('INFO: Training on {} frames'.format(self.generator.num_frames()))
        else:
            self.generator = GeneratorUnrealCV(opt.batch_size // opt.stride, self.camera_params, self.poses_3d, self.keypoints,
                                               self.poses_scores, self.poses_vis, self.poses_2d_gt, pad=self.pad,
                                               augment=False, kps_left=self.kps_left, kps_right=self.kps_right)
            print('INFO: Testing on {} frames'.format(self.generator.num_frames()))

    def fetch(self, dataset, train):
        idx_dct, poses_3d, poses_2d, poses_scores, poses_vis, camera_params, keypoints = \
            dict(), dict(), dict(), dict(), dict(), dict(), dict()

        file = '{}_keypoints_scores_{}'.format('train' if train else 'test', self.keypoint_file.lower())
        keypoints_data = np.load(os.path.join(dataset.path, file + '.npz'), allow_pickle=True)
        kps_coordinates = keypoints_data['{}_keypoints'.format('train' if train else 'test')].item()
        kps_scores = keypoints_data['{}_scores'.format('train' if train else 'test')].item()

        nitems = len(dataset)
        # Iterate over all frames and assign to a video to create sequences
        for i in range(nitems):
            db_rec = dataset[i]

            center = np.array(self.orig_image_size) / 2
            scale = np.array(self.cropped_image_size) / 200
            rotation = 0

            kps = kps_coordinates[db_rec['image']]
            scores = kps_scores[db_rec['image']][..., None]

            # Crop image to be a square
            trans = get_affine_transform(center, scale, rotation, self.cropped_image_size)

            for j in range(kps.shape[0]):
                kps[j, 0:2] = affine_transform(kps[j, 0:2], trans)

            joints_2d = db_rec['joints_2d'].copy()
            joints_vis = db_rec['joints_vis'].copy()
            for j in range(self.n_joints):
                joints_2d[j, 0:2] = affine_transform(joints_2d[j, 0:2], trans)
                # TODO: Check if negative 2D joints should be marked as occluded or not
                # if joints_vis[j] > 0.0:
                #     if np.min(joints_2d[j, :2]) < 0 or joints_2d[j, 0] >= self.cropped_image_size[0] or joints_2d[j, 1] >= self.cropped_image_size[1]:
                #         joints_vis[j] = 0

            cam_params = db_rec['camera'].copy()
            # TODO: Normalize center and focal length of camera + adapt camera intrinsics for cropping
            cam_intrinsics = np.concatenate((np.array((cam_params['fx'], cam_params['fy'],
                                                       cam_params['cx'], cam_params['cy'])),
                                             cam_params['k'].flatten(), cam_params['p'].flatten()))

            # Meters instead of millimeters
            joints_gt = db_rec['joints_gt'].copy() / 1000
            R = cam_params['R']
            T = cam_params['T'] / 1000
            joints_3d = (R.dot(joints_gt.T - T)).T  # rotate and translate
            # Zero center 3D ground truth on root
            joints_3d[1:] -= joints_3d[:1]

            video_id, image_id, cam_id, action = db_rec['video_id'], db_rec['image_id'], db_rec['camera_id'], db_rec['action']
            key = (video_id, cam_id, action)
            value = image_id

            # Normalize coordinates of 2D ground truth and keypoint
            kps = normalize_screen_coordinates(kps, w=self.cropped_image_size[0], h=self.cropped_image_size[1])
            joints_2d = normalize_screen_coordinates(joints_2d, w=self.cropped_image_size[0], h=self.cropped_image_size[1])

            if sum(joints_vis) == 0:  # Only add images where at least one joint is visible
                continue

            if key not in idx_dct:
                # Add new video
                idx_dct[key] = {value: i}
                keypoints[key] = {value: kps}
                poses_3d[key] = {value: joints_3d}
                poses_2d[key] = {value: joints_2d}
                poses_scores[key] = {value: scores}
                poses_vis[key] = {value: joints_vis}
                # poses_vis[key] = {value: np.array([i])}
                camera_params[key] = cam_intrinsics
            elif value not in idx_dct[key]:
                # Add new frame to existing video
                idx_dct[key][value] = i
                keypoints[key][value] = kps
                poses_3d[key][value] = joints_3d
                poses_2d[key][value] = joints_2d
                poses_scores[key][value] = scores
                poses_vis[key][value] = joints_vis
                # poses_vis[key][value] = np.array([i])

                # Check if cam intrinsics is coherent across frames
                assert np.array_equal(camera_params[key], cam_intrinsics)

        # for key in list(poses_vis.keys()):
        #     seq = poses_vis[key]
        #     if (np.stack(list(seq.values())) == np.zeros(dataset.num_joints)).all(axis=1).any():
        #         # Delete videos with images where no joint is visible
        #         idx_dct.pop(key, None)
        #         keypoints.pop(key, None)
        #         poses_3d.pop(key, None)
        #         poses_2d.pop(key, None)
        #         poses_scores.pop(key, None)
        #         poses_vis.pop(key, None)
        #         camera_params.pop(key, None)

        def sort_dict(d):
            return dict(sorted(d.items()))

        def get_np_from_dict_values(d):
            return np.array(list(d.values()))

        # Iterate over all videos and sort frames by id to get videos with frames in correct order
        for key in idx_dct:
            idx_dct[key] = get_np_from_dict_values(sort_dict(idx_dct[key]))
            keypoints[key] = get_np_from_dict_values(sort_dict(keypoints[key]))
            poses_3d[key] = get_np_from_dict_values(sort_dict(poses_3d[key]))
            poses_2d[key] = get_np_from_dict_values(sort_dict(poses_2d[key]))
            poses_scores[key] = get_np_from_dict_values(sort_dict(poses_scores[key]))
            poses_vis[key] = get_np_from_dict_values(sort_dict(poses_vis[key]))

        return idx_dct, poses_2d, poses_scores, poses_vis, poses_3d, camera_params, keypoints

    def __len__(self):
        return len(self.generator.pairs)

    def __getitem__(self, index):
        seq_name, start_3d, end_3d, flip, reverse = self.generator.pairs[index]

        cam, gt_3D, gt_2D, vis, keypoints, scores, extra = self.generator.get_batch(seq_name, start_3d, end_3d, flip, reverse)

        if not self.train and self.test_aug:
            _, _, _, vis_aug, keypoints_aug, scores_aug, _ = self.generator.get_batch(seq_name, start_3d, end_3d,
                                                                                      flip=True, reverse=reverse)
            keypoints = np.concatenate((np.expand_dims(keypoints, axis=0), np.expand_dims(keypoints_aug, axis=0)), 0)
            vis = np.concatenate((np.expand_dims(vis, axis=0), np.expand_dims(vis_aug, axis=0)), 0)
            scores = np.concatenate((np.expand_dims(scores, axis=0), np.expand_dims(scores_aug, axis=0)), 0)

        keypoints_update = keypoints
        vis_update = vis
        scores_update = scores

        return cam, gt_3D, gt_2D, vis_update, keypoints_update, scores_update, extra
