import torch
import torch.nn.functional as F
import torch.utils.data as data
import numpy as np
import os

from common.utils import deterministic_random, get_torso_size, calc_dists, get_head_size, get_normalized_distance
from common.camera import world_to_camera, normalize_screen_coordinates, wrap, project_to_2d, image_coordinates
from common.generator_h36m import GeneratorH36M


class Fusion(data.Dataset):
    def __init__(self, opt, dataset, root_path, train=True, keypoints='CPN'):
        self.opt = opt
        self.data_type = opt.dataset
        self.train = train
        self.root_path = root_path
        self.keypoints = keypoints

        self.train_list = opt.subjects_train.split(',')
        self.test_list = opt.subjects_test.split(',')
        self.action_filter = None if opt.actions == '*' else opt.actions.split(',')
        self.downsample = opt.downsample
        self.subset = opt.subset
        self.stride = opt.stride
        self.crop_uv = opt.crop_uv
        self.test_aug = opt.test_augmentation
        self.pad = opt.pad

        # For occlusion simulation
        self.local_random_gen = torch.Generator()

        if self.train:
            self.keypoints, self.scores = self.prepare_data(dataset, self.train_list)
            self.cameras_train, self.poses_train_3d, self.poses_train_2d_keypoints, self.poses_train_scores, \
            self.poses_train_2d_gt, self.poses_train_vis, self.poses_train_dist = \
                self.fetch(dataset, self.train_list, subset=self.subset)
            self.generator = GeneratorH36M(opt.batch_size // opt.stride, self.cameras_train, self.poses_train_3d, self.poses_train_2d_keypoints,
                                           self.poses_train_scores, self.poses_train_vis, self.poses_train_2d_gt,
                                           self.poses_train_dist, self.stride,  pad=self.pad,
                                           augment=opt.data_augmentation, reverse_aug=opt.reverse_augmentation,
                                           kps_left=self.kps_left, kps_right=self.kps_right,
                                           joints_left=self.joints_left,
                                           joints_right=self.joints_right, out_all=opt.train_out_all)
            print('INFO: Training on {} frames'.format(self.generator.num_frames()))
        else:
            self.keypoints, self.scores = self.prepare_data(dataset, self.test_list)
            self.cameras_test, self.poses_test, self.poses_test_2d_keypoints, self.poses_test_scores, \
            self.poses_test_2d_gt, self.poses_test_vis, self.poses_test_dist = \
                self.fetch(dataset, self.test_list, subset=self.subset)
            self.generator = GeneratorH36M(opt.batch_size // opt.stride, self.cameras_test, self.poses_test, self.poses_test_2d_keypoints,
                                           self.poses_test_scores, self.poses_test_vis, self.poses_test_2d_gt,
                                           self.poses_test_dist, pad=self.pad, augment=False,
                                           kps_left=self.kps_left, kps_right=self.kps_right, joints_left=self.joints_left,
                                           joints_right=self.joints_right, out_all=opt.test_out_all)
            self.key_index = self.generator.saved_index
            print('INFO: Testing on {} frames'.format(self.generator.num_frames()))

    def prepare_data(self, dataset, folder_list):
        for subject in folder_list:
            for action in dataset[subject].keys():
                anim = dataset[subject][action]

                positions_3d = []
                for cam in anim['cameras']:
                    pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])
                    pos_3d[:, 1:] -= pos_3d[:, :1]
                    positions_3d.append(pos_3d)
                anim['positions_3d'] = positions_3d

        # keypoints = np.load(os.path.join(self.root_path, 'data_2d_h36m_detectron_ft_h36m.npz'), allow_pickle=True)
        keypoints = np.load(os.path.join(self.root_path, 'data_2d_h36m_cpn_ft_h36m_dbb.npz'), allow_pickle=True)
        keypoints_symmetry = keypoints['metadata'].item()['keypoints_symmetry']

        self.kps_left, self.kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
        self.joints_left, self.joints_right = list(dataset.skeleton().joints_left()), list(dataset.skeleton().joints_right())

        if self.keypoints == 'CPN':
            self.use_hrnet_detections = False
            keypoints = keypoints['positions_2d'].item()

            for subject in folder_list:
                assert subject in keypoints, 'Subject {} is missing from the 2D detections dataset'.format(subject)
                for action in dataset[subject].keys():
                    assert action in keypoints[
                        subject], 'Action {} of subject {} is missing from the 2D detections dataset'.format(action,
                                                                                                             subject)
                    for cam_idx in range(len(keypoints[subject][action])):

                        mocap_length = dataset[subject][action]['positions_3d'][cam_idx].shape[0]
                        assert keypoints[subject][action][cam_idx].shape[0] >= mocap_length

                        if keypoints[subject][action][cam_idx].shape[0] > mocap_length:
                            keypoints[subject][action][cam_idx] = keypoints[subject][action][cam_idx][:mocap_length]

            scores = None
        elif self.keypoints == 'HRNet':
            self.use_hrnet_detections = True
            keypoints_data = np.load(os.path.join(self.root_path, 'keypoints_scores_hrnet.npz'), allow_pickle=True)
            scores = keypoints_data['scores'].item()
            keypoints = keypoints_data['keypoints'].item()

            for subject in folder_list:
                assert subject in keypoints, 'Subject {} is missing from the 2D detections dataset'.format(subject)
                assert subject in scores, 'Subject {} is missing from the 2D scores dataset'.format(subject)
                for action in dataset[subject].keys():
                    assert action in keypoints[
                        subject], 'Action {} of subject {} is missing from the 2D detections dataset'.format(action,
                                                                                                             subject)
                    assert action in scores[
                        subject], 'Action {} of subject {} is missing from the 2D scores dataset'.format(action,
                                                                                                         subject)
                    for cam_idx in range(len(keypoints[subject][action])):
                        assert cam_idx in range(len(scores[subject][action])), \
                            'Cam {} for action {} in subject {} is missing from the 2D scores dataset'.format(cam_idx,
                                                                                                              action,
                                                                                                              subject)
                        mocap_length = dataset[subject][action]['positions_3d'][cam_idx].shape[0]

                        assert keypoints[subject][action][cam_idx].shape[0] >= mocap_length
                        if keypoints[subject][action][cam_idx].shape[0] > mocap_length:
                            keypoints[subject][action][cam_idx] = keypoints[subject][action][cam_idx][:mocap_length]

                        assert scores[subject][action][cam_idx].shape[0] >= mocap_length
                        if scores[subject][action][cam_idx].shape[0] > mocap_length:
                            scores[subject][action][cam_idx] = scores[subject][action][cam_idx][:mocap_length]
        else:
            raise KeyError('Invalid dataset')

        for subject in folder_list:
            assert subject in keypoints.keys()
            for action in dataset[subject].keys():
                assert action in keypoints[subject].keys()
                anim = dataset[subject][action]

                positions_2d = []
                vis = []
                dist = []
                for cam_idx, cam in enumerate(anim['cameras']):
                    assert dataset.cameras()[subject][cam_idx] == cam
                    pos_2d_pred = keypoints[subject][action][cam_idx]

                    pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])
                    pos_2d = wrap(project_to_2d, pos_3d, cam['intrinsic'], unsqueeze=True)
                    pos_2d_pixel_space = image_coordinates(pos_2d, w=cam['res_w'], h=cam['res_h'])

                    # Create artificial visibility target
                    # norm = get_torso_size(pos_2d_pixel_space)
                    # dists = calc_dists(pos_2d_pred, pos_2d_pixel_space, norm)

                    head_size = get_head_size(pos_2d_pixel_space)
                    normalized_distance = get_normalized_distance(pos_2d_pixel_space, pos_2d_pred, head_size)

                    pos_vis = np.zeros((pos_2d_pred.shape[0], pos_2d_pred.shape[1], 1))
                    for i in range(pos_2d_pred.shape[1]):
                        pos_vis[:, i] = np.less(normalized_distance[:, i], self.opt.error_thr)[..., None]
                        # pos_vis[:, i] = np.less(dists[i], self.opt.error_thr)[..., None]
                    vis.append(pos_vis)
                    dist.append(normalized_distance[..., None])

                    if self.crop_uv == 0:
                        pos_2d_pixel_space[..., :2] = normalize_screen_coordinates(pos_2d_pixel_space[..., :2], w=cam['res_w'], h=cam['res_h'])
                        pos_2d_pred[..., :2] = normalize_screen_coordinates(pos_2d_pred[..., :2], w=cam['res_w'], h=cam['res_h'])
                    positions_2d.append(pos_2d_pixel_space.astype('float32'))
                    keypoints[subject][action][cam_idx] = pos_2d_pred

                anim['positions_2d'] = positions_2d
                anim['vis'] = vis
                anim['dist'] = dist
        return keypoints, scores

    def fetch(self, dataset, subjects, subset=1, parse_3d_poses=True):
        out_poses_3d = {}
        out_poses_2d_gt = {}
        out_poses_2d = {}
        out_poses_vis = {}
        out_poses_dist = {}
        out_camera_params = {}
        if self.use_hrnet_detections:
            out_poses_scores = {}
        else:
            out_poses_scores = None

        for subject in subjects:
            for action in self.keypoints[subject].keys():
                if out_poses_scores is not None:
                    assert action in self.scores[subject].keys(), 'Missing action in scores'
                if self.action_filter is not None:
                    found = False
                    for a in self.action_filter:
                        if action.startswith(a):
                            found = True
                            break
                    if not found:
                        continue

                poses_2d = self.keypoints[subject][action]

                if out_poses_scores is not None:
                    poses_scores = self.scores[subject][action]
                    assert len(poses_2d) == len(poses_scores), 'Scores and keypoints mismatch'

                for i in range(len(poses_2d)):
                    out_poses_2d[(subject, action, i)] = poses_2d[i]
                    if out_poses_scores is not None:
                        out_poses_scores[(subject, action, i)] = poses_scores[i][..., None]

                if subject in dataset.cameras():
                    cams = dataset.cameras()[subject]
                    assert len(cams) == len(poses_2d), 'Camera count mismatch'
                    if out_poses_scores is not None:
                        assert len(cams) == len(poses_scores), 'Camera count mismatch'
                    for i, cam in enumerate(cams):
                        if 'intrinsic' in cam:
                            out_camera_params[(subject, action, i)] = cam['intrinsic']

                if parse_3d_poses and 'positions_3d' in dataset[subject][action]:
                    poses_3d = dataset[subject][action]['positions_3d']
                    assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
                    for i in range(len(poses_3d)):
                        assert len(poses_3d[i]) == len(self.keypoints[subject][action][i])
                        out_poses_3d[(subject, action, i)] = poses_3d[i]

                if 'positions_2d' in dataset[subject][action]:
                    poses_2d_gt = dataset[subject][action]['positions_2d']
                    assert len(poses_2d_gt) == len(poses_2d), 'Camera count mismatch'
                    for i in range(len(poses_2d_gt)):
                        assert len(poses_2d_gt[i]) == len(self.keypoints[subject][action][i])
                        out_poses_2d_gt[(subject, action, i)] = poses_2d_gt[i]

                if 'vis' in dataset[subject][action]:
                    poses_vis = dataset[subject][action]['vis']
                    assert len(poses_vis) == len(poses_2d), 'Camera count mismatch'
                    for i in range(len(poses_vis)):
                        assert len(poses_vis[i]) == len(self.keypoints[subject][action][i])
                        out_poses_vis[(subject, action, i)] = poses_vis[i]

                if 'dist' in dataset[subject][action]:
                    poses_dist = dataset[subject][action]['dist']
                    assert len(poses_dist) == len(poses_2d), 'Camera count mismatch'
                    for i in range(len(poses_dist)):
                        assert len(poses_dist[i]) == len(self.keypoints[subject][action][i])
                        out_poses_dist[(subject, action, i)] = poses_dist[i]

        if len(out_camera_params) == 0:
            out_camera_params = None
        if len(out_poses_3d) == 0:
            out_poses_3d = None
        if len(out_poses_2d_gt) == 0:
            out_poses_2d_gt = None
        if len(out_poses_vis) == 0:
            out_poses_vis = None

        stride = self.downsample
        if subset < 1:
            for key in out_poses_2d.keys():
                n_frames = int(round(len(out_poses_2d[key]) // stride * subset) * stride)
                start = deterministic_random(0, len(out_poses_2d[key]) - n_frames + 1, str(len(out_poses_2d[key])))
                out_poses_2d[key] = out_poses_2d[key][start:start + n_frames:stride]
                if out_poses_scores is not None:
                    out_poses_scores[key] = out_poses_scores[key][start:start + n_frames:stride]
                if out_poses_3d is not None:
                    out_poses_3d[key] = out_poses_3d[key][start:start + n_frames:stride]
                if out_poses_2d_gt is not None:
                    out_poses_2d_gt[key] = out_poses_2d_gt[key][start:start + n_frames:stride]
                if out_poses_vis is not None:
                    out_poses_vis[key] = out_poses_vis[key][start:start + n_frames:stride]
                if out_poses_dist is not None:
                    out_poses_dist[key] = out_poses_dist[key][start:start + n_frames:stride]
        elif stride > 1:
            for key in out_poses_2d.keys():
                out_poses_2d[key] = out_poses_2d[key][::stride]
                if out_poses_scores is not None:
                    out_poses_scores[key] = out_poses_scores[key][::stride]
                if out_poses_3d is not None:
                    out_poses_3d[key] = out_poses_3d[key][::stride]
                if out_poses_2d_gt is not None:
                    out_poses_2d_gt[key] = out_poses_2d_gt[key][::stride]
                if out_poses_vis is not None:
                    out_poses_vis[key] = out_poses_vis[key][::stride]
                if out_poses_dist is not None:
                    out_poses_dist[key] = out_poses_dist[key][::stride]

        return out_camera_params, out_poses_3d, out_poses_2d, out_poses_scores, out_poses_2d_gt, out_poses_vis, out_poses_dist

    def __len__(self):
        return len(self.generator.pairs)

    def __getitem__(self, index):
        seq_name, start_3d, end_3d, flip, reverse = self.generator.pairs[index]

        cam, gt_3D, gt_2D, vis, input_2D, input_scores, dist, extra = self.generator.get_batch(seq_name, start_3d, end_3d, flip, reverse)

        orig_input_2D = input_2D.copy()
        if input_scores is not None:
            orig_input_scores = input_scores.copy()

        if self.opt.occlusion_augmentation_train or self.opt.occlusion_augmentation_test:
            mask = self.generate_occlusion_mask()
            mask_flip = mask.clone()
            mask_flip[:, self.joints_left + self.joints_right] = mask_flip[:, self.joints_right + self.joints_left]

        if self.opt.occlusion_augmentation_train:
            input_2D[mask.unsqueeze(2).repeat(1, 1, 2)] = 0
            if input_scores is not None:
                input_scores[mask] = 0

        if not self.train and self.test_aug:
            _, _, gt_2D_aug, _, input_2D_aug, input_scores_aug, _, _ = self.generator.get_batch(seq_name, start_3d, end_3d, flip=True, reverse=reverse)

            if self.opt.occlusion_augmentation_test:
                input_2D_aug[mask_flip.unsqueeze(2).repeat(1, 1, 2)] = 0
                if input_scores is not None:
                    input_scores_aug[mask_flip] = 0
            else:
                input_2D = orig_input_2D
                if input_scores is not None:
                    input_scores = orig_input_scores

            input_2D = np.concatenate((np.expand_dims(input_2D,axis=0),np.expand_dims(input_2D_aug,axis=0)),0)
            # gt_2D = np.concatenate((np.expand_dims(gt_2D,axis=0),np.expand_dims(gt_2D_aug,axis=0)),0)
            if input_scores is not None:
                input_scores = np.concatenate((np.expand_dims(input_scores,axis=0),np.expand_dims(input_scores_aug,axis=0)),0)

        bb_box = np.array([0, 0, 1, 1])
        input_2D_update = input_2D
        gt_2D_update = gt_2D
        input_scores_update = input_scores

        scale = np.float(1.0)

        if input_scores_update is not None:
            return cam, gt_3D, gt_2D_update, vis, input_2D_update, input_scores_update, dist, scale, bb_box, extra
        else:
            return cam, gt_3D, gt_2D_update, vis, input_2D_update, torch.empty(vis.shape), dist, scale, bb_box, extra

    def reset_seed(self, seed):
        self.local_random_gen.manual_seed(seed)

    def generate_non_consecutive_occlusion_mask(self):
        # Select frames to occlude
        if self.opt.num_occluded_f is None:
            # Occlude a random number of frames per sequence (between 0 and self.opt.frames)
            num_occluded_f = torch.randint(low=0, high=self.opt.frames // 2, size=[1], generator=self.local_random_gen).squeeze()

            # Generate a variable number of elements in each row (i.e. per sequence)
            frames_idx = F.one_hot(num_occluded_f, self.opt.frames)
            frames_idx = 1 - torch.cumsum(frames_idx, dim=0)

            # Shuffle each row
            indices = torch.argsort(torch.randn(self.opt.frames), dim=0)
            mask_f = frames_idx[indices]
        elif 0 < self.opt.num_occluded_f <= self.opt.frames:
            # For each sequence occlude specific number of frames, each frame with equal probability
            prob_f = torch.ones(self.opt.frames)
            frames_idx = torch.multinomial(prob_f, self.opt.num_occluded_f, generator=self.local_random_gen)
            mask_f = torch.zeros(self.opt.frames).scatter(0, frames_idx, 1)
        else:
            raise KeyError('Invalid number of occluded frames')

        joints_idx = []
        # Select joints to occlude
        if self.opt.occluded_joint is None:
            if self.opt.num_occluded_j <= 0 or self.opt.out_joints < self.opt.num_occluded_j:
                raise KeyError('Invalid number of occluded joints')
            else:
                for _ in range(self.opt.num_occluded_j):
                    # For each frame select a random joint
                    joints_idx.append(torch.multinomial(self.opt.joint_probability, self.opt.frames,
                                                        replacement=True, generator=self.local_random_gen))
        elif 0 <= self.opt.occluded_joint < self.opt.out_joints:
            # Occlude the same joint in each frame
            joints_idx.append(torch.full((self.opt.frames,), self.opt.occluded_joint))
        else:
            raise KeyError('Invalid index of occluded joint')

        return mask_f, joints_idx

    def generate_consecutive_occlusion_mask(self):
        # Select frames to occlude

        # Occlude a specific number of subsets with consecutive frames for each sequence
        if self.opt.num_occluded_f is None:
            num_occluded_f = torch.randint(low=self.opt.subset_size, high=self.opt.frames // 2, size=[1], generator=self.local_random_gen).squeeze()
        elif self.opt.num_occluded_f <= 0 or self.opt.frames < self.opt.num_occluded_f:
            raise KeyError('Invalid number of occluded frames')
        else:
            num_occluded_f = self.opt.num_occluded_f

        if 0 < self.opt.subset_size <= self.opt.frames:
            # Occlude a random number of frames per sequence (between self.opt.subset_size and self.opt.frames)
            num_subsets = torch.div(num_occluded_f, self.opt.subset_size, rounding_mode='floor')
        else:
            raise KeyError('Invalid subset size')

        mod = self.opt.subset_size % 2
        pad = self.opt.subset_size // 2

        # Sample middle frames of subsets to occlude with equal probability
        prob_f = torch.ones(self.opt.frames)
        prob_f[:pad] = 0
        prob_f[self.opt.frames - pad - mod + 1:] = 0
        rand_middle_f = torch.multinomial(prob_f, num_subsets, generator=self.local_random_gen)

        # Create subsets to occlude from middle frame ids
        indices = torch.argsort(torch.zeros(self.opt.subset_size)).repeat(num_subsets)
        frames_idx = indices + rand_middle_f.repeat_interleave(self.opt.subset_size, dim=0) - pad

        mask_f = torch.zeros(self.opt.frames).scatter(0, frames_idx, 1)

        # Select joints to occlude
        joints_idx = []
        if self.opt.occluded_joint is None:
            if self.opt.num_occluded_j <= 0 or self.opt.out_joints < self.opt.num_occluded_j:
                raise KeyError('Invalid number of occluded joints')
            else:
                for _ in range(self.opt.num_occluded_j):
                    # For each subset select a num_joints random joints
                    rand_middle_j = torch.multinomial(self.opt.joint_probability, num_subsets,
                                                      replacement=True, generator=self.local_random_gen)
                    indices = rand_middle_j.repeat_interleave(self.opt.subset_size, dim=0)
                    joints_idx.append(torch.zeros(self.opt.frames, dtype=indices.dtype).scatter(0, frames_idx, indices))
        elif 0 <= self.opt.occluded_joint < self.opt.out_joints:
            # Occlude the same joint in each frame
            joints_idx.append(torch.full((self.opt.frames,), self.opt.occluded_joint))
        else:
            raise KeyError('Invalid index of occluded joint')

        return mask_f, joints_idx

    def generate_occlusion_mask(self):
        if not self.opt.consecutive_frames:
            mask_f, joints_idx = self.generate_non_consecutive_occlusion_mask()
        else:
            mask_f, joints_idx = self.generate_consecutive_occlusion_mask()
        # Generate frame mask from indices - occludes all joints within affected frames
        mask_f = mask_f.unsqueeze(1).repeat(1, 17)

        # Generate joint mask from indices - occludes affected joints in all frames
        mask_j = F.one_hot(joints_idx[0], self.opt.out_joints)
        if len(joints_idx) > 1:
            for i in range(len(joints_idx)):
                temp_mask_j = F.one_hot(joints_idx[i], self.opt.out_joints)
                mask_j = mask_j.logical_or(temp_mask_j)

        # Filter out all frames and joints that should not be occluded
        mask = mask_f.logical_and(mask_j)

        # Repeat mask for x and y coordinate
        # return mask.unsqueeze(2).repeat(1, 1, 2)
        return mask


