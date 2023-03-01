import numpy as np

from common.generator import ChunkedGenerator


class GeneratorH36M(ChunkedGenerator):
    def __init__(self, batch_size, cameras, poses_3d, poses_2d_keypoints, poses_scores, poses_vis, poses_2d_gt,
                 poses_dist, chunk_length=1, pad=0, causal_shift=0,
                 shuffle=False, random_seed=1234, augment=False, reverse_aug=False, kps_left=None, kps_right=None,
                 joints_left=None, joints_right=None, endless=False, out_all=False):
        assert poses_dist is None or len(poses_dist) == len(poses_2d_keypoints)
        super().__init__(batch_size, cameras, poses_3d, poses_2d_keypoints, poses_scores, poses_vis, poses_2d_gt,
                         chunk_length, pad, causal_shift,
                         shuffle, random_seed, augment, reverse_aug, kps_left, kps_right, joints_left, joints_right,
                         endless, out_all, dataset='h36m')

        if poses_dist is not None:
            example_key = next(iter(poses_dist))
            self.batch_dist = np.empty(
                (batch_size, chunk_length + 2 * pad, poses_dist[example_key].shape[-2], poses_dist[example_key].shape[-1]))
        self.poses_dist = poses_dist

    def get_batch(self, seq_i, start_3d, end_3d, flip, reverse):
        subject, action, cam_index = seq_i
        seq_name = (subject, action, int(cam_index))
        extra = action, subject, int(cam_index)

        super().augment_batch(seq_name, start_3d, end_3d, flip, reverse)

        if self.poses_dist is not None:
            self.batch_dist = self.augment_seq(self.batch_dist, self.poses_dist, seq_name, start_3d, end_3d, flip, reverse,
                                               self.kps_left, self.kps_right)

        if self.poses_scores is None:
            return self.batch_cam, self.batch_3d.copy(), self.batch_2d_gt.copy(), self.batch_vis.copy(), self.batch_2d.copy(), None, self.batch_dist.copy(), extra
        else:
            return self.batch_cam, self.batch_3d.copy(), self.batch_2d_gt.copy(), self.batch_vis.copy(), self.batch_2d.copy(), self.batch_scores.copy(), self.batch_dist.copy(), extra

