import numpy as np

from common.generator import ChunkedGenerator


class GeneratorUnrealCV(ChunkedGenerator):
    def __init__(self, batch_size, cameras, poses_3d, poses_2d_keypoints, poses_scores, poses_vis, poses_2d_gt, chunk_length=1, pad=0, causal_shift=0,
                 shuffle=False, random_seed=1234, augment=False, reverse_aug=False, kps_left=None, kps_right=None,
                 endless=False, out_all=False):
        super().__init__(batch_size, cameras, poses_3d, poses_2d_keypoints, poses_scores, poses_vis, poses_2d_gt,
                         chunk_length, pad, causal_shift,
                         shuffle, random_seed, augment, reverse_aug, kps_left, kps_right, kps_left, kps_right,
                         endless, out_all, dataset='unrealcv')

    def get_batch(self, seq_i, start_3d, end_3d, flip, reverse):
        video_id, cam_index, action = seq_i
        seq_name = (video_id, int(cam_index), int(action))
        extra = video_id, int(cam_index), int(action)

        super().augment_batch(seq_name, start_3d, end_3d, flip, reverse)

        self.batch_vis = self.augment_seq(self.batch_vis, self.poses_vis, seq_name, start_3d, end_3d, flip, reverse,
                                          self.kps_left, self.kps_right)

        return self.batch_cam, self.batch_3d.copy(), self.batch_2d_gt.copy(), self.batch_vis.copy(), self.batch_2d.copy(), self.batch_scores.copy(), extra
