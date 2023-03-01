import os
import cv2
import numpy as np
import sys

sys.path.append(os.getcwd())

from demo.lib.preprocess import h36m_coco_format
from demo.vis import show2Dpose

data_dir = '/home/patricia/dev/StridedTransformer-Pose3D/dataset/unrealcv/'
file = 'train_keypoints_scores_cpn_raw.npz'
keypoints_data = np.load(data_dir + file, allow_pickle=True)

kps_coordinates = keypoints_data['train_keypoints'].item()
kps_scores = keypoints_data['train_scores'].item()

train_keypoints = dict()
train_scores = dict()

for i, file in enumerate(kps_coordinates.keys()):
    assert file in kps_scores.keys()

    kps_coco = kps_coordinates[file]
    assert len(kps_coco.shape) == 2
    kps_coco = kps_coco.reshape((1, 1, kps_coco.shape[0], kps_coco.shape[1]))

    scores_coco = kps_scores[file]
    assert len(scores_coco.shape) == 1
    scores_coco = scores_coco.reshape((1, 1, scores_coco.shape[0]))

    kps_h36m, scores_h36m, valid_frames = h36m_coco_format(kps_coco, scores_coco)

    train_keypoints[file] = kps_h36m.squeeze()
    train_scores[file] = scores_h36m.squeeze()

output_dir = '/home/patricia/dev/StridedTransformer-Pose3D/dataset/unrealcv/train_keypoints_scores_cpn_converted.npz'
np.savez_compressed(output_dir, train_keypoints=train_keypoints, train_scores=train_scores)

#################

file = 'test_keypoints_scores_cpn_raw.npz'
keypoints_data = np.load(data_dir + file, allow_pickle=True)

kps_coordinates = keypoints_data['test_keypoints'].item()
kps_scores = keypoints_data['test_scores'].item()

test_keypoints = dict()
test_scores = dict()

for file in kps_coordinates.keys():
    assert file in kps_scores.keys()

    kps_coco = kps_coordinates[file]
    assert len(kps_coco.shape) == 2
    kps_coco = kps_coco.reshape((1, 1, kps_coco.shape[0], kps_coco.shape[1]))

    scores_coco = kps_scores[file]
    assert len(scores_coco.shape) == 1
    scores_coco = scores_coco.reshape((1, 1, scores_coco.shape[0]))

    kps_h36m, scores_h36m, valid_frames = h36m_coco_format(kps_coco, scores_coco)

    test_keypoints[file] = kps_h36m.squeeze()
    test_scores[file] = scores_h36m.squeeze()

output_dir = '/home/patricia/dev/StridedTransformer-Pose3D/dataset/unrealcv/test_keypoints_scores_cpn_converted.npz'
np.savez_compressed(output_dir, test_keypoints=test_keypoints, test_scores=test_scores)

