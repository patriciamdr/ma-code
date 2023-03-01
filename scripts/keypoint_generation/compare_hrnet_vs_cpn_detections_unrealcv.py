import os

import numpy as np
import torch
import sys

sys.path.append(os.getcwd())

from common.unrealcv_dataset import UnrealCvDataset
from common.utils import AccumLoss

dataset_dir = '/home/patricia/dev/StridedTransformer-Pose3D/dataset/unrealcv'

total_loss_cpn = AccumLoss()
total_loss_hrnet = AccumLoss()

print('==> Loading train dataset...')
train_dataset = UnrealCvDataset(dataset_dir, train=True)

cpn_file = 'train_keypoints_scores_cpn'
cpn_keypoints_data = np.load(os.path.join(train_dataset.path, cpn_file + '.npz'), allow_pickle=True)
cpn_kps_coordinates = cpn_keypoints_data['train_keypoints'].item()
total_train_loss_cpn = AccumLoss()

hrnet_file = 'train_keypoints_scores_hrnet'
hrnet_keypoints_data = np.load(os.path.join(train_dataset.path, hrnet_file + '.npz'), allow_pickle=True)
hrnet_kps_coordinates = hrnet_keypoints_data['train_keypoints'].item()
total_train_loss_hrnet = AccumLoss()

for db_rec in train_dataset.db:
    cpn_kps = torch.from_numpy(cpn_kps_coordinates[db_rec['image']])
    cpn_kps = torch.cat([cpn_kps[:9], cpn_kps[11:]])

    hrnet_kps = torch.from_numpy(hrnet_kps_coordinates[db_rec['image']])
    hrnet_kps = torch.cat([hrnet_kps[:9], hrnet_kps[11:]])

    joints_2d = torch.from_numpy(db_rec['joints_2d'])

    loss_cpn = torch.nn.MSELoss(reduction='mean')(cpn_kps, joints_2d)
    loss_hrnet = torch.nn.MSELoss(reduction='mean')(hrnet_kps, joints_2d)

    total_train_loss_cpn.update(loss_cpn, 1)
    total_loss_cpn.update(loss_cpn, 1)
    total_train_loss_hrnet.update(loss_hrnet, 1)
    total_loss_hrnet.update(loss_hrnet, 1)

print('Average loss CPN train detections: ' + str(total_train_loss_cpn.avg))
print('Average loss HRNet train detections: ' + str(total_train_loss_hrnet.avg))

#########################

print('==> Loading test dataset...')
test_dataset = UnrealCvDataset(dataset_dir)

cpn_file = 'test_keypoints_scores_cpn'
cpn_keypoints_data = np.load(os.path.join(test_dataset.path, cpn_file + '.npz'), allow_pickle=True)
cpn_kps_coordinates = cpn_keypoints_data['test_keypoints'].item()
total_test_loss_cpn = AccumLoss()

hrnet_file = 'test_keypoints_scores_hrnet'
hrnet_keypoints_data = np.load(os.path.join(test_dataset.path, hrnet_file + '.npz'), allow_pickle=True)
hrnet_kps_coordinates = hrnet_keypoints_data['test_keypoints'].item()
total_test_loss_hrnet = AccumLoss()

for db_rec in test_dataset.db:
    cpn_kps = torch.from_numpy(cpn_kps_coordinates[db_rec['image']])
    cpn_kps = torch.cat([cpn_kps[:9], cpn_kps[11:]])

    hrnet_kps = torch.from_numpy(hrnet_kps_coordinates[db_rec['image']])
    hrnet_kps = torch.cat([hrnet_kps[:9], hrnet_kps[11:]])

    joints_2d = torch.from_numpy(db_rec['joints_2d'])

    loss_cpn = torch.nn.MSELoss(reduction='mean')(cpn_kps, joints_2d)
    loss_hrnet = torch.nn.MSELoss(reduction='mean')(hrnet_kps, joints_2d)

    total_test_loss_cpn.update(loss_cpn, 1)
    total_loss_cpn.update(loss_cpn, 1)
    total_test_loss_hrnet.update(loss_hrnet, 1)
    total_loss_hrnet.update(loss_hrnet, 1)

print('Average loss CPN test detections: ' + str(total_test_loss_cpn.avg))
print('Average loss HRNet test detections: ' + str(total_test_loss_hrnet.avg))

print('Average loss CPN detections: ' + str(total_loss_cpn.avg))
print('Average loss HRNet detections: ' + str(total_loss_hrnet.avg))
