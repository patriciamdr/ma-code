import argparse
import os
import sys

from tqdm import tqdm

sys.path.append(os.getcwd())
from common.utils import get_variable

import torch
from common.opt import graformer_opts
from common.load_data_h36m import Fusion as Fusion_h36m
from common.h36m_dataset import Human36mDataset


if __name__ == "__main__":
    print('==> Loading dataset...')

    parser = argparse.ArgumentParser()
    opt = graformer_opts(parser).get_graformer_args()

    dataset_dir = '/home/patricia/dev/StridedTransformer-Pose3D/dataset/h36m'
    dataset_path = os.path.join(dataset_dir, 'data_3d_h36m.npz')
    dataset = Human36mDataset(dataset_path, opt)

    train_data = Fusion_h36m(opt=opt, train=True, dataset=dataset, root_path=dataset_dir)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=opt.batch_size,
                                                   shuffle=True, num_workers=int(opt.workers), pin_memory=True)

    print('Start analysis ... ')
    train_total_vis = []
    for i, data in enumerate(tqdm(train_dataloader)):
        batch_cam, gt_3d, gt_2d, vis, inputs_2d, inputs_scores, dist, scale, bb_box, extra = data
        action, subject, cam_ind = extra
        [inputs_2d, inputs_scores, gt_3d, batch_cam, vis] = get_variable('train', [inputs_2d, inputs_scores, gt_3d, batch_cam, vis])

        train_total_vis.append(vis)

    train_vis = torch.cat(train_total_vis, dim=0)

    total_train_vis = train_vis.flatten().shape
    print('Number joints - Train: ' + str(total_train_vis))

    percentage_train_vis = (train_vis == 1).float().mean()
    print('Precentage visible joints - Train: ' + str(percentage_train_vis.item() * 100))

    sum_train_vis = (train_vis == 1).float().sum()
    print('Number visible joints - Train: ' + str(sum_train_vis.item()))

    sum_train_non_vis = (train_vis == 0).float().sum()
    print('Number non-visible joints - Train: ' + str(sum_train_non_vis.item()))

    test_data = Fusion_h36m(opt=opt, train=False, dataset=dataset, root_path=dataset_dir)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=opt.batch_size,
                                                  shuffle=False, num_workers=int(opt.workers), pin_memory=True)

    test_total_vis = []
    for i, data in enumerate(tqdm(test_dataloader)):
        batch_cam, gt_3d, gt_2d, vis, inputs_2d, inputs_scores, dist, scale, bb_box, extra = data
        action, subject, cam_ind = extra
        [inputs_2d, inputs_scores, gt_3d, batch_cam, vis] = get_variable('test', [inputs_2d, inputs_scores, gt_3d, batch_cam, vis])

        test_total_vis.append(vis)

    test_vis = torch.cat(test_total_vis, dim=0)

    total_test_vis = test_vis.flatten().shape
    print('Number joints - Test: ' + str(total_test_vis))

    percentage_test_vis = (test_vis == 1).float().mean()
    print('Precentage visible joints - Test: ' + str(percentage_test_vis.item() * 100))

    sum_test_vis = (test_vis == 1).float().sum()
    print('Number visible joints - Test: ' + str(sum_test_vis.item()))

    sum_test_non_vis = (test_vis == 0).float().sum()
    print('Number non-visible joints - Test: ' + str(sum_test_non_vis.item()))
