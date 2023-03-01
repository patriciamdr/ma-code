import argparse
import logging
import os

import numpy as np
import torch
from einops import rearrange
from tqdm import tqdm

from common.camera import normalize_screen_coordinates
from common.h36m_dataset import Human36mDataset
from common.opt import graformer_opts
from common.utils import define_actions, define_error_mpjpe_list, define_acc_list, define_binary_class_metrics_list, \
    test_calculation_binary_class_metrics, test_calculation_acc, test_calculation_mpjpe, print_error_mpjpe, print_acc, \
    print_binary_class_metrics, get_variable, get_head_size, get_normalized_distance
from model.auxiliary_models import AuxiliaryVisModel
from model.block.utils import edges_h36m, adj_mx_from_edges
from common.load_data_h36m import Fusion as Fusion_h36m

def main(opt):
    manualSeed = 0

    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.cuda.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)

    print('==> Using settings {}'.format(opt))

    print('==> Loading dataset...')
    dataset_dir = os.path.join(opt.root_path, opt.dataset)
    if opt.dataset == 'h36m':
        edges = edges_h36m
        dataset_path = os.path.join(dataset_dir, 'data_3d_' + opt.dataset + '.npz')
        dataset = Human36mDataset(dataset_path, opt)

        test_data = Fusion_h36m(opt=opt, train=False, dataset=dataset, root_path=dataset_dir, keypoints=opt.keypoints)
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=opt.batch_size,
                                                      shuffle=False, num_workers=int(opt.workers), pin_memory=True)
    else:
        raise KeyError('Invalid dataset')

    actions = define_actions(opt.actions, opt.dataset)

    # Create model
    print("==> Creating model...")
    adj = adj_mx_from_edges(num_pts=opt.n_joints, edges=edges, sparse=False)
    model = AuxiliaryVisModel(adj=adj.cuda(), in_dim=opt.in_dim, hid_dim=opt.dim_model, n_pts=opt.n_joints,
                              pose_embed_dim=opt.pose_embed_dim, vis_embed_dim=opt.vis_embed_dim,
                              num_layers=opt.n_layer, n_head=opt.n_head, dropout=opt.dropout,
                              lin_layers=opt.lin_layers).cuda()

    print("==> Total parameters: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))

    count_model_params = sum(p.numel() for p in model.parameters())
    print('INFO: Parameter count:', count_model_params)
    count_trainable_model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('INFO: Trainable parameter count:', count_trainable_model_params)

    ckpt_path = opt.evaluate

    if os.path.isfile(ckpt_path):
        print("==> Loading checkpoint '{}'".format(ckpt_path))
        ckpt = torch.load(ckpt_path)
        start_epoch = ckpt['epoch']
        if len(ckpt['extra']) == 2:
            opt.previous_best_threshold = ckpt['extra']
        else:
            opt.previous_best_threshold, opt.step = ckpt['extra']

        model.load_state_dict(ckpt['model_pos'])
        print("==> Loaded checkpoint (Epoch: {} | Error: {} | Acc: {})".format(start_epoch,
                                                                               opt.previous_best_threshold[0],
                                                                               opt.previous_best_threshold[1]))
        opt.checkpoint = os.path.dirname(ckpt_path)
    else:
        raise RuntimeError("==> No checkpoint found at '{}'".format(ckpt_path))

    t = 0
    while t <= 1.0:
        print(t)
        p1, p2, acc = step(opt, actions, test_dataloader, model, t)

        info = 'p1: %.2f, acc: %.4f' % (p1, acc)
        logging.info(info)
        print(info)
        t += 0.1

    return


def step(opt, actions, dataLoader, model, threshold):
    model.eval()

    action_error_sum_pose = define_error_mpjpe_list(actions)
    action_error_sum_vis_acc = define_acc_list(actions)

    for i, data in enumerate(tqdm(dataLoader)):
        batch_cam, gt_3d, gt_2d, vis, inputs_2d, inputs_scores, dist, scale, bb_box, extra = data
        action, subject, cam_ind = extra

        if opt.ground_truth_input:
            inputs_2d = gt_2d
            inputs_scores[:] = 1

        [inputs_2d, inputs_scores, gt_2d, vis, gt_3d, dist, batch_cam, scale, bb_box] = \
            get_variable('test', [inputs_2d, inputs_scores, gt_2d, vis, gt_3d, dist, batch_cam, scale, bb_box])

        target = gt_3d.clone()
        target[:, :, 0] = 0

        min_thr = threshold
        max_thr = threshold + 0.1
        vis_mask = (max_thr > dist).logical_and(dist >= min_thr)

        inputs_2d, pose_prediction, vis_prediction = input_augmentation(inputs_2d, model)

        action_error_sum_vis_acc = test_calculation_acc(vis_prediction, vis, action, action_error_sum_vis_acc, opt.dataset, vis_mask)

        pose_prediction[:, :, 0, :] = 0
        action_error_sum_pose = test_calculation_mpjpe(pose_prediction, target, action, action_error_sum_pose, opt.dataset)

    p1, p2 = print_error_mpjpe(opt.dataset, action_error_sum_pose, opt.train)
    acc = print_acc(opt.dataset, action_error_sum_vis_acc, opt.train)
    return p1, p2, acc

def input_augmentation(input_2D, model_trans):
    joints_left = [4, 5, 6, 11, 12, 13]
    joints_right = [1, 2, 3, 14, 15, 16]

    input_2D_non_flip = input_2D[:, 0]
    input_2D_flip = input_2D[:, 1]

    output_3D_non_flip, output_3D_non_flip_VTE, output_non_flip_vis, _, _, _, _ = model_trans(input_2D_non_flip)
    output_3D_flip, output_3D_flip_VTE, output_flip_vis, _, _, _, _ = model_trans(input_2D_flip)

    output_3D_flip_VTE[:, :, :, 0] *= -1
    output_3D_flip[:, :, :, 0] *= -1

    output_3D_flip_VTE[:, :, joints_left + joints_right, :] = output_3D_flip_VTE[:, :, joints_right + joints_left, :]
    output_3D_flip[:, :, joints_left + joints_right, :] = output_3D_flip[:, :, joints_right + joints_left, :]

    output_3D_VTE = (output_3D_non_flip_VTE + output_3D_flip_VTE) / 2
    output_3D = (output_3D_non_flip + output_3D_flip) / 2

    input_2D = input_2D_non_flip

    output_flip_vis[:, :, joints_left + joints_right, :] = output_flip_vis[:, :, joints_right + joints_left, :]
    output_vis = (output_non_flip_vis + output_flip_vis) / 2

    return input_2D, output_3D, output_3D_VTE, output_vis



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--self_supervision', action='store_true')
    parser.add_argument('--lin_layers', action='store_true')
    parser.add_argument('--pos_weight', default=(6577949/(6577949 + 19937836)))
    parser.add_argument('--pose_weight_factor', type=float, default=1.0)
    parser.add_argument('--pretrained_graformer_init', action='store_true')
    parser.add_argument('--pretrained_graformer', type=str,
                        default='checkpoint/pretrained/graformer/small/best_83_5448.pth')

    opt = graformer_opts(parser).get_auxiliary_vis_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

    main(opt)
