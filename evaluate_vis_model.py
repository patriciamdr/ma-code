import argparse
import logging
import os

import numpy as np
import torch
from common.opt import opts
from einops import rearrange
from tqdm import tqdm
import torch.utils.data
import glob

from model.block.refine import refine
from model.strided_vis_graformer import Model as StridedVisGraformer
from common.utils import *
from common.camera import get_uvd2xyz
from common.load_data_h36m import Fusion as Fusion_h36m
from common.h36m_dataset import Human36mDataset

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
        dataset_path = os.path.join(dataset_dir, 'data_3d_' + opt.dataset + '.npz')
        dataset = Human36mDataset(dataset_path, opt)

        test_data = Fusion_h36m(opt=opt, train=False, dataset=dataset, root_path=dataset_dir, keypoints=opt.keypoints)
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=opt.batch_size,
                                                      shuffle=False, num_workers=int(opt.workers), pin_memory=True)
    else:
        raise KeyError('Invalid dataset')

    actions = define_actions(opt.actions, opt.dataset)

    # Create model
    model = {}
    model_trans = StridedVisGraformer(opt).cuda()

    # Use pretrained weights for spatial part without pose regression head
    if opt.pretrained_spatial_module_init:
        filename = os.path.join(opt.pretrained_spatial_module_dir, opt.pretrained_spatial_module)
        pretrained_dict = torch.load(filename)['model_pos']

        model_trans.Transformer.load_state_dict(pretrained_dict, strict=False)
        opt.freeze_spatial_module = True
        model_trans.freeze_spatial_module()

    model['trans'] = model_trans
    model['refine'] = refine(opt).cuda()

    model_dict = model['trans'].state_dict()

    all_param = []
    lr_spatial = opt.spatial_module_lr
    lr = opt.lr
    lr_refine = opt.lr_refine
    for i_model in model:
        all_param += list(model[i_model].parameters())

    epoch_start = 1
    if opt.reload:
        model_path = sorted(glob.glob(os.path.join(opt.previous_dir, '*.pth')))

        no_refine_path = []
        for path in model_path:
            if path.split('/')[-1][0] == 'n' and 'best' in path:
                no_refine_path = path
                print(no_refine_path)
                break

        pre_dict = torch.load(no_refine_path)
        pre_dict_model = pre_dict['model_pos']
        for name, key in model_dict.items():
            model_dict[name] = pre_dict_model[name]
        model['trans'].load_state_dict(model_dict)

        if opt.freeze_spatial_module:
            model['trans'].freeze_spatial_module()

        if opt.freeze_trans_module:
            model['trans'].freeze()

    refine_dict = model['refine'].state_dict()

    if opt.refine_reload:
        model_path = sorted(glob.glob(os.path.join(opt.previous_dir, '*.pth')))

        refine_path = []
        for path in model_path:
            if path.split('/')[-1][0] == 'r' and 'best' in path:
                refine_path = path
                print(refine_path)
                break

        pre_dict_refine = torch.load(refine_path)
        pre_dict_refine_model = pre_dict_refine['model_pos']
        for name, key in refine_dict.items():
            refine_dict[name] = pre_dict_refine_model[name]
        model['refine'].load_state_dict(refine_dict)

    count_model_params = sum(p.numel() for p in all_param)
    print('INFO: Parameter count:', count_model_params)
    count_trainable_model_params = sum(p.numel() for p in all_param if p.requires_grad)
    print('INFO: Trainable parameter count:', count_trainable_model_params)

    t = 1.0
    while t <= 1.0:
        print(t)
        # p1, p2, acc, ap, tnr = step(opt, actions, test_dataloader, model, t)
        p1, p2, acc = step(opt, actions, test_dataloader, model, t)

        # info = 'p1: %.2f, acc: %.4f, ap: %.4f, tnr: %.4f' % (p1, acc, ap, tnr)
        info = 'p1: %.2f, acc: %.4f' % (p1, acc)
        logging.info(info)
        print(info)
        t += 0.1
    return


def step(opt, actions, dataLoader, model, threshold):
    model_trans = model['trans']
    model_refine = model['refine']

    model_trans.eval()
    model_refine.eval()

    action_error_sum = define_error_mpjpe_list(actions)
    action_error_sum_refine = define_error_mpjpe_list(actions)
    action_error_sum_vis_acc = define_acc_list(actions)
    # action_error_sum_vis_binary_class_metrics = define_binary_class_metrics_list(actions)

    min_thr = threshold
    max_thr = threshold + 0.1

    for i, data in enumerate(tqdm(dataLoader, 0)):
        batch_cam, gt_3D, gt_2D, vis, input_2D, inputs_2D_score, dist, scale, bb_box, extra = data
        action, subject, cam_ind = extra
        [input_2D, vis, gt_3D, gt_2D, batch_cam, scale, bb_box, dist] = get_variable('test', [input_2D, vis, gt_3D, gt_2D, batch_cam, scale, bb_box, dist])

        input_2D, output_3D, output_3D_VTE, output_vis = input_augmentation_vis(input_2D, model_trans)

        out_target = gt_3D.clone()
        out_target[:, :, 0] = 0

        if out_target.size(1) > 1:
            out_target_single = out_target[:, opt.pad].unsqueeze(1)
            gt_3D_single = gt_3D[:, opt.pad].unsqueeze(1)
        else:
            out_target_single = out_target
            gt_3D_single = gt_3D

        if opt.refine:
            pred_uv = input_2D[:, opt.pad, :, :].unsqueeze(1)
            uvd = torch.cat((pred_uv, output_3D[:, :, :, 2].unsqueeze(-1)), -1)
            xyz = get_uvd2xyz(uvd, gt_3D_single, batch_cam)
            xyz[:, :, 0, :] = 0
            output_3D = model_refine(output_3D, xyz)

        N, F = input_2D.size(0), input_2D.size(1)
        output_3D[:, :, 0, :] = 0

        # vis_mask = (max_thr > dist).logical_and(dist >= min_thr)
        vis_mask = dist >= min_thr

        action_error_sum = test_calculation_mpjpe(output_3D, out_target, action, action_error_sum)

        if opt.refine:
            action_error_sum_refine = test_calculation_mpjpe(output_3D, out_target, action, action_error_sum_refine)

        action_error_sum_vis_acc = test_calculation_acc(output_vis, vis, action, action_error_sum_vis_acc, opt.dataset, vis_mask)
        # action_error_sum_vis_binary_class_metrics = test_calculation_binary_class_metrics(output_vis, vis, action, action_error_sum_vis_binary_class_metrics, opt.dataset, vis_mask)

    acc = print_acc(opt.dataset, action_error_sum_vis_acc, opt.train)
    # ap, npv, tnr, tpr = print_binary_class_metrics(opt.dataset, action_error_sum_vis_binary_class_metrics, opt.train)
    if opt.refine:
        p1, p2 = print_error_mpjpe(opt.dataset, action_error_sum_refine, opt.train)
    else:
        p1, p2 = print_error_mpjpe(opt.dataset, action_error_sum, opt.train)
    # return p1, p2, acc, ap, tnr
    return p1, p2, acc


def input_augmentation_vis(input_2D, model_trans):
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
    opt = opts().parse()
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

    main(opt)
