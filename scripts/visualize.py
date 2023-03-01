# Source: https://gitlab.lrz.de/ge46xav/posegraphnet/-/blob/master/code/visualize.py
import glob
import sys
import os
sys.path.append(os.getcwd())
import random

import matplotlib.pyplot as plt
# This import registers the 3D projection, but is otherwise unused.
import torch
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MultipleLocator
import cv2
import numpy as np
import io

from tqdm import tqdm

from common.camera import get_uvd2xyz
from common.h36m_dataset import Human36mDataset
from common.utils import define_actions, get_variable
from common.load_data_h36m import Fusion as Fusion_h36m
from model.block.refine import refine
from model.strided_graformer import Model as StridedGraformer
from model.strided_vis_graformer import Model as StridedVisGraformer
import re
import seaborn as sns
import json

import matplotlib.pyplot as plt
import itertools

import numpy as np
from scipy.special import softmax

from common.opt import opts


def plot_adjacency_matrix(adj_group, plot_name="../out/plot_adjacency.jpg", attention=False):
    if attention:
        figure, axs = plt.subplots(1, adj_group.shape[0], figsize=(80,20))
    else:
        figure, axs = plt.subplots(1, adj_group.shape[0], figsize=(40,20))

    joint_names = ['Root', 'R hip', 'R knee', 'R foot', 'L hip', 'L knee', 'L foot', 'Spine', 'Thorax', 'Neck', 'Head', 'L shoulder', 'L elbow','L wrist','R shoulder','R elbow','R wrist']

    adj_group = np.around(adj_group, decimals=2)

    if adj_group.shape[0] == 1:
        axs = [axs]

    for group_id in range(adj_group.shape[0]):
        ax = axs[group_id]
        adj = adj_group[group_id]
        if attention:
            im = ax.imshow(adj, vmin=0,vmax=1, cmap=plt.cm.viridis)
        else:
            im = ax.imshow(adj, vmin=-1,vmax=1, interpolation='nearest', cmap=plt.cm.Blues)
        tick_marks = np.arange(len(joint_names))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(joint_names, rotation=90, fontsize=20, horizontalalignment="right")
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(joint_names, fontsize=20)
        ax.set_ylim(len(adj)-0.5, -0.5)
        if attention:
            ax.set_title('Head ' + str(group_id), fontsize=20)
        else:
            ax.set_title('(' + str(group_id) + ')', fontsize=20)

        # Use white text if squares are dark; otherwise black.
        threshold = 0.5
        for i, j in itertools.product(range(adj.shape[0]), range(adj.shape[1])):
            if attention:
                color = "black" if adj[i, j] > threshold else "white"
                ax.text(j, i, adj[i, j], horizontalalignment="center", color=color)
            else:
                color = "white" if adj[i, j] > threshold else "black"
                ax.text(j, i, adj[i, j], horizontalalignment="center", color=color, fontsize=15)

    # figure.colorbar(im, ax=ax, shrink=0.75)
    cbar = figure.colorbar(im, ax=axs.ravel().tolist(), shrink=0.75)

    plt.savefig(plot_name, bbox_inches="tight")
    return

def plot_attention_matrix(adj_group, plot_name="../out/plot_attention.jpg", frames=81):
    figure, axs = plt.subplots(2, 4, figsize=(20,8))
    # figure, axs = plt.subplots(1, 1, figsize=(20,20))

    adj_group = np.around(adj_group, decimals=2)

    if adj_group.shape[0] == 1:
        axs = [axs]

    for group_id in range(adj_group.shape[0]):
        ax = axs[int(group_id / 4), group_id % 4]
        adj = adj_group[group_id]
        im = ax.imshow(adj, vmin=0, vmax=1, cmap=plt.cm.viridis)
        tick_marks = np.arange(frames)
        # ax.set_xticks(tick_marks)
        # ax.set_xticklabels(frame_names, rotation=90, fontsize=20, horizontalalignment="right")
        # ax.set_yticks(tick_marks)
        # ax.set_yticklabels(frame_names, fontsize=20)
        # ax.set_ylim(len(adj)-0.5, -0.5)
        ax.set_title('Head ' + str(group_id))

        # Use white text if squares are dark; otherwise black.
        # threshold = 0.5
        # for i, j in itertools.product(range(adj.shape[0]), range(adj.shape[1])):
        #     color = "white" if adj[i, j] > threshold else "black"
        #     ax.text(j, i, adj[i, j], horizontalalignment="center", color=color, fontsize=15)

    # figure.colorbar(im, ax=ax, shrink=0.5)
    cbar = figure.colorbar(im, ax=axs.ravel().tolist())

    plt.savefig(plot_name, bbox_inches="tight")
    return

def val(opt, actions, val_loader, model):
    with torch.no_grad():
        return step('test', opt, actions, val_loader, model)

def step(split, opt, actions, dataLoader, model, optimizer=None, epoch=None):
    model_trans = model['trans']
    model_refine = model['refine']

    model_trans.eval()
    model_refine.eval()

    att_mat_VTE_list, att_mat_STE_list, att_mat_SGT_list, att_mat_SGT_list2  = [], [], [], []
    N = 0
    for i, data in enumerate(tqdm(dataLoader, 0)):
        batch_cam, gt_3D, gt_2d, vis, input_2D, inputs_2D_score, dist, scale, bb_box, extra = data
        action, subject, cam_ind = extra
        [input_2D, vis, gt_3D, batch_cam, scale, bb_box] = get_variable(split, [input_2D, vis, gt_3D, batch_cam, scale, bb_box])

        if opt.use_visibility:
            input_2D, output_3D, output_3D_VTE, attn_mat_SGT, attn_mat_SGT2, attn_mat_VTE, attn_mat_STE = input_augmentation_vis(input_2D, model_trans)
        else:
            input_2D, output_3D, output_3D_VTE, attn_mat_SGT, attn_mat_SGT2, attn_mat_VTE, attn_mat_STE = input_augmentation(input_2D, model_trans)
        # att_mat_SGT_list.append(attn_mat_SGT2.detach())
        # N += attn_mat_VTE.shape[0] * 81
        # att_mat_VTE_list.append(attn_mat_VTE.detach())
        att_mat_STE_list.append(attn_mat_STE.detach())

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

    # att_mat_SGT = torch.zeros(4, 17, 17).cuda()
    # for i in range(len(att_mat_SGT_list)):
    #     att_mat_SGT += att_mat_SGT_list[i]
    # print(N)

    # att_mat_VTE = torch.zeros(8, 81, 81).cuda()
    # N = 0
    # for i in range(len(att_mat_VTE_list)):
    #     N += att_mat_VTE_list[i].shape[0]
    #     att_mat_VTE += torch.sum(att_mat_VTE_list[i], dim=0)
    # print(N)

    att_mat_STE = torch.zeros(8, 81, 81).cuda()
    N = 0
    for i in range(len(att_mat_STE_list)):
        N += att_mat_STE_list[i].shape[0]
        att_mat_STE += torch.sum(att_mat_STE_list[i], dim=0)
    print(N)

    return att_mat_STE.cpu(), N

def input_augmentation_vis(input_2D, model_trans):
    joints_left = [4, 5, 6, 11, 12, 13]
    joints_right = [1, 2, 3, 14, 15, 16]

    input_2D_non_flip = input_2D[:, 0]
    input_2D_flip = input_2D[:, 1]

    attn_weights_SGT_list = []
    attn_weights_SGT_list2 = []
    attn_weights_VTE_list = []
    attn_weights_STE_list = []

    output_3D_non_flip, output_3D_non_flip_VTE, _, attn_weights_SGT, attn_weights_SGT2, attn_weights_VTE, attn_weights_STE = model_trans(input_2D_non_flip)
    attn_weights_SGT_list.append(attn_weights_SGT)
    attn_weights_SGT_list2.append(attn_weights_SGT2)
    attn_weights_VTE_list.append(attn_weights_VTE)
    attn_weights_STE_list.append(attn_weights_STE)

    output_3D_flip, output_3D_flip_VTE, _, attn_weights_SGT, attn_weights_SGT2, attn_weights_VTE, attn_weights_STE = model_trans(input_2D_flip)
    attn_weights_SGT_list.append(attn_weights_SGT)
    attn_weights_SGT_list2.append(attn_weights_SGT2)
    attn_weights_VTE_list.append(attn_weights_VTE)
    attn_weights_STE_list.append(attn_weights_STE)

    output_3D_flip_VTE[:, :, :, 0] *= -1
    output_3D_flip[:, :, :, 0] *= -1

    output_3D_flip_VTE[:, :, joints_left + joints_right, :] = output_3D_flip_VTE[:, :, joints_right + joints_left, :]
    output_3D_flip[:, :, joints_left + joints_right, :] = output_3D_flip[:, :, joints_right + joints_left, :]

    output_3D_VTE = (output_3D_non_flip_VTE + output_3D_flip_VTE) / 2
    output_3D = (output_3D_non_flip + output_3D_flip) / 2

    input_2D = input_2D_non_flip

    attn_weights_SGT = torch.stack(attn_weights_SGT_list, dim=0).mean(dim=0)
    attn_weights_SGT2 = torch.stack(attn_weights_SGT_list2, dim=0).mean(dim=0)
    attn_weights_VTE = torch.stack(attn_weights_VTE_list, dim=0).mean(dim=0)
    attn_weights_STE = torch.stack(attn_weights_STE_list, dim=0).mean(dim=0)

    return input_2D, output_3D, output_3D_VTE, attn_weights_SGT, attn_weights_SGT2, attn_weights_VTE, attn_weights_STE

def input_augmentation(input_2D, model_trans):
    joints_left = [4, 5, 6, 11, 12, 13]
    joints_right = [1, 2, 3, 14, 15, 16]

    input_2D_non_flip = input_2D[:, 0]
    input_2D_flip = input_2D[:, 1]

    attn_weights_SGT_list = []
    attn_weights_SGT_list2 = []
    attn_weights_VTE_list = []
    attn_weights_STE_list = []

    output_3D_non_flip, output_3D_non_flip_VTE, attn_weights_SGT, attn_weights_SGT2, attn_weights_VTE, attn_weights_STE = model_trans(input_2D_non_flip)
    attn_weights_SGT_list.append(attn_weights_SGT)
    attn_weights_SGT_list2.append(attn_weights_SGT2)
    attn_weights_VTE_list.append(attn_weights_VTE)
    attn_weights_STE_list.append(attn_weights_STE)

    output_3D_flip, output_3D_flip_VTE, attn_weights_SGT, attn_weights_SGT2, attn_weights_VTE, attn_weights_STE = model_trans(input_2D_flip)
    attn_weights_SGT_list.append(attn_weights_SGT)
    attn_weights_SGT_list2.append(attn_weights_SGT2)
    attn_weights_VTE_list.append(attn_weights_VTE)
    attn_weights_STE_list.append(attn_weights_STE)

    output_3D_flip_VTE[:, :, :, 0] *= -1
    output_3D_flip[:, :, :, 0] *= -1

    output_3D_flip_VTE[:, :, joints_left + joints_right, :] = output_3D_flip_VTE[:, :, joints_right + joints_left, :]
    output_3D_flip[:, :, joints_left + joints_right, :] = output_3D_flip[:, :, joints_right + joints_left, :]

    output_3D_VTE = (output_3D_non_flip_VTE + output_3D_flip_VTE) / 2
    output_3D = (output_3D_non_flip + output_3D_flip) / 2

    input_2D = input_2D_non_flip

    attn_weights_SGT = torch.stack(attn_weights_SGT_list, dim=0).mean(dim=0)
    attn_weights_SGT2 = torch.stack(attn_weights_SGT_list2, dim=0).mean(dim=0)
    attn_weights_VTE = torch.stack(attn_weights_VTE_list, dim=0).mean(dim=0)
    attn_weights_STE = torch.stack(attn_weights_STE_list, dim=0).mean(dim=0)

    return input_2D, output_3D, output_3D_VTE, attn_weights_SGT, attn_weights_SGT2, attn_weights_VTE, attn_weights_STE


opt = opts().parse()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu


model = {}
assert opt.spatial_module
if opt.use_visibility:
    model_trans = StridedVisGraformer(opt).cuda()
else:
    model_trans = StridedGraformer(opt).cuda()


model['trans'] = model_trans
model['refine'] = refine(opt).cuda()

model_dict = model['trans'].state_dict()
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

adj_list=[]
for i in range(opt.graformer_depth):
    adj_mtx = model['trans'].Transformer.atten_layers[i].feed_forward.A_hat
    adj_list.append(adj_mtx)
plot_adjacency_matrix(torch.stack(adj_list).cpu().detach().numpy(), "out/plot_adjacency.jpg")

actions_h36m = ["Directions", "Discussion", "Eating", "Greeting",
        "Phoning", "Photo", "Posing", "Purchases",
        "Sitting", "SittingDown", "Smoking", "Waiting",
        "WalkDog", "Walking", "WalkTogether"]

# actions_h36m = ["SittingDown"]

opt.manualSeed = 0

random.seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
torch.cuda.manual_seed(opt.manualSeed)
torch.cuda.manual_seed_all(opt.manualSeed)

root_path = opt.root_path

print('Loading dataset...')
dataset_dir = os.path.join(root_path, opt.dataset)

dataset_path = os.path.join(dataset_dir, 'data_3d_' + opt.dataset + '.npz')
dataset = Human36mDataset(dataset_path, opt)

# attn_SGT_list = []
# attn_SGT_list_b = []
#
# M = 0
# for action in actions_h36m:
#     opt.actions = action
#     attn_SGT_list2 = []
#     attn_SGT_list2_b = []
#     N = 0
#     for subject in ['S11', 'S9']:
#         opt.subjects_test = subject
#
#         test_data = Fusion_h36m(opt=opt, train=False, dataset=dataset, root_path=dataset_dir)
#         test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=opt.batch_size,
#                 shuffle=False, num_workers=int(opt.workers), pin_memory=True)
#
#         opt.out_joints = dataset.skeleton().num_joints()
#
#         actions = define_actions(opt.actions, opt.dataset)
#
#         # Reset seed to get the same occluded val dataset per epoch
#         if opt.occlusion_augmentation_test:
#             test_data.reset_seed(201)
#
#         att_SGT, n = val(opt, actions, test_dataloader, model)
#         N += n
#         attn_SGT_list2.append(att_SGT)
#         # attn_SGT_list2_b.append(att_SGT2)
#         torch.cuda.empty_cache()
#     att_SGT = torch.zeros(4, 17, 17)
#     # att_SGT2 = torch.zeros(4, 17, 17)
#     print(len(attn_SGT_list2))
#     for i in range(len(attn_SGT_list2)):
#         att_SGT += attn_SGT_list2[i]
#         # att_SGT2 += attn_SGT_list2_b[i]
#     print(N)
#     att_SGT /= N
#     # att_SGT2 /= N
#     attn_SGT_list.append(att_SGT)
#     # attn_SGT_list_b.append(att_SGT2)
#     torch.cuda.empty_cache()
#     M += 1
#
# att_SGT = torch.zeros(4, 17, 17)
# # att_SGT2 = torch.zeros(4, 17, 17)
# print(len(attn_SGT_list))
# for i in range(len(attn_SGT_list)):
#     att_SGT += attn_SGT_list[i]
#     # att_SGT2 += attn_SGT_list_b[i]
#
# print(M)
# att_SGT /= M
# # att_SGT2 /= M
# att_mat_SGT = att_SGT.numpy()
# # att_mat_SGT2 = att_SGT2.numpy()
#
# att_SGT_norm = (att_mat_SGT - np.min(att_mat_SGT)) / (np.max(att_mat_SGT) - np.min(att_mat_SGT))
# # att_SGT2_norm = (att_mat_SGT2 - np.min(att_mat_SGT2)) / (np.max(att_mat_SGT2) - np.min(att_mat_SGT2))
# plot_adjacency_matrix(att_SGT_norm, "out/plot_attention_SGT2.jpg", attention=True)
# # plot_attention_matrix(att_SGT2_norm, "out/plot_attention_SGT2.jpg")

# attn_VTE_list = []
#
# M = 0
# for action in actions_h36m:
#     opt.actions = action
#     attn_VTE_list2 = []
#     N = 0
#     for subject in ['S11', 'S9']:
#         opt.subjects_test = subject
#
#         test_data = Fusion_h36m(opt=opt, train=False, dataset=dataset, root_path=dataset_dir)
#         test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=opt.batch_size,
#                 shuffle=False, num_workers=int(opt.workers), pin_memory=True)
#
#         opt.out_joints = dataset.skeleton().num_joints()
#
#         actions = define_actions(opt.actions, opt.dataset)
#
#         # Reset seed to get the same occluded val dataset per epoch
#         if opt.occlusion_augmentation_test:
#             test_data.reset_seed(201)
#
#         att_VTE, n = val(opt, actions, test_dataloader, model)
#         N += n
#         attn_VTE_list2.append(att_VTE)
#         torch.cuda.empty_cache()
#     att_VTE = torch.zeros(8, 81, 81)
#     print(len(attn_VTE_list2))
#     for i in range(len(attn_VTE_list2)):
#         att_VTE += attn_VTE_list2[i]
#     print(N)
#     att_VTE /= N
#     attn_VTE_list.append(att_VTE)
#     torch.cuda.empty_cache()
#     M += 1
#
# att_VTE = torch.zeros(8, 81, 81)
# print(len(attn_VTE_list))
# for i in range(len(attn_VTE_list)):
#     att_VTE += attn_VTE_list[i]
#
# print(M)
# att_VTE /= M
# att_mat_VTE = att_VTE.numpy()
#
# att_VTE_norm = (att_mat_VTE - np.min(att_mat_VTE)) / (np.max(att_mat_VTE) - np.min(att_mat_VTE))
# plot_attention_matrix(att_VTE_norm, "out/plot_attention_VTE.jpg")

attn_STE_list = []

M = 0
for action in actions_h36m:
    opt.actions = action
    attn_STE_list2 = []
    N = 0
    for subject in ['S11', 'S9']:
        opt.subjects_test = subject

        test_data = Fusion_h36m(opt=opt, train=False, dataset=dataset, root_path=dataset_dir)
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=opt.batch_size,
                shuffle=False, num_workers=int(opt.workers), pin_memory=True)

        opt.out_joints = dataset.skeleton().num_joints()

        actions = define_actions(opt.actions, opt.dataset)

        # Reset seed to get the same occluded val dataset per epoch
        if opt.occlusion_augmentation_test:
            test_data.reset_seed(201)

        att_STE, n = val(opt, actions, test_dataloader, model)
        N += n
        attn_STE_list2.append(att_STE)
        torch.cuda.empty_cache()
    att_STE = torch.zeros(8, 81, 81)
    print(len(attn_STE_list2))
    for i in range(len(attn_STE_list2)):
        att_STE += attn_STE_list2[i]
    print(N)
    att_STE /= N
    attn_STE_list.append(att_STE)
    torch.cuda.empty_cache()
    M += 1

att_STE = torch.zeros(8, 81, 81)
print(len(attn_STE_list))
for i in range(len(attn_STE_list)):
    att_STE += attn_STE_list[i]

print(M)
att_STE /= M
att_mat_STE = att_STE.numpy()

att_STE_norm = (att_mat_STE - np.min(att_mat_STE)) / (np.max(att_mat_STE) - np.min(att_mat_STE))
plot_attention_matrix(att_STE_norm, "out/plot_attention_STE.jpg")
