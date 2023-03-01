import argparse
import os.path as path
import torch.nn as nn
import logging

from common.h36m_dataset import Human36mDataset
from common.opt import graformer_opts
from torch.utils.data import DataLoader
from common.unrealcv_dataset import UnrealCvDataset
from common.utils import *
from model.auxiliary_models import AuxiliaryVisModel
from model.block.utils import adj_mx_from_edges, edges_unrealcv, edges_h36m
from common.load_data_h36m import Fusion as Fusion_h36m
from common.load_data_unrealcv import Fusion as Fusion_unrealcv
from tqdm import tqdm


def main(opt):
    manualSeed = 600

    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.cuda.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)

    print('==> Using settings {}'.format(opt))

    print('==> Loading dataset...')
    dataset_dir = os.path.join(opt.root_path, opt.dataset)
    if opt.dataset == 'unrealcv':
        edges = edges_unrealcv
        if opt.train:
            dataset = UnrealCvDataset(dataset_dir, train=True)
            train_data = Fusion_unrealcv(opt=opt, dataset=dataset, train=True, keypoint_file=opt.keypoints)
            train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=opt.batch_size,
                                                           shuffle=True, num_workers=int(opt.workers),
                                                           pin_memory=True)

        dataset = UnrealCvDataset(dataset_dir)
        test_data = Fusion_unrealcv(opt=opt, dataset=dataset, keypoint_file=opt.keypoints)
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=opt.batch_size,
                                                      shuffle=False, num_workers=int(opt.workers),
                                                      pin_memory=True)

    elif opt.dataset == 'h36m':
        edges = edges_h36m
        dataset_path = os.path.join(dataset_dir, 'data_3d_' + opt.dataset + '.npz')
        dataset = Human36mDataset(dataset_path, opt)

        if opt.train:
            train_data = Fusion_h36m(opt=opt, train=True, dataset=dataset, root_path=dataset_dir, keypoints=opt.keypoints)
            train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=opt.batch_size,
                                                           shuffle=True, num_workers=int(opt.workers), pin_memory=True)

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

    if opt.pos_weight == str(None):
        criterion_list = {'pose': nn.MSELoss(reduction='mean').cuda(), 'vis': nn.BCEWithLogitsLoss().cuda()}
    else:
        criterion_list = {'pose': nn.MSELoss(reduction='mean').cuda(), 'vis': nn.BCEWithLogitsLoss(pos_weight=torch.tensor(opt.pos_weight)).cuda()}

    if opt.pretrained_graformer_init:
        ckpt = torch.load(opt.pretrained_graformer)
        model.load_state_dict(ckpt['model_pos'], strict=False)

    if opt.freeze_main_pipeline:
        model.freeze_main_pipeline()
        opt.pose_weight_factor = 0

    # optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=opt.lr)

    if opt.lin_layers:
        optimizer = torch.optim.Adam([
            {"params": model.gconv_input.parameters()},
            {"params": model.gconv_layers.parameters()},
            {"params": model.atten_layers.parameters()},
            {"params": model.last_gconv_layer.parameters()},
            {"params": model.head.parameters()},
            {"params": model.visibility_class_head.parameters(), "lr": opt.lr_vis},
            {"params": model.vis_branch.parameters(), "lr": opt.lr_vis},
        ], lr=opt.lr, amsgrad=True)
    else:
        optimizer = torch.optim.Adam([
            {"params": model.gconv_input.parameters()},
            {"params": model.gconv_layers.parameters()},
            {"params": model.atten_layers.parameters()},
            {"params": model.last_gconv_layer.parameters()},
            {"params": model.head.parameters()},
            {"params": model.gconv_layers_vis.parameters(), "lr": opt.lr_vis},
            {"params": model.atten_layers_vis.parameters(), "lr": opt.lr_vis},
            {"params": model.last_gconv_layer_vis.parameters(), "lr": opt.lr_vis},
            {"params": model.visibility_class_head.parameters(), "lr": opt.lr_vis},
        ], lr=opt.lr, amsgrad=True)

    count_model_params = sum(p.numel() for p in model.parameters())
    print('INFO: Parameter count:', count_model_params)
    count_trainable_model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('INFO: Trainable parameter count:', count_trainable_model_params)

    if opt.resume or opt.evaluate:
        ckpt_path = (opt.resume if opt.resume else opt.evaluate)

        if path.isfile(ckpt_path):
            print("==> Loading checkpoint '{}'".format(ckpt_path))
            ckpt = torch.load(ckpt_path)
            start_epoch = ckpt['epoch']
            # opt.previous_best_threshold, opt.step = ckpt['extra']
            p1, acc, opt.step = ckpt['extra']
            opt.previous_best_threshold = (p1, acc)
            model.load_state_dict(ckpt['model_pos'])
            optimizer.load_state_dict(ckpt['optimizer'])
            opt.lr_now = optimizer.param_groups[0]['lr']
            opt.vis_lr_now = optimizer.param_groups[5]['lr']
            print("==> Loaded checkpoint (Epoch: {} | Error: {} | Acc: {})".format(start_epoch,
                                                                                   opt.previous_best_threshold[0],
                                                                                   opt.previous_best_threshold[1]))
            start_epoch += 1
            opt.checkpoint = path.dirname(ckpt_path)
        else:
            raise RuntimeError("==> No checkpoint found at '{}'".format(ckpt_path))
    else:
        start_epoch = 0
        opt.step = 0
        opt.lr_now = opt.lr
        opt.vis_lr_now = opt.lr_vis

        if opt.train:
            logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S',
                                filename=os.path.join(opt.checkpoint, 'train.log'), level=logging.INFO)

    for epoch in range(start_epoch, opt.epochs):
        if opt.train:
            print('\nEpoch: %d | LR: %.8f | LR Vis: %.8f' % (epoch, opt.lr_now, opt.vis_lr_now))
            # Train for one epoch
            loss_total, loss_vis, loss_pose, train_p1, train_p2, train_acc, =\
                train(opt, actions, train_dataloader, model, criterion_list, optimizer)

        # Evaluate
        p1, p2, acc, ap, npv, tnr, tpr, test_loss_vis, test_loss_pose = evaluate(opt, actions, test_dataloader, model, criterion_list)

        # Save checkpoint
        if opt.previous_best_threshold[0] > p1:
            opt.previous_best_threshold = (p1, acc)
            opt.previous_name = save_model(opt.previous_name, opt.checkpoint, epoch, p1, optimizer, model, 'best',
                                           extra=(opt.previous_best_threshold, opt.step))

        if (epoch + 1) % opt.snapshot == 0:
            save_model(None, opt.checkpoint, epoch, p1, optimizer, model, 'snapshot',
                       extra=(p1, acc, opt.step))

        if not opt.train:
            print('p1: %.2f, acc: %.2f' % (p1, acc))
            info = 'test_loss_pose: %.6f, test_loss_vis: %.6f, p1: %.2f, acc: %.4f, ap: %.4f, npv: %.4f, tnr: %.4f, tpr: %.4f' % (
                test_loss_pose, test_loss_vis, p1, acc, ap, npv, tnr, tpr)
            print(info)
            break
        else:
            info = 'epoch: %d, lr: %.7f, vis_lr: %.7f, loss_total: %.6f, loss_pose: %.6f, loss_vis: %.6f, ' \
                   'train_p1: %.2f, train_acc: %.4f, ' \
                   'test_loss_pose: %.6f, test_loss_vis: %.6f, p1: %.2f, acc: %.4f, ap: %.4f, npv: %.4f, tnr: %.4f, tpr: %.4f' % (
                       epoch, opt.lr_now, opt.vis_lr_now, loss_total, loss_pose, loss_vis, train_p1, train_acc,
                       test_loss_pose, test_loss_vis, p1, acc, ap, npv, tnr, tpr)
            logging.info(info)
            print(info)
    return


def train(opt, actions, train_loader, model, criterion_list, optimizer):
    return step('train', opt, actions, train_loader, model, criterion_list, optimizer)


def evaluate(opt, actions, val_loader, model, criterion_list):
    with torch.no_grad():
        return step('test', opt, actions, val_loader, model, criterion_list)


def step(split, opt, actions, dataLoader, model, criterion_list, optimizer=None):
    if split == 'train':
        model.train()
        if opt.freeze_main_pipeline and opt.set_main_pipeline_eval_mode:
            model.set_main_pipeline_eval_mode()
    else:
        model.eval()

    epoch_loss_3d = AccumLoss()
    epoch_loss_3d_vis = AccumLoss()
    epoch_loss_3d_pose = AccumLoss()
    epoch_loss_3d_vis_test = AccumLoss()
    epoch_loss_3d_pose_test = AccumLoss()
    action_error_sum_pose = define_error_mpjpe_list(actions)
    action_error_sum_vis_acc = define_acc_list(actions)
    action_error_sum_vis_binary_class_metrics = define_binary_class_metrics_list(actions)

    for i, data in enumerate(tqdm(dataLoader)):
        if opt.dataset == 'unrealcv':
            batch_cam, gt_3d, gt_2d, vis, inputs_2d, inputs_scores, extra = data
            video_id, cam_ind, action = extra

            if opt.ground_truth_input:
                inputs_2d[:, :, :9] = gt_2d[:, :, :9]
                inputs_2d[:, :, 11:] = gt_2d[:, :, 9:]
                inputs_scores[:, :, 9] = 1
                inputs_scores[:, :, 11:] = 1

            [inputs_2d, inputs_scores, gt_3d, batch_cam, vis] = get_variable(split, [inputs_2d, inputs_scores, gt_3d, batch_cam, vis])
        else:
            batch_cam, gt_3d, gt_2d, vis, inputs_2d, inputs_scores, dist, scale, bb_box, extra = data
            action, subject, cam_ind = extra

            if opt.ground_truth_input:
                inputs_2d = gt_2d
                inputs_scores[:] = 1

            ## 2D
            # image = show2Dpose(inputs_2d[0, 0], np.zeros((1000, 1000, 3)))
            # cv2.imwrite('/home/patricia/dev/StridedTransformer-Pose3D/' + str(('%04d'% i)) + '_2D.png', image)

            # inputs_2d[0, 0] = torch.from_numpy(normalize_screen_coordinates(inputs_2d[0, 0].numpy(), w=1000, h=1000))

            [inputs_2d, inputs_scores, gt_2d, vis, gt_3d, batch_cam, scale, bb_box] = \
                get_variable(split, [inputs_2d, inputs_scores, gt_2d, vis, gt_3d, batch_cam, scale, bb_box])

        target = gt_3d.clone()
        target[:, :, 0] = 0

        if opt.in_dim == 1:
            pose_prediction, vis_prediction = model(inputs_scores)
        elif opt.in_dim == 2:
            if split == 'train':
                pose_prediction, vis_prediction = model(inputs_2d)
            else:
                inputs_2d, pose_prediction, vis_prediction = input_augmentation(inputs_2d, model)
        elif opt.in_dim == 3:
            input = torch.cat((inputs_2d, inputs_scores), dim=3)
            pose_prediction, vis_prediction = model(input)
        else:
            raise KeyError('Invalid input dimension')

        N = inputs_2d.size(0)
        if split == 'train':

            opt.step += 1
            if opt.step % opt.lr_decay == 0 or opt.step == 1:
                opt.lr_now = lr_decay(opt.step, opt.lr, opt.lr_decay, opt.lr_gamma)
                for param_group in optimizer.param_groups[0:5]:
                    param_group['lr'] = opt.lr_now
                opt.vis_lr_now = lr_decay(opt.step, opt.lr_vis, opt.lr_decay, opt.lr_gamma)
                for param_group in optimizer.param_groups[5:7]:
                    param_group['lr'] = opt.vis_lr_now

            loss_vis = criterion_list['vis'](vis_prediction, vis)
            if opt.pose_weight_factor > 0:
                loss_pose = criterion_list['pose'](pose_prediction, target)
                loss = opt.pose_weight_factor * loss_pose + loss_vis
            else:
                loss = loss_vis

            optimizer.zero_grad()
            loss.backward()
            if opt.max_norm:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()

            epoch_loss_3d.update(loss.detach().cpu().numpy() * N, N)
            epoch_loss_3d_vis.update(loss_vis.detach().cpu().numpy() * N, N)
            if opt.pose_weight_factor > 0:
                epoch_loss_3d_pose.update(loss_pose.detach().cpu().numpy() * N, N)
        elif split == 'test':
            loss_vis = nn.BCEWithLogitsLoss().cuda()(vis_prediction, vis)
            loss_pose = nn.MSELoss(reduction='mean').cuda()(pose_prediction, target)
            epoch_loss_3d_vis_test.update(loss_vis.detach().cpu().numpy() * N, N)
            epoch_loss_3d_pose_test.update(loss_pose.detach().cpu().numpy() * N, N)

            action_error_sum_vis_binary_class_metrics = \
                test_calculation_binary_class_metrics(vis_prediction, vis, action, action_error_sum_vis_binary_class_metrics, opt.dataset)

        action_error_sum_vis_acc = test_calculation_acc(vis_prediction, vis, action, action_error_sum_vis_acc, opt.dataset)

        pose_prediction[:, :, 0, :] = 0
        action_error_sum_pose = test_calculation_mpjpe(pose_prediction, target, action, action_error_sum_pose, opt.dataset)

    p1, p2 = print_error_mpjpe(opt.dataset, action_error_sum_pose, opt.train)
    acc = print_acc(opt.dataset, action_error_sum_vis_acc, opt.train)

    if split == 'train':
        return epoch_loss_3d.avg, epoch_loss_3d_vis.avg, epoch_loss_3d_pose.avg, p1, p2, acc
    elif split == 'test':
        ap, npv, tnr, tpr = print_binary_class_metrics(opt.dataset, action_error_sum_vis_binary_class_metrics, opt.train)
        return p1, p2, acc, ap, npv, tnr, tpr, epoch_loss_3d_vis_test.avg, epoch_loss_3d_pose_test.avg


def input_augmentation(input_2D, model):
    joints_left = [4, 5, 6, 11, 12, 13]
    joints_right = [1, 2, 3, 14, 15, 16]

    input_2D_non_flip = input_2D[:, 0]
    input_2D_flip = input_2D[:, 1]

    output_3D_non_flip_pose, output_3D_non_flip_vis = model(input_2D_non_flip)
    output_3D_flip_pose, output_3D_flip_vis = model(input_2D_flip)

    # output_3D_flip_vis[:, :, :, 0] *= -1
    output_3D_flip_pose[:, :, :, 0] *= -1

    output_3D_flip_pose[:, :, joints_left + joints_right, :] = output_3D_flip_pose[:, :, joints_right + joints_left, :]
    output_3D_flip_vis[:, :, joints_left + joints_right, :] = output_3D_flip_vis[:, :, joints_right + joints_left, :]

    output_3D_vis = (output_3D_non_flip_vis + output_3D_flip_vis) / 2
    output_3D_pose = (output_3D_non_flip_pose + output_3D_flip_pose) / 2

    input_2D = input_2D_non_flip

    return input_2D, output_3D_pose, output_3D_vis


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--self_supervision', action='store_true')
    parser.add_argument('--pos_weight', default=(6577949/(6577949 + 19937836)))
    parser.add_argument('--lin_layers', action='store_true')
    parser.add_argument('--pose_weight_factor', type=float, default=1.0)
    parser.add_argument('--pretrained_graformer_init', action='store_true')
    parser.add_argument('--pretrained_graformer', type=str,
                        default='checkpoint/pretrained/graformer/small/best_83_5448.pth')
    parser.add_argument('--freeze_main_pipeline', action='store_true')

    opt = graformer_opts(parser).get_auxiliary_vis_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

    main(opt)

