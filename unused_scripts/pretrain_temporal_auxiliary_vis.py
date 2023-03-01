import argparse
import os.path as path
import torch.nn as nn
import logging

from common.h36m_dataset import Human36mDataset
from common.opt import opts, graformer_opts
from torch.utils.data import DataLoader
from common.utils import *
from model.auxiliary_models import AuxiliaryVisModel, AuxiliaryVisTemporalModel
from model.block.utils import adj_mx_from_edges, edges_h36m
from common.load_data_h36m import Fusion as Fusion_h36m
from tqdm import tqdm


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
    model = AuxiliaryVisTemporalModel(adj=adj.cuda(), in_dim=opt.in_dim, hid_dim=opt.dim_model, n_pts=opt.n_joints,
                                      pose_embed_dim=opt.pose_embed_dim, vis_embed_dim=opt.vis_embed_dim,
                                      num_layers=opt.n_layer, n_head=opt.n_head, dropout=opt.dropout,
                                      d_hid=opt.d_hid, frames=opt.frames).cuda()

    print("==> Total parameters: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))

    criterion_list = {'pose': nn.MSELoss(reduction='mean').cuda(), 'vis': nn.BCELoss().cuda(), 'pose_seq': mpjpe_cal}

    if opt.pretrained_graformer_init:
        ckpt = torch.load(opt.pretrained_graformer)
        model.load_state_dict(ckpt['model_pos'], strict=False)

        if opt.freeze_main_pipeline:
            model.freeze_main_pipeline()

    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=opt.lr)

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
            opt.previous_best_threshold, opt.step = ckpt['extra']
            model.load_state_dict(ckpt['model_pos'])
            optimizer.load_state_dict(ckpt['optimizer'])
            opt.lr_now = optimizer.param_groups[0]['lr']
            print("==> Loaded checkpoint (Epoch: {} | Error: {} | Acc: {})".format(start_epoch,
                                                                                   opt.previous_best_threshold[0],
                                                                                   opt.previous_best_threshold[1]))
            opt.checkpoint = path.dirname(ckpt_path)
        else:
            raise RuntimeError("==> No checkpoint found at '{}'".format(ckpt_path))
    else:
        start_epoch = 0
        opt.step = 0
        opt.lr_now = opt.lr

        if opt.train:
            logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S',
                                filename=os.path.join(opt.checkpoint, 'train.log'), level=logging.INFO)

    for epoch in range(start_epoch, opt.epochs):
        if opt.train:
            print('\nEpoch: %d | LR: %.8f' % (epoch, opt.lr_now))
            # Train for one epoch
            loss_total, loss_vis, loss_pose, loss_pose_seq, \
            train_p1, train_p2, train_p1_seq, train_p2_seq, train_acc = \
                train(opt, actions, train_dataloader, model, criterion_list, optimizer)

        # Evaluate
        p1, p2, p1_seq, p2_seq, acc, ap, tnr = evaluate(opt, actions, test_dataloader, model, criterion_list)

        # Save checkpoint
        if opt.previous_best_threshold[0] > p1_seq:
            opt.previous_best_threshold = (p1, acc)
            opt.previous_name = save_model(opt.previous_name, opt.checkpoint, epoch, p1, optimizer, model, 'best',
                                           extra=(opt.previous_best_threshold, opt.step))

        if (epoch + 1) % opt.snapshot == 0:
            save_model(None, opt.checkpoint, epoch, p1, optimizer, model, 'snapshot',
                       extra=(opt.previous_best_threshold, opt.step))

        if not opt.train:
            print('p1: %.2f, p1_seq: %.2f, acc: %.2f' % (p1, p1_seq, acc))
            break
        else:
            info = 'epoch: %d, lr: %.7f, loss_total: %.6f, loss_pose: %.6f, loss_vis: %.6f, loss_pose_seq: %.6f, ' \
                   'train_p1: %.2f, train_acc: %.4f, train_p1_seq: %.4f, ' \
                   'p1: %.2f, p1_seq: %.4f, acc: %.4f, ap: %.4f, tnr: %.4f' % (
                       epoch, opt.lr_now, loss_total, loss_pose, loss_vis, loss_pose_seq,
                       train_p1, train_acc, train_p1_seq, p1, p1_seq, acc, ap, tnr)
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
    else:
        model.eval()

    epoch_loss_3d = AccumLoss()
    epoch_loss_3d_vis = AccumLoss()
    epoch_loss_3d_pose = AccumLoss()
    epoch_loss_3d_pose_seq = AccumLoss()
    action_error_sum_pose = define_error_mpjpe_list(actions)
    action_error_sum_pose_seq = define_error_mpjpe_list(actions)
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
            batch_cam, gt_3d, gt_2d, vis, inputs_2d, inputs_scores, scale, bb_box, extra = data
            action, subject, cam_ind = extra

            if opt.ground_truth_input:
                inputs_2d = gt_2d
                inputs_scores[:] = 1

            [inputs_2d, inputs_scores, gt_2d, vis, gt_3d, batch_cam, scale, bb_box] = \
                get_variable(split, [inputs_2d, inputs_scores, gt_2d, vis, gt_3d, batch_cam, scale, bb_box])

        target = gt_3d.clone()
        target[:, :, 0] = 0

        if opt.in_dim == 2:
            pose_prediction, vis_prediction, pose_prediction_seq = model(inputs_2d)
        else:
            raise KeyError('Invalid input dimension')

        if split == 'train':
            N = inputs_2d.size(0)

            opt.step += 1
            if opt.step % opt.lr_decay == 0 or opt.step == 1:
                opt.lr_now = lr_decay(optimizer, opt.step, opt.lr, opt.lr_decay, opt.lr_gamma)

            loss_pose = criterion_list['pose'](pose_prediction, target)
            loss_vis = criterion_list['vis'](vis_prediction, vis)
            loss_pose_seq = criterion_list['pose_seq'](pose_prediction_seq, target)
            loss = loss_pose + loss_vis + loss_pose_seq

            optimizer.zero_grad()
            loss.backward()
            if opt.max_norm:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()

            epoch_loss_3d.update(loss.detach().cpu().numpy() * N, N)
            epoch_loss_3d_vis.update(loss_vis.detach().cpu().numpy() * N, N)
            epoch_loss_3d_pose.update(loss_pose.detach().cpu().numpy() * N, N)
            epoch_loss_3d_pose_seq.update(loss_pose_seq.detach().cpu().numpy() * N, N)
        elif split == 'test':
            action_error_sum_vis_binary_class_metrics = \
                test_calculation_binary_class_metrics(vis_prediction, vis, action, action_error_sum_vis_binary_class_metrics, opt.dataset)

        action_error_sum_vis_acc = test_calculation_acc(vis_prediction, vis, action, action_error_sum_vis_acc, opt.dataset)

        pose_prediction[:, :, 0, :] = 0
        pose_prediction_seq[:, :, 0, :] = 0

        target = rearrange(target, 'b f j c  -> (b f) j c', )
        pose_prediction = rearrange(pose_prediction, 'b f j c  -> (b f) j c', )
        pose_prediction_seq = rearrange(pose_prediction_seq, 'b f j c  -> (b f) j c', )
        action_error_sum_pose = test_calculation_mpjpe(pose_prediction, target, action, action_error_sum_pose, opt.dataset)
        action_error_sum_pose_seq = test_calculation_mpjpe(pose_prediction_seq, target, action, action_error_sum_pose_seq, opt.dataset)

    p1, p2 = print_error_mpjpe(opt.dataset, action_error_sum_pose, opt.train)
    p1_seq, p2_seq = print_error_mpjpe(opt.dataset, action_error_sum_pose_seq, opt.train)
    acc = print_acc(opt.dataset, action_error_sum_vis_acc, opt.train)

    if split == 'train':
        return epoch_loss_3d.avg, epoch_loss_3d_vis.avg, epoch_loss_3d_pose.avg, epoch_loss_3d_pose_seq.avg,\
               p1, p2, p1_seq, p2_seq, acc
    elif split == 'test':
        ap, tnr = print_binary_class_metrics(opt.dataset, action_error_sum_vis_binary_class_metrics, opt.train)
        return p1, p2, p1_seq, p2_seq, acc, ap, tnr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--self_supervision', action='store_true')
    parser.add_argument('--pretrained_graformer_init', action='store_true')
    parser.add_argument('--pretrained_graformer', type=str,
                        default='checkpoint/pretrained/graformer/small/best_83_5448.pth')
    parser.add_argument('--freeze_main_pipeline', action='store_true')

    opt = graformer_opts(parser).get_temporal_auxiliary_vis_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

    main(opt)
