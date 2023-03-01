import argparse
import os.path as path

import torch.nn as nn
import logging

from common.h36m_dataset import Human36mDataset
from common.opt import graformer_opts
from torch.utils.data import DataLoader
from common.unrealcv_dataset import UnrealCvDataset
from common.load_data_h36m import Fusion as Fusion_h36m
from common.load_data_unrealcv import Fusion as Fusion_unrealcv
from common.utils import *
from model.auxiliary_models import LinLayerNet
from tqdm import tqdm

from model.block.utils import edges_h36m


def main(opt):
    manualSeed = 0

    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.cuda.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)

    print('==> Using settings {}'.format(opt))

    print('==> Loading dataset...')
    dataset_dir = os.path.join(opt.root_path, opt.dataset)
    if opt.dataset == 'unrealcv':
        if opt.train:
            dataset = UnrealCvDataset(dataset_dir, train=True)
            train_data = Fusion_unrealcv(opt=opt, dataset=dataset, train=True)
            train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=opt.batch_size,
                                                           shuffle=True, num_workers=int(opt.workers),
                                                           pin_memory=True)

        dataset = UnrealCvDataset(dataset_dir)
        test_data = Fusion_unrealcv(opt=opt, dataset=dataset)
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
    model = LinLayerNet(n_pts=opt.n_joints, dropout=opt.dropout, in_dim=opt.in_dim).cuda()

    print("==> Total parameters: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))

    criterion = nn.BCELoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    if opt.resume or opt.evaluate:
        ckpt_path = (opt.resume if opt.resume else opt.evaluate)

        if path.isfile(ckpt_path):
            print("==> Loading checkpoint '{}'".format(ckpt_path))
            ckpt = torch.load(ckpt_path)
            start_epoch = ckpt['epoch']
            optimizer.load_state_dict(ckpt['optimizer'])
            opt.lr_now = optimizer.param_groups[0]['lr']
            opt.previous_best_threshold, opt.step = ckpt['extra']
            model.load_state_dict(ckpt['model_pos'])
            print("==> Loaded checkpoint (Epoch: {} | Error: {})".format(start_epoch, opt.previous_best_threshold))

            opt.checkpoint = path.dirname(ckpt_path)
        else:
            raise RuntimeError("==> No checkpoint found at '{}'".format(ckpt_path))
    else:
        start_epoch = 0
        opt.lr_now = opt.lr
        opt.step = 0

        if opt.train:
            logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S',
                                filename=os.path.join(opt.checkpoint, 'train.log'), level=logging.INFO)

    for epoch in range(start_epoch, opt.epochs):
        if opt.train:
            print('\nEpoch: %d | LR: %.8f' % (epoch, opt.lr_now))
            # Train for one epoch
            loss, train_acc = train(opt, actions, train_dataloader, model, criterion, optimizer)

        # Evaluate
        val_acc = evaluate(opt, actions, test_dataloader, model, criterion)

        # Save checkpoint
        if val_acc > opt.previous_best_threshold:
            opt.previous_best_threshold = val_acc
            opt.previous_name = save_model(opt.previous_name, opt.checkpoint, epoch, val_acc, optimizer, model, 'best',
                                           extra=(opt.previous_best_threshold, opt.step))

        if (epoch + 1) % opt.snapshot == 0:
            save_model(None, opt.checkpoint, epoch, val_acc, optimizer, model, 'snapshot',
                       extra=(opt.previous_best_threshold, opt.step))

        if not opt.train:
            print('acc: %.2f' % val_acc)
            break
        else:
            info = 'epoch: %d, lr: %.7f, loss: %.4f, train_acc: %.4f, val_acc: %.4f' % (epoch, opt.lr_now, loss,
                                                                                        train_acc, val_acc)
            logging.info(info)
            print(info)


def train(opt, actions, train_loader, model, criterion, optimizer):
    return step('train', opt, actions, train_loader, model, criterion, optimizer)


def evaluate(opt, actions, val_loader, model, criterion):
    with torch.no_grad():
        return step('test', opt, actions, val_loader, model, criterion)


def step(split, opt, actions, dataLoader, model, criterion, optimizer=None):
    if split == 'train':
        model.train()
    else:
        model.eval()

    epoch_loss_3d = AccumLoss()
    action_error_sum_vis = define_acc_list(actions)

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

            [inputs_2d, inputs_scores, vis, gt_3d, batch_cam, scale, bb_box] = \
                get_variable(split, [inputs_2d, inputs_scores, vis, gt_3d, batch_cam, scale, bb_box])

        target = gt_3d.clone()
        target[:, :, 0] = 0

        if opt.in_dim == 1:
            vis_prediction = model(inputs_scores)
        elif opt.in_dim == 2:
            vis_prediction = model(inputs_2d)
        elif opt.in_dim == 3:
            input = torch.cat((inputs_2d, inputs_scores), dim=3)
            vis_prediction = model(input)
        else:
            raise KeyError('Invalid input dimension')

        if opt.dataset == 'unrealcv' and vis_prediction.shape[2] == 17:
            # Ignore neck and head from model prediction because no ground truth available
            vis_prediction = torch.cat([vis_prediction[:, :, :9], vis_prediction[:, :, 11:]], dim=2)

        if split == 'train':
            N = inputs_scores.size(0)

            opt.step += 1
            if opt.step % opt.lr_decay == 0 or opt.step == 1:
                opt.lr_now = lr_decay(optimizer, opt.step, opt.lr, opt.lr_decay, opt.lr_gamma)
            loss = criterion(vis_prediction, vis)

            optimizer.zero_grad()
            loss.backward()
            if opt.max_norm:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()

            epoch_loss_3d.update(loss.detach().cpu().numpy() * N, N)

        action_error_sum_vis = test_calculation_acc(vis_prediction, vis, action, action_error_sum_vis, opt.dataset)

    acc = print_acc(opt.dataset, action_error_sum_vis, opt.train)

    if split == 'train':
        return epoch_loss_3d.avg, acc
    elif split == 'test':
        return acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    opt = graformer_opts(parser).get_vis_prediction_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

    main(opt)
