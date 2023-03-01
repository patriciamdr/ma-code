import argparse
import os.path as path
import torch.nn as nn
import logging
from common.opt import graformer_opts
from torch.utils.data import DataLoader
from common.unrealcv_dataset import UnrealCvDataset
from common.load_data_unrealcv import Fusion as Fusion_unrealcv
from common.utils import *
from model.auxiliary_models import AuxiliaryModel
from model.block.utils import adj_mx_from_edges, edges_unrealcv
from tqdm import tqdm

src_mask = torch.tensor([[[True, True, True, True, True, True, True, True, True, True,
                           True, True, True, True, True]]]).cuda()

### Auxiliary Heatmap Model not up to date!!!
### Code not used currently


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
        edges = edges_unrealcv
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

        opt.n_joints = dataset.skeleton().num_joints()
    else:
        raise KeyError('Invalid dataset')

    actions = define_actions(opt.actions, opt.dataset)

    # Create model
    print("==> Creating model...")
    adj = adj_mx_from_edges(num_pts=opt.n_joints, edges=edges, sparse=False)
    model = AuxiliaryModel(adj=adj.cuda(), hid_dim=opt.dim_model, n_pts=opt.n_joints,
                           num_layers=opt.n_layer, n_head=opt.n_head, dropout=opt.dropout,
                           heatmap_res_out=opt.heatmap_res_out, heatmap_res_hid=opt.heatmap_res_hid).cuda()

    print("==> Total parameters: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))

    criterion_list = {'vis': nn.BCELoss().cuda(), 'heatmap': nn.MSELoss(reduction='none').cuda()}

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

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
            print("==> Loaded checkpoint (Epoch: {} | Lr: {})".format(start_epoch, opt.lr_now))

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
        print('\nEpoch: %d | LR: %.8f' % (epoch + 1, opt.lr_now))

        # Train for one epoch
        if opt.train:
            loss = train(opt, actions, train_dataloader, model, criterion_list, optimizer)

        # Evaluate
        vis, heatmap = evaluate(opt, actions, test_dataloader, model, criterion_list)

        p = vis + heatmap

        # Save checkpoint
        if opt.previous_best_threshold > p:
            opt.previous_best_threshold = p
            opt.previous_name = save_model(opt.previous_name, opt.checkpoint, epoch, p, optimizer, model, 'best',
                                           extra=(opt.previous_best_threshold, opt.step))

        if (epoch + 1) % opt.snapshot == 0:
            save_model(None, opt.checkpoint, epoch, p, optimizer, model, 'snapshot',
                       extra=(opt.previous_best_threshold, opt.step))

        if not opt.train:
            print('vis: %.2f, heatmap: %.2f' % (vis, heatmap))
            break
        else:
            logging.info('epoch: %d, lr: %.7f, loss: %.4f, vis: %.2f, heatmap: %.2f' % (
                epoch, opt.lr_now, loss, vis, heatmap))
            print('e: %d, lr: %.7f, loss: %.4f, vis: %.2f, heatmap: %.2f' % (
                epoch, opt.lr_now, loss, vis, heatmap))
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
    action_error_sum_class = define_error_mpjpe_list(actions)
    action_error_sum_reg = define_error_mpjpe_list(actions)

    for i, data in enumerate(tqdm(dataLoader)):
        batch_cam, gt_3d, gt_2d, vis, extra = data
        video_id, cam_ind, action = extra
        [gt_2d, gt_3d, batch_cam, vis] = get_variable(split, [gt_2d, gt_3d, batch_cam, vis])

        keypoints = gt_2d.clone()  # TODO: Load 2d keypoints of CPN detection for unreal_cv images

        vis_prediction, heatmaps_prediction = model(keypoints, src_mask)

        if split == 'train':
            N = gt_2d.size(0)

            opt.step += 1
            if opt.step % opt.lr_decay == 0 or opt.step == 1:
                opt.lr_now = lr_decay(optimizer, opt.step, opt.lr, opt.lr_decay, opt.lr_gamma)

            loss_heatmap = heatmap_loss(criterion_list['heatmap'], opt, heatmaps_prediction, gt_2d)
            loss_vis = criterion_list['vis'](vis_prediction, vis)
            loss = loss_vis + loss_heatmap

            optimizer.zero_grad()
            loss.backward()
            if opt.max_norm:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()

            epoch_loss_3d.update(loss.detach().cpu().numpy() * N, N)

        elif split == 'test':
            # out1[:, :, 0, :] = 0
            # outputs['pos'] = out1
            #
            # action_error_sum_class = test_calculation_acc(vis_prediction, vis, action, action_error_sum_class)
            # action_error_sum_reg = test_calculation_pdj(heatmaps_prediction, gt_2d, action, action_error_sum_reg)
            return

    if split == 'train':
        return epoch_loss_3d.avg
    elif split == 'test':
        # vis, heatmap = print_error_unreal(opt.dataset, action_error_sum_class, opt.train)
        # return vis, heatmap
        return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--heatmap_sigma', type=float, default=1.0)
    parser.add_argument('--heatmap_res_out', type=int, default=32)
    parser.add_argument('--heatmap_res_hid', type=int, default=32)

    opt = graformer_opts(parser).get_auxiliary_heatmap_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

    main(opt)
