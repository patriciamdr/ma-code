import argparse
import os
import math
import time
import torch


class opts():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def init(self):
        self.parser.add_argument('--layers', default=3, type=int)
        self.parser.add_argument('--use_2d_gt', action='store_true')
        self.parser.add_argument('--pos_weight', default=(6577949/(6577949 + 19937836)), help='for visibility model')
        self.parser.add_argument('--pose_weight_factor', default=1, type=int)
        self.parser.add_argument('--channel', default=256, type=int)
        self.parser.add_argument('--d_hid', default=512, type=int)
        self.parser.add_argument('--dataset', type=str, default='h36m')
        self.parser.add_argument('-k', '--keypoints', default='CPN', type=str)
        self.parser.add_argument('--data_augmentation', type=bool, default=True)
        self.parser.add_argument('--reverse_augmentation', type=bool, default=False)
        self.parser.add_argument('--test_augmentation', type=bool, default=True)
        self.parser.add_argument('--crop_uv', type=int, default=0)
        self.parser.add_argument('--root_path', type=str, default='dataset/')
        self.parser.add_argument('-a', '--actions', default='*', type=str)
        self.parser.add_argument('--downsample', default=1, type=int)
        self.parser.add_argument('--subset', default=1, type=float)
        self.parser.add_argument('-s', '--stride', default=1, type=int)
        self.parser.add_argument('--gpu', default='0', type=str, help='')
        self.parser.add_argument('--train', default=1)
        self.parser.add_argument('--test', action='store_true')
        self.parser.add_argument('--nepoch', type=int, default=50)
        self.parser.add_argument('--batch_size', type=int, default=256, help='can be changed depending on your machine')
        self.parser.add_argument('--lr', type=float, default=1e-3)
        self.parser.add_argument('--spatial_module_lr', type=float, default=1e-4)
        self.parser.add_argument('--lr_refine', type=float, default=1e-5)
        self.parser.add_argument('--lr_decay_large', type=float, default=0.5)
        self.parser.add_argument('--save_ckpt_intervall', type=int, default=5)
        self.parser.add_argument('--large_decay_epoch', type=int, default=5)
        self.parser.add_argument('--workers', type=int, default=8)
        self.parser.add_argument('-lrd', '--lr_decay', default=0.95, type=float)
        self.parser.add_argument('--frames', type=int, default=351)
        self.parser.add_argument('--refine', action='store_true')
        self.parser.add_argument('--reload', action='store_true')
        self.parser.add_argument('--refine_reload', action='store_true')
        self.parser.add_argument('--checkpoint', type=str, default='')
        self.parser.add_argument('--previous_dir', type=str, default='')
        self.parser.add_argument('--n_joints', type=int, default=17)
        self.parser.add_argument('--out_joints', type=int, default=17)
        self.parser.add_argument('--train_out_all', type=int, default=1)
        self.parser.add_argument('--test_out_all', type=int, default=0)
        self.parser.add_argument('--in_channels', type=int, default=2)
        self.parser.add_argument('--out_channels', type=int, default=3)
        self.parser.add_argument('--resume', action='store_true', help='resume training')
        self.parser.add_argument('--occlusion_augmentation_train', action='store_true',
                                 help='apply occlusion-based data augmentation during training')
        self.parser.add_argument('--occlusion_augmentation_test', action='store_true',
                                 help='apply occlusion-based data augmentation during testing')
        self.parser.add_argument('--non_uniform_joint_prob', action='store_true',
                                 help='Occlude joints with non-uniform probability based on joint importance analysis')
        self.parser.add_argument('--occluded_joint', type=int, default=None,
                                 help='index of joint to occlude ∈ [0, 17['
                                      'if not specified, choose randomly per frame')
        self.parser.add_argument('--num_occluded_f', type=int, default=None,
                                 help='number of frames to occlude ∈ [0, opt.frames['
                                      'if not specified, choose random number of frames')
        self.parser.add_argument('--num_occluded_j', type=int, default=1,
                                 help='number of joints to occlude ∈ [0, opt.n_joints[')
        self.parser.add_argument('--consecutive_frames', action='store_true', dest='consecutive_frames',
                                 help='apply occlusion to randomly selected consecutive subset of frames'
                                      'instead of randomly selected frames that may not be consecutive')
        self.parser.add_argument('--subset_size', type=int, default=6,
                                 help='Number of consecutive frames to occlude,'
                                      'only considered when consecutive_frames is true.'
                                      'Occlude (subset_size) consecutive frames.'
                                      'If num_occluded_f is None random number of subsets are occluded per sequence.'
                                      'If num_occluded_f is specified then (num_occluded_f // subset_size) number of subsets are occluded per sequence')
        self.parser.add_argument('-previous_best_threshold', type=float, default=math.inf)
        self.parser.add_argument('-previous_name', type=str, default='')
        self.parser.add_argument('-previous_refine_name', type=str, default='')
        self.parser.add_argument('--spatial_module', action='store_true')
        self.parser.add_argument('--use_visibility', action='store_true')
        self.parser.add_argument('--extra_vis_hidden', action='store_true')
        self.parser.add_argument('--extra_pose_hidden', action='store_true')
        self.parser.add_argument('--use_visibility_gt', action='store_true')
        self.parser.add_argument('--freeze_spatial_module', action='store_true')
        self.parser.add_argument('--freeze_trans_module', action='store_true')
        self.parser.add_argument('--set_spatial_module_eval_mode', action='store_true', help='Only relevant if spatial module is frozen')
        self.parser.add_argument('--graformer_head', type=int, default=4)
        self.parser.add_argument('--graformer_hid_dim', type=int, default=64)
        self.parser.add_argument('--graformer_pose_embed_dim', type=int, default=16)
        self.parser.add_argument('--graformer_vis_embed_dim', type=int, default=8)
        self.parser.add_argument('--graformer_dropout', type=float, default=0.1)
        self.parser.add_argument('--graformer_depth', type=int, default=2)
        self.parser.add_argument('--pretrained_spatial_module_init', action='store_true')
        self.parser.add_argument('--pretrained_spatial_module_dir', type=str,
                                 default='checkpoint/pretrained/graformer/small')
        self.parser.add_argument('--pretrained_spatial_module', type=str, default='best_83_5448.pth',
                                 help='file to use in pretrained_spatial_module_dir')
        self.parser.add_argument('--scale_factor', type=int, default=0, help='for unrealcv dataset')
        self.parser.add_argument('--rotation_factor', type=int, default=0, help='for unrealcv dataset')
        self.parser.add_argument('--error_thr', type=float, default=0.2, help='for self-supervised visibility targets')

    def parse(self):
        self.init()
        self.opt = self.parser.parse_args()

        # Check invalid configuration
        if self.opt.use_visibility_gt and self.opt.use_visibility:
            print('Invalid flags: --use_visibility and --use_visibility_gt cannot be set at the same time')
            exit()

        if self.opt.test:
            self.opt.train = 0

        self.opt.pad = (self.opt.frames - 1) // 2

        if self.opt.dataset == 'unrealcv':
            self.opt.cropped_image_size = (1000, 1000)

        stride_num = {
            '27': [3, 3, 3],
            '81': [9, 3, 3],
            '243': [3, 9, 9],
            '351': [3, 9, 13],
        }

        if str(self.opt.frames) in stride_num:
            self.opt.stride_num = stride_num[str(self.opt.frames)]
        else:
            self.opt.stride_num = None
            print('no stride_num')
            exit()

        self.opt.subjects_train = 'S1,S5,S6,S7,S8'
        self.opt.subjects_test = 'S9,S11'

        if self.opt.train:
            logtime = time.strftime('%m%d_%H%M_%S_')

            self.opt.checkpoint = 'checkpoint/' + logtime + '%d' % (self.opt.pad * 2 + 1) + \
                                  '%s' % ('' if self.opt.refine else '_no')

            if not os.path.exists(self.opt.checkpoint):
                os.makedirs(self.opt.checkpoint)

        if self.opt.train:
            args = dict((name, getattr(self.opt, name)) for name in dir(self.opt)
                        if not name.startswith('_'))

            file_name = os.path.join(self.opt.checkpoint, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('==> Args:\n')
                for k, v in sorted(args.items()):
                    opt_file.write('  %s: %s\n' % (str(k), str(v)))
                opt_file.write('==> Args:\n')

        if self.opt.non_uniform_joint_prob:
            # Assign different occlusion probabilities per joint based on joint importance analysis
            # self.opt.joint_probability = \
            #     torch.tensor(
            #         [10 / 85, 5 / 85, 3 / 85, 2 / 85, 5 / 85, 3 / 85, 2 / 85, 6 / 85, 5 / 85, 10 / 85, 6 / 85, 6 / 85,
            #          5 / 85, 3 / 85, 6 / 85, 5 / 85, 3 / 85])
            self.opt.joint_probability = \
                torch.tensor(
                    [11.9, 2.6, 1.6, 1, 2.6, 1.6, 1, 2.6, 3.1, 4.7, 4.7, 3.1, 3.1, 4, 3.1, 3.1, 4])
                    # [12, 6, 2, 1, 6, 2, 1, 6, 8, 10, 10, 8, 8, 4, 8, 8, 4])
        else:
            self.opt.joint_probability = torch.ones(self.opt.out_joints)

        return self.opt


class graformer_opts():
    def __init__(self, parser):
        self.parser = parser

    def init(self):
        # General arguments
        self.parser.add_argument('--gpu', default='0', type=str, help='')
        self.parser.add_argument('--root_path', type=str, default='dataset/')
        self.parser.add_argument('--frames', type=int, default=1)
        self.parser.add_argument('--train', default=1)
        self.parser.add_argument('--n_joints', type=int, default=17)
        self.parser.add_argument('-d', '--dataset', type=str, metavar='NAME', help='target dataset')
        self.parser.add_argument('-k', '--keypoints', default='CPN', type=str, metavar='NAME',
                                 help='2D detections to use')
        self.parser.add_argument('-a', '--actions', default='*', type=str, metavar='LIST',
                                 help='actions to train/test on, separated by comma, or * for all')
        self.parser.add_argument('--test', default='', type=str, metavar='FILENAME',
                                 help='checkpoint to evaluate (file name)')
        self.parser.add_argument('-r', '--resume', default='', type=str, metavar='FILENAME',
                                 help='checkpoint to resume (file name)')
        self.parser.add_argument('--evaluate', default='', type=str, metavar='FILENAME',
                                 help='checkpoint to evaluate (file name)')
        self.parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                                 help='checkpoint directory')
        self.parser.add_argument('--snapshot', default=5, type=int, help='save models for every #snapshot epochs')
        self.parser.add_argument('--n_head', type=int, default=4, help='num head')
        self.parser.add_argument('--dim_model', type=int, default=64, help='dim model')
        self.parser.add_argument('--n_layer', type=int, default=2, help='num layer')
        self.parser.add_argument('--dropout', default=0.1, type=float, help='dropout rate')
        self.parser.add_argument('--max_norm', default=None, type=float, help='max gradient norm')
        self.parser.add_argument('-b', '--batch_size', default=64, type=int, metavar='N',
                                 help='batch size in terms of predicted frames')
        self.parser.add_argument('-e', '--epochs', default=100, type=int, metavar='N', help='number of training epochs')
        self.parser.add_argument('--workers', default=1, type=int, metavar='N',
                                 help='num of workers for data loading')
        self.parser.add_argument('--lr', default=1.0e-3, type=float, metavar='LR', help='initial learning rate')
        self.parser.add_argument('--lr_vis', default=1.0e-3, type=float, metavar='LR', help='initial learning rate for visibility branch')
        self.parser.add_argument('--lr_decay', type=int, default=50000, help='num of steps of learning rate decay')
        self.parser.add_argument('--lr_gamma', type=float, default=0.9, help='gamma of learning rate decay')
        self.parser.add_argument('--pose_embed_dim', type=int, default=16)
        self.parser.add_argument('--vis_embed_dim', type=int, default=8)
        self.parser.add_argument('--ground_truth_input', action='store_true',
                                 help='use 2D ground truth poses instead of keypoints as input')
        self.parser.add_argument('--error_thr', type=float, default=0.5, help='for self-supervised visibility targets')
        self.parser.add_argument('--occlusion_augmentation_train', action='store_true',
                                 help='apply occlusion-based data augmentation during training')
        self.parser.add_argument('--occlusion_augmentation_test', action='store_true',
                                 help='apply occlusion-based data augmentation during testing')
        self.parser.add_argument('--scale_factor', type=int, default=0, help='for unrealcv dataset')
        self.parser.add_argument('--rotation_factor', type=int, default=0, help='for unrealcv dataset')
        self.parser.add_argument('--previous_name', type=str, default='')
        self.parser.add_argument('--d_hid', default=512, type=int)
        self.parser.add_argument('--in_dim', type=int, default=2, help='input dim'
                                                                       'dim = 1: score as input'
                                                                       'dim = 2: pose as input'
                                                                       'dim = 3: score and pose as input')

    def parse(self):
        self.opt.crop_uv = 0
        self.opt.downsample = 1
        self.opt.stride = 1
        self.opt.subset = 1
        self.opt.data_augmentation = True
        self.opt.reverse_augmentation = False
        self.opt.test_augmentation = True
        self.opt.train_out_all = 1
        self.opt.test_out_all = 1

        if self.opt.dataset == 'unrealcv':
            self.opt.cropped_image_size = (1000, 1000)

        if self.opt.evaluate:
            self.opt.train = 0

            # Check invalid configuration
            if self.opt.resume:
                print('Invalid flags: --resume and --evaluate cannot be set at the same time')
                exit()

        if self.opt.train:
            logtime = time.strftime('%m%d_%H%M_%S')

            self.opt.checkpoint = os.path.join(self.opt.checkpoint, logtime)

            if not os.path.exists(self.opt.checkpoint):
                os.makedirs(self.opt.checkpoint)
                print('==> Making checkpoint dir: {}'.format(self.opt.checkpoint))

            args = dict((name, getattr(self.opt, name)) for name in dir(self.opt)
                        if not name.startswith('_'))

            file_name = os.path.join(self.opt.checkpoint, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('==> Args:\n')
                for k, v in sorted(args.items()):
                    opt_file.write('  %s: %s\n' % (str(k), str(v)))
                opt_file.write('==> Args:\n')

        self.opt.pad = (self.opt.frames - 1) // 2

        self.opt.subjects_train = 'S1,S5,S6,S7,S8'
        self.opt.subjects_test = 'S9,S11'

        return self.opt

    def get_temporal_auxiliary_vis_args(self):
        self.init()
        self.opt = self.parser.parse_args()

        model_name = 'TemporalGraFormerVisSelfSupervision' if self.opt.self_supervision else 'TemporalGraFormerVis'

        if self.opt.train:
            self.opt.checkpoint = os.path.join(
                self.opt.checkpoint, model_name,
                'error-thr-%s' % self.opt.error_thr,
                self.opt.keypoints,
                'in-dim-%s' % self.opt.in_dim,
                'head-%s' % self.opt.n_head + '_layers-%s' % self.opt.n_layer + '_dim-%s' % self.opt.dim_model + '_last-dim-%s' % self.opt.pose_embed_dim,
                'lr-step-%s' % self.opt.lr_decay + '_lr-gamma-%s' % self.opt.lr_gamma + '_drop-%s' % self.opt.dropout)

        if self.opt.self_supervision:
            # Check invalid configuration
            if self.opt.dataset is None:
                self.opt.dataset = 'h36m'
            elif self.opt.dataset == 'unrealcv':
                if not self.opt.evaluate:
                    print('Invalid flags: graformer vis self supervision model is ment to be trained on data without visibility flags'
                          'Use graformer vis model for UnrealCV since visibility flags are available')
                    exit()
        else:
            # Check invalid configuration
            if self.opt.dataset is None:
                self.opt.dataset = 'unrealcv'
            elif self.opt.dataset == 'h36m':
                if not self.opt.evaluate:
                    print('Invalid flags: graformer vis model cannot be trained on h36m, only tested')
                    exit()

        self.opt.previous_best_threshold = 0

        return self.parse()

    def get_graformer_vis_args(self):
        self.init()
        self.opt = self.parser.parse_args()

        model_name = 'GraFormerVisSelfSupervision' if self.opt.self_supervision else 'GraFormerVis'

        if self.opt.train:
            self.opt.checkpoint = os.path.join(
                self.opt.checkpoint, model_name,
                'error-thr-%s' % self.opt.error_thr,
                self.opt.keypoints,
                'in-dim-%s' % self.opt.in_dim,
                'head-%s' % self.opt.n_head + '_layers-%s' % self.opt.n_layer + '_dim-%s' % self.opt.dim_model + '_last-dim-%s' % self.opt.pose_embed_dim,
                'lr-step-%s' % self.opt.lr_decay + '_lr-gamma-%s' % self.opt.lr_gamma + '_drop-%s' % self.opt.dropout)

        if self.opt.self_supervision:
            # Check invalid configuration
            if self.opt.dataset is None:
                self.opt.dataset = 'h36m'
            elif self.opt.dataset == 'unrealcv':
                if not self.opt.evaluate:
                    print('Invalid flags: graformer vis self supervision model is ment to be trained on data without visibility flags'
                          'Use graformer vis model for UnrealCV since visibility flags are available')
                    exit()
        else:
            # Check invalid configuration
            if self.opt.dataset is None:
                self.opt.dataset = 'unrealcv'
            elif self.opt.dataset == 'h36m':
                if not self.opt.evaluate:
                    print('Invalid flags: graformer vis model cannot be trained on h36m, only tested')
                    exit()

        self.opt.previous_best_threshold = 0

        return self.parse()

    def get_graformer_args(self):
        self.init()
        self.opt = self.parser.parse_args()

        if self.opt.train:
            self.opt.checkpoint = os.path.join(
                self.opt.checkpoint, 'GraFormer', self.opt.keypoints,
                'in-dim-%s' % self.opt.in_dim,
                'head-%s' % self.opt.n_head + '_layers-%s' % self.opt.n_layer + '_dim-%s' % self.opt.dim_model + '_last-dim-%s' % self.opt.pose_embed_dim,
                'lr-step-%s' % self.opt.lr_decay + '_lr-gamma-%s' % self.opt.lr_gamma + '_drop-%s' % self.opt.dropout)

        if self.opt.dataset is None:
            self.opt.dataset = 'h36m'

        self.opt.previous_best_threshold = math.inf

        return self.parse()

    def get_auxiliary_vis_args(self):
        self.init()
        self.opt = self.parser.parse_args()

        model_name = 'AuxiliaryVisSelfSupervisionModel' if self.opt.self_supervision else 'AuxiliaryVisModel'
        sub_dir = 'linlayers_vis_branch' if self.opt.lin_layers else 'big_graformer_vis_branch'

        if self.opt.train:
            self.opt.checkpoint = os.path.join(
                self.opt.checkpoint, model_name,
                sub_dir,
                'error-thr-%s' % self.opt.error_thr,
                self.opt.keypoints,
                'in-dim-%s' % self.opt.in_dim,
                'head-%s' % self.opt.n_head + '_layers-%s' % self.opt.n_layer + '_dim-%s' % self.opt.dim_model,
                'lr-step-%s' % self.opt.lr_decay + '_lr-gamma-%s' % self.opt.lr_gamma + '_drop-%s' % self.opt.dropout,
                'weight-factor-%s' % self.opt.pose_weight_factor)

        if self.opt.self_supervision:
            # Check invalid configuration
            if self.opt.dataset is None:
                self.opt.dataset = 'h36m'
            elif self.opt.dataset == 'unrealcv':
                if not self.opt.evaluate:
                    print('Invalid flags: auxiliary vis self supervision model is ment to be trained on data without visibility flags'
                          'Use auxiliary vis model for UnrealCV since visibility flags are available')
                    exit()
        else:
            # Check invalid configuration
            if self.opt.dataset is None:
                self.opt.dataset = 'unrealcv'
            elif self.opt.dataset == 'h36m':
                if not self.opt.evaluate:
                    print('Invalid flags: auxiliary vis model cannot be trained on h36m, only tested')
                    exit()

        self.opt.previous_best_threshold = (math.inf, 0)

        return self.parse()

    def get_vis_prediction_args(self):
        self.init()
        self.opt = self.parser.parse_args()

        if self.opt.train:
            self.opt.checkpoint = os.path.join(
                self.opt.checkpoint, 'VisPrediction', self.opt.keypoints,
                'in-dim-%s' % self.opt.in_dim,
                'head-%s' % self.opt.n_head + '_layers-%s' % self.opt.n_layer + '_dim-%s' % self.opt.dim_model,
                'lr-step-%s' % self.opt.lr_decay + '_lr-gamma-%s' % self.opt.lr_gamma + '_drop-%s' % self.opt.dropout)

        # Check invalid configuration
        if self.opt.dataset is None:
            self.opt.dataset = 'unrealcv'

        self.opt.previous_best_threshold = 0

        return self.parse()

    def get_auxiliary_heatmap_args(self):
        self.init()
        self.opt = self.parser.parse_args()

        if self.opt.train:
            self.opt.checkpoint = os.path.join(
                self.opt.checkpoint, 'AuxiliaryHeatmapModel', self.opt.keypoints,
                'heatmap-res-hid-%s' % self.opt.heatmap_res_hid + '_heatmap-res-out-%s' % self.opt.heatmap_res_out,
                'sigma-%s' % self.opt.heatmap_sigma,
                'head-%s' % self.opt.n_head + '_layers-%s' % self.opt.n_layer + '_dim-%s' % self.opt.dim_model,
                'lr-step-%s' % self.opt.lr_decay + '_lr-gamma-%s' % self.opt.lr_gamma + '_drop-%s' % self.opt.dropout)

        # Check invalid configuration
        if self.opt.dataset is None:
            self.opt.dataset = 'h36m'

        return self.parse()
