import torch
import torch.nn as nn
from einops import rearrange

from model.block.chebconv import ChebConv
from model.block.spatial_module_encoder import SpatialGraphTransformer
from model.block.strided_transformer_encoder import Transformer as Transformer_reduce
from model.block.vanilla_transformer_encoder import Transformer as Transformer_full
from model.block.utils import adj_mx_from_edges, edges_unrealcv, edges_h36m


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()

        edges = edges_unrealcv if args.dataset == 'unrealcv' else edges_h36m

        self.adj = adj_mx_from_edges(num_pts=args.n_joints, edges=edges).cuda()

        self.Transformer = SpatialGraphTransformer(adj=self.adj, hid_dim=args.graformer_hid_dim,
                                                   num_layers=args.graformer_depth, n_head=args.graformer_head,
                                                   dropout=args.graformer_dropout, out_dim=args.graformer_pose_embed_dim,
                                                   n_pts=args.n_joints)
        channel = args.graformer_pose_embed_dim * args.n_joints
        self.Transformer_full = Transformer_reduce(1, channel, args.d_hid, length=args.frames, stride_num=[1])
        self.Transformer_reduce = Transformer_reduce(len(args.stride_num), channel, args.d_hid,
                                                     length=args.frames, stride_num=args.stride_num)

        self.fcn = nn.Sequential(
            nn.BatchNorm1d(channel, momentum=0.1),
            nn.Conv1d(channel, 3 * args.out_joints, kernel_size=1)
        )

        self.head = nn.Sequential(
            nn.BatchNorm1d(channel, momentum=0.1),
            nn.Conv1d(channel, 3 * args.out_joints, kernel_size=1)
        )

    def forward(self, x):
        B, F, J, C = x.shape

        x = rearrange(x, 'b f j c  -> (b f) j c', )
        x, atten_scores, atten_scores2 = self.Transformer(x)
        x = rearrange(x, '(b f) j c -> b f (j c)', f=F)

        x, atten_scores_full = self.Transformer_full(x)


        x_VTE = x
        x_VTE = x_VTE.permute(0, 2, 1).contiguous()
        x_VTE = self.head(x_VTE)
        x_VTE = rearrange(x_VTE, 'b (j c) f -> b f j c', j=J).contiguous()

        x, atten_scores_reduce = self.Transformer_reduce(x)

        x = x.permute(0, 2, 1).contiguous()
        x = self.fcn(x)
        x = rearrange(x, 'b (j c) f -> b f j c', j=J).contiguous()

        return x, x_VTE, torch.sum(atten_scores, dim=0), torch.sum(atten_scores2, dim=0), atten_scores_full, atten_scores_reduce

    def freeze(self):
        self.freeze_spatial_module()
        for p in self.Transformer_full.parameters():
            p.requires_grad = False
        for p in self.Transformer_reduce.parameters():
            p.requires_grad = False
        for p in self.fcn.parameters():
            p.requires_grad = False
        for p in self.head.parameters():
            p.requires_grad = False

    def freeze_spatial_module(self):
        for p in self.Transformer.parameters():
            p.requires_grad = False

    def set_spatial_module_eval_mode(self):
        self.Transformer.eval()
