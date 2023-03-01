import torch
import torch.nn as nn
from einops import rearrange

from model.block.chebconv import ChebConv
from model.block.spatial_module_encoder import SpatialGraphTransformer
from model.block.strided_transformer_encoder import Transformer as Transformer_reduce
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
        self.vis_embed_layer = ChebConv(in_c=1, out_c=args.graformer_vis_embed_dim, K=2)
        channel = (args.graformer_pose_embed_dim + args.graformer_vis_embed_dim) * args.n_joints
        self.Transformer_reduce = Transformer_reduce(len(args.stride_num), channel, args.d_hid,
                                                     length=args.frames, stride_num=args.stride_num,
                                                     entire_seq=True)

        self.fcn = nn.Sequential(
            nn.BatchNorm1d(channel, momentum=0.1),
            nn.Conv1d(channel, 3 * args.out_joints, kernel_size=1)
        )

        self.head = nn.Sequential(
            nn.BatchNorm1d(channel, momentum=0.1),
            nn.Conv1d(channel, 3 * args.out_joints, kernel_size=1)
        )

    def forward(self, x, y):
        B, F, J, C = x.shape

        x = rearrange(x, 'b f j c  -> (b f) j c', )
        x = self.Transformer(x)

        y = rearrange(y, 'b f j c  -> (b f) j c', )
        vis_embedding = self.vis_embed_layer(y, self.adj)
        x = torch.concat([x, vis_embedding], dim=2)

        x = rearrange(x, '(b f) j c -> b f (j c)', f=F)

        x, x_temp = self.Transformer_reduce(x)

        x_VTE = x_temp
        x_VTE = x_VTE.permute(0, 2, 1).contiguous()
        x_VTE = self.head(x_VTE)
        x_VTE = rearrange(x_VTE, 'b (j c) f -> b f j c', j=J).contiguous()

        x = x.permute(0, 2, 1).contiguous()
        x = self.fcn(x)
        x = rearrange(x, 'b (j c) f -> b f j c', j=J).contiguous()

        return x, x_VTE

    def freeze_spatial_module(self):
        for p in self.Transformer.parameters():
            p.requires_grad = False

    def set_spatial_module_eval_mode(self):
        self.Transformer.eval()
