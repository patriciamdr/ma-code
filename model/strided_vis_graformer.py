import torch
import torch.nn as nn
from einops import rearrange
from model.block.spatial_module_encoder import SpatialVisGraphTransformer
from model.block.strided_transformer_encoder import Transformer as Transformer_reduce
from model.block.utils import adj_mx_from_edges, edges_unrealcv, edges_h36m


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()

        edges = edges_unrealcv if args.dataset == 'unrealcv' else edges_h36m

        adj = adj_mx_from_edges(num_pts=args.n_joints, edges=edges)

        self.Transformer = SpatialVisGraphTransformer(adj=adj.cuda(), hid_dim=args.graformer_hid_dim,
                                                      num_layers=args.graformer_depth, n_head=args.graformer_head,
                                                      dropout=args.graformer_dropout, pose_embed_dim=args.graformer_pose_embed_dim,
                                                      vis_embed_dim=args.graformer_vis_embed_dim, n_pts=args.n_joints)
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

        self.visibility_class_head = nn.Sequential(
            nn.Linear(args.graformer_vis_embed_dim, 1),
            nn.BatchNorm1d(args.n_joints)
        )

        c = args.graformer_pose_embed_dim + args.graformer_vis_embed_dim
        self.reduce = nn.Sequential(
            nn.Linear(c, args.graformer_pose_embed_dim),
            nn.ReLU()
        )

        self.extra_pose_hidden = args.extra_pose_hidden
        self.extra_vis_hidden = args.extra_vis_hidden
        if self.extra_pose_hidden:
            self.extra_pose_hidden = nn.Sequential(
                nn.Linear(args.graformer_pose_embed_dim, args.graformer_pose_embed_dim),
                nn.ReLU()
            )
        if self.extra_vis_hidden:
            self.extra_vis_hidden = nn.Sequential(
                nn.Linear(args.graformer_vis_embed_dim, args.graformer_vis_embed_dim),
                nn.ReLU()
            )

    def forward(self, x):
        B, F, J, C = x.shape

        x = rearrange(x, 'b f j c  -> (b f) j c', )
        pose_embedding, vis_embedding, atten_scores, atten_scores2 = self.Transformer(x)

        out_vis = self.visibility_class_head(vis_embedding)
        out_vis = rearrange(out_vis, '(b f) j c -> b f j c', f=F).contiguous()

        if self.extra_pose_hidden:
            pose_embedding = self.extra_pose_hidden(pose_embedding)
        if self.extra_vis_hidden:
            vis_embedding = self.extra_vis_hidden(vis_embedding)

        x = torch.cat([pose_embedding, vis_embedding], dim=2)
        x = self.reduce(x)
        x = rearrange(x.clone(), '(b f) j c -> b f (j c)', f=F).contiguous()

        x, atten_scores_full = self.Transformer_full(x)

        x_VTE = x
        x_VTE = x_VTE.permute(0, 2, 1).contiguous()
        x_VTE = self.head(x_VTE)
        x_VTE = rearrange(x_VTE, 'b (j c) f -> b f j c', j=J).contiguous()

        x, atten_scores_reduce = self.Transformer_reduce(x)

        x = x.permute(0, 2, 1).contiguous()
        x = self.fcn(x)
        x = rearrange(x, 'b (j c) f -> b f j c', j=J).contiguous()

        return x, x_VTE, out_vis, torch.sum(atten_scores, dim=0), torch.sum(atten_scores2, dim=0), atten_scores_full, atten_scores_reduce

    def freeze_spatial_module(self):
        for p in self.Transformer.parameters():
            p.requires_grad = False

    def set_spatial_module_eval_mode(self):
        self.Transformer.eval()
