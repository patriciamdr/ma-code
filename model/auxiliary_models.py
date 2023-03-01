import copy
import math

import torch
from einops import rearrange
from torch import nn

from model.block.chebconv import ChebConv, _ResChebGC
from model.block.spatial_module_encoder import MultiHeadedAttention, GraphNet, GraAttenLayer
from model.block.strided_transformer_encoder import Transformer as Transformer_reduce


class LinLayerNet(nn.Module):
    def __init__(self, in_dim=1, hid_dim=64, num_classes=1, n_pts=17, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim, num_classes)
        )

        self.vis_class_head = nn.Sequential(
            nn.BatchNorm1d(n_pts),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, f = x.shape[0], x.shape[1]
        x = rearrange(x, 'b f j c  -> (b f) j c', )
        x = self.net(x)
        x = self.vis_class_head(x)
        x = rearrange(x, '(b f) j c -> b f j c', f=f).contiguous()
        return x


class GraFormer(nn.Module):
    def __init__(self, adj, in_dim=2, hid_dim=64, last_dim=16, num_layers=2,
                 n_head=8, dropout=0.1, n_pts=17):
        super(GraFormer, self).__init__()
        self.n_layers = num_layers
        self.adj = adj
        self.n_pts = n_pts
        self.src_mask = torch.full((self.n_pts, ), True).cuda()

        self.gconv_input = ChebConv(in_c=in_dim, out_c=hid_dim, K=2)
        _gconv_layers = []
        _attention_layer = []

        self.dim_model = hid_dim
        c = copy.deepcopy
        attn = MultiHeadedAttention(n_head, self.dim_model)
        gcn = GraphNet(in_features=self.dim_model, out_features=self.dim_model, n_pts=n_pts)

        for i in range(num_layers):
            _gconv_layers.append(_ResChebGC(adj=self.adj, input_dim=hid_dim, output_dim=hid_dim,
                                            hid_dim=hid_dim, p_dropout=0.1))
            _attention_layer.append(GraAttenLayer(self.dim_model, c(attn), c(gcn), dropout))

        self.gconv_layers = nn.ModuleList(_gconv_layers)
        self.atten_layers = nn.ModuleList(_attention_layer)

        self.last_gconv_layer = ChebConv(in_c=hid_dim, out_c=last_dim, K=2)
        # self.activation = nn.ReLU()

        channel = last_dim * n_pts
        self.head = nn.Sequential(
            nn.BatchNorm1d(channel, momentum=0.1),
            nn.Conv1d(channel, 3 * n_pts, kernel_size=1)
        )
        # self.head = ChebConv(in_c=last_dim, out_c=3, K=2)

    def forward(self, x, mask=None):
        if mask is None:
            mask = self.src_mask

        b, f = x.shape[0], x.shape[1]
        x = rearrange(x, 'b f j c  -> (b f) j c', )

        x = self.gconv_input(x, self.adj)
        for i in range(self.n_layers):
            x = self.atten_layers[i](x, mask)
            x = self.gconv_layers[i](x)

        x = self.last_gconv_layer(x, self.adj)
        # x = self.activation(x)
        x = rearrange(x, '(b f) j c -> b (j c) f', f=f)
        x = self.head(x)
        out1 = rearrange(x, 'b (j c) f -> b f j c', j=self.n_pts).contiguous()
        # out1 = self.head(x, self.adj)
        # out1 = rearrange(out1, '(b f) j c -> b f j c', f=f).contiguous()
        return out1


class AuxiliaryVisModel(nn.Module):
    def __init__(self, adj, in_dim=2, hid_dim=64, num_layers=2, n_head=8, dropout=0.1, n_pts=17,
                 pose_embed_dim=16, vis_embed_dim=8, num_classes=1, lin_layers=False):
        super(AuxiliaryVisModel, self).__init__()
        self.n_layers = num_layers
        self.adj = adj
        self.n_pts = n_pts
        self.src_mask = torch.full((self.n_pts, ), True).cuda()

        self.gconv_input = ChebConv(in_c=in_dim, out_c=hid_dim, K=2)
        _gconv_layers = []
        _attention_layer = []

        self.dim_model = hid_dim
        c = copy.deepcopy
        attn = MultiHeadedAttention(n_head, self.dim_model)
        gcn = GraphNet(in_features=self.dim_model, out_features=self.dim_model, n_pts=n_pts)

        for i in range(num_layers):
            _gconv_layers.append(_ResChebGC(adj=self.adj, input_dim=hid_dim, output_dim=hid_dim,
                                            hid_dim=hid_dim, p_dropout=0.1))
            _attention_layer.append(GraAttenLayer(self.dim_model, c(attn), c(gcn), dropout))

        self.gconv_layers = nn.ModuleList(_gconv_layers)
        self.atten_layers = nn.ModuleList(_attention_layer)

        self.lin_layers = lin_layers
        if lin_layers:
            self.vis_branch = nn.Sequential(
                nn.Linear(self.dim_model, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, vis_embed_dim),
                nn.ReLU(),
            )
        else:
            _gconv_layers_vis = []
            _attention_layer_vis = []

            for i in range(num_layers):
                _gconv_layers_vis.append(_ResChebGC(adj=self.adj, input_dim=hid_dim, output_dim=hid_dim,
                                                    hid_dim=hid_dim, p_dropout=0.1))
                _attention_layer_vis.append(GraAttenLayer(self.dim_model, c(attn), c(gcn), dropout))

            self.gconv_layers_vis = nn.ModuleList(_gconv_layers_vis)
            self.atten_layers_vis = nn.ModuleList(_attention_layer_vis)

            self.last_gconv_layer_vis = ChebConv(in_c=self.dim_model, out_c=vis_embed_dim, K=2)

        self.visibility_class_head = nn.Sequential(
            nn.Linear(vis_embed_dim, num_classes),
            nn.BatchNorm1d(self.n_pts),
            # nn.Sigmoid()
        )

        self.last_gconv_layer = ChebConv(in_c=self.dim_model, out_c=pose_embed_dim, K=2)
        channel = pose_embed_dim * n_pts
        self.head = nn.Sequential(
            nn.BatchNorm1d(channel, momentum=0.1),
            nn.Conv1d(channel, 3 * n_pts, kernel_size=1)
        )

    def set_main_pipeline_eval_mode(self):
        self.gconv_input.eval()
        self.gconv_layers.eval()
        self.atten_layers.eval()
        self.last_gconv_layer.eval()
        self.head.eval()

    def freeze_main_pipeline(self):
        for p in self.gconv_input.parameters():
            p.requires_grad = False
        for p in self.gconv_layers.parameters():
            p.requires_grad = False
        for p in self.atten_layers.parameters():
            p.requires_grad = False
        for p in self.last_gconv_layer.parameters():
            p.requires_grad = False
        for p in self.head.parameters():
            p.requires_grad = False

    def forward(self, x, mask=None):
        if mask is None:
            mask = self.src_mask

        b, f = x.shape[0], x.shape[1]
        x = rearrange(x, 'b f j c  -> (b f) j c', )

        x = self.gconv_input(x, self.adj)
        for i in range(self.n_layers):
            x = self.atten_layers[i](x, mask)
            x = self.gconv_layers[i](x)

        out_pose = self.last_gconv_layer(x, self.adj)
        out_pose = rearrange(out_pose, '(b f) j c -> b (j c) f', f=f)
        out_pose = self.head(out_pose)
        out_pose = rearrange(out_pose, 'b (j c) f -> b f j c', j=self.n_pts).contiguous()

        if self.lin_layers:
            out_vis = self.vis_branch(x)
        else:
            for i in range(self.n_layers):
                x = self.atten_layers_vis[i](x, mask)
                x = self.gconv_layers_vis[i](x)

            out_vis = self.last_gconv_layer_vis(x, self.adj)

        out_vis = self.visibility_class_head(out_vis)
        out_vis = rearrange(out_vis, '(b f) j c -> b f j c', f=f).contiguous()

        return out_pose, out_vis


class GraFormerVis(nn.Module):
    def __init__(self, adj, in_dim=2, hid_dim=64, num_layers=2, n_head=8, dropout=0.1, n_pts=17,
                 vis_embed_dim=8, num_classes=1):
        super(GraFormerVis, self).__init__()
        self.n_layers = num_layers
        self.adj = adj
        self.n_pts = n_pts
        self.src_mask = torch.full((self.n_pts, ), True).cuda()

        self.gconv_input = ChebConv(in_c=in_dim, out_c=hid_dim, K=2)
        _gconv_layers = []
        _attention_layer = []

        self.dim_model = hid_dim
        c = copy.deepcopy
        attn = MultiHeadedAttention(n_head, self.dim_model)
        gcn = GraphNet(in_features=self.dim_model, out_features=self.dim_model, n_pts=n_pts)

        for i in range(num_layers):
            _gconv_layers.append(_ResChebGC(adj=self.adj, input_dim=hid_dim, output_dim=hid_dim,
                                            hid_dim=hid_dim, p_dropout=0.1))
            _attention_layer.append(GraAttenLayer(self.dim_model, c(attn), c(gcn), dropout))

        self.gconv_layers_vis = nn.ModuleList(_gconv_layers)
        self.atten_layers_vis = nn.ModuleList(_attention_layer)

        self.last_gconv_layer_vis = ChebConv(in_c=self.dim_model, out_c=vis_embed_dim, K=2)
        self.visibility_class_head = nn.Sequential(
            nn.Linear(vis_embed_dim, num_classes),
            nn.BatchNorm1d(self.n_pts),
            # nn.Sigmoid()
        )

    def forward(self, x, mask=None):
        if mask is None:
            mask = self.src_mask

        b, f = x.shape[0], x.shape[1]
        x = rearrange(x, 'b f j c  -> (b f) j c', )

        x = self.gconv_input(x, self.adj)
        for i in range(self.n_layers):
            x = self.atten_layers_vis[i](x, mask)
            x = self.gconv_layers_vis[i](x)

        out_vis = self.last_gconv_layer_vis(x, self.adj)
        out_vis = self.visibility_class_head(out_vis)
        out_vis = rearrange(out_vis, '(b f) j c -> b f j c', f=f).contiguous()

        return out_vis


class AuxiliaryVisRefineModel(nn.Module):
    def __init__(self, adj, in_dim=2, hid_dim=64, num_layers=2, n_head=8, dropout=0.1, n_pts=17,
                 pose_embed_dim=16, vis_embed_dim=8, num_classes=1):
        super(AuxiliaryVisRefineModel, self).__init__()
        self.n_layers = num_layers
        self.adj = adj
        self.n_pts = n_pts
        self.src_mask = torch.full((self.n_pts, ), True).cuda()

        self.gconv_input = ChebConv(in_c=in_dim, out_c=hid_dim, K=2)
        _gconv_layers = []
        _attention_layer = []

        self.dim_model = hid_dim
        c = copy.deepcopy
        attn = MultiHeadedAttention(n_head, self.dim_model)
        gcn = GraphNet(in_features=self.dim_model, out_features=self.dim_model, n_pts=n_pts)

        for i in range(num_layers):
            _gconv_layers.append(_ResChebGC(adj=self.adj, input_dim=hid_dim, output_dim=hid_dim,
                                            hid_dim=hid_dim, p_dropout=0.1))
            _attention_layer.append(GraAttenLayer(self.dim_model, c(attn), c(gcn), dropout))

        self.gconv_layers = nn.ModuleList(_gconv_layers)
        self.atten_layers = nn.ModuleList(_attention_layer)

        _gconv_layers_vis = []
        _attention_layer_vis = []

        for i in range(num_layers):
            _gconv_layers_vis.append(_ResChebGC(adj=self.adj, input_dim=hid_dim, output_dim=hid_dim,
                                                hid_dim=hid_dim, p_dropout=0.1))
            _attention_layer_vis.append(GraAttenLayer(self.dim_model, c(attn), c(gcn), dropout))

        self.gconv_layers_vis = nn.ModuleList(_gconv_layers_vis)
        self.atten_layers_vis = nn.ModuleList(_attention_layer_vis)

        self.last_gconv_layer_vis = ChebConv(in_c=self.dim_model, out_c=vis_embed_dim, K=2)
        self.visibility_class_head = nn.Sequential(
            nn.Linear(vis_embed_dim, num_classes),
            nn.BatchNorm1d(self.n_pts),
            nn.Sigmoid()
        )

        self.last_gconv_layer = ChebConv(in_c=self.dim_model, out_c=pose_embed_dim, K=2)
        channel = pose_embed_dim * n_pts
        self.head = nn.Sequential(
            nn.BatchNorm1d(channel, momentum=0.1),
            nn.Conv1d(channel, 3 * n_pts, kernel_size=1)
        )

    def forward(self, x, mask=None):
        if mask is None:
            mask = self.src_mask

        b, f = x.shape[0], x.shape[1]
        x = rearrange(x, 'b f j c  -> (b f) j c', )
        orig_x = copy.deepcopy(x)

        x = self.gconv_input(x, self.adj)
        for i in range(self.n_layers):
            x = self.atten_layers_vis[i](x, mask)
            x = self.gconv_layers_vis[i](x)

        x = self.last_gconv_layer_vis(x, self.adj)

        out_vis = self.visibility_class_head(x)
        out_vis = rearrange(out_vis, '(b f) j c -> b f j c', f=f).contiguous()

        x = torch.concat([orig_x, x], dim=2)
        for i in range(self.n_layers):
            x = self.atten_layers[i](x, mask)
            x = self.gconv_layers[i](x)

        out_pose = self.last_gconv_layer(x, self.adj)
        out_pose = rearrange(out_pose, '(b f) j c -> b (j c) f', f=f)
        out_pose = self.head(out_pose)
        out_pose = rearrange(out_pose, 'b (j c) f -> b f j c', j=self.n_pts).contiguous()

        return out_pose, out_vis


class AuxiliaryVisTemporalModel(nn.Module):
    def __init__(self, adj, in_dim=2, hid_dim=64, num_layers=2, n_head=8, dropout=0.1, n_pts=17,
                 pose_embed_dim=16, vis_embed_dim=8, num_classes=1, d_hid=512, frames=81):
        super(AuxiliaryVisTemporalModel, self).__init__()
        self.n_layers = num_layers
        self.adj = adj
        self.n_pts = n_pts
        self.src_mask = torch.full((self.n_pts, ), True).cuda()

        self.gconv_input = ChebConv(in_c=in_dim, out_c=hid_dim, K=2)
        _gconv_layers = []
        _attention_layer = []

        self.dim_model = hid_dim
        c = copy.deepcopy
        attn = MultiHeadedAttention(n_head, self.dim_model)
        gcn = GraphNet(in_features=self.dim_model, out_features=self.dim_model, n_pts=n_pts)

        for i in range(num_layers):
            _gconv_layers.append(_ResChebGC(adj=self.adj, input_dim=hid_dim, output_dim=hid_dim,
                                            hid_dim=hid_dim, p_dropout=0.1))
            _attention_layer.append(GraAttenLayer(self.dim_model, c(attn), c(gcn), dropout))

        self.gconv_layers = nn.ModuleList(_gconv_layers)
        self.atten_layers = nn.ModuleList(_attention_layer)

        _gconv_layers_vis = []
        _attention_layer_vis = []

        for i in range(num_layers):
            _gconv_layers_vis.append(_ResChebGC(adj=self.adj, input_dim=hid_dim, output_dim=hid_dim,
                                                hid_dim=hid_dim, p_dropout=0.1))
            _attention_layer_vis.append(GraAttenLayer(self.dim_model, c(attn), c(gcn), dropout))

        self.gconv_layers_vis = nn.ModuleList(_gconv_layers_vis)
        self.atten_layers_vis = nn.ModuleList(_attention_layer_vis)

        self.last_gconv_layer_vis = ChebConv(in_c=self.dim_model, out_c=vis_embed_dim, K=2)
        self.visibility_class_head = nn.Sequential(
            nn.Linear(vis_embed_dim, num_classes),
            nn.BatchNorm1d(self.n_pts),
            nn.Sigmoid()
        )

        self.last_gconv_layer = ChebConv(in_c=self.dim_model, out_c=pose_embed_dim, K=2)
        channel = pose_embed_dim * n_pts
        self.head = nn.Sequential(
            nn.BatchNorm1d(channel, momentum=0.1),
            nn.Conv1d(channel, 3 * n_pts, kernel_size=1)
        )

        channel_seq = (pose_embed_dim + vis_embed_dim) * self.n_pts
        self.Transformer_reduce = Transformer_reduce(0, channel_seq, d_hid, length=frames, entire_seq=True)

        self.head_seq = nn.Sequential(
            nn.BatchNorm1d(channel_seq, momentum=0.1),
            nn.Conv1d(channel_seq, 3 * self.n_pts, kernel_size=1)
        )

    def forward(self, x, mask=None):
        if mask is None:
            mask = self.src_mask

        b, f, j = x.shape[0], x.shape[1], x.shape[2]
        x = rearrange(x, 'b f j c  -> (b f) j c', )

        x = self.gconv_input(x, self.adj)
        for i in range(self.n_layers):
            x = self.atten_layers[i](x, mask)
            x = self.gconv_layers[i](x)

        out_pose_x = self.last_gconv_layer(x, self.adj)

        out_pose = rearrange(out_pose_x, '(b f) j c -> b (j c) f', f=f)
        out_pose = self.head(out_pose)
        out_pose = rearrange(out_pose, 'b (j c) f -> b f j c', j=self.n_pts).contiguous()

        for i in range(self.n_layers):
            x = self.atten_layers_vis[i](x, mask)
            x = self.gconv_layers_vis[i](x)

        out_vis_x = self.last_gconv_layer_vis(x, self.adj)

        out_vis = self.visibility_class_head(out_vis_x)
        out_vis = rearrange(out_vis, '(b f) j c -> b f j c', f=f).contiguous()

        x = torch.concat([out_pose_x, out_vis_x], dim=2)

        x = rearrange(x, '(b f) j c -> b f (j c)', f=f)
        _, x_temp = self.Transformer_reduce(x)
        x_VTE = x_temp
        x_VTE = x_VTE.permute(0, 2, 1).contiguous()
        x_VTE = self.head_seq(x_VTE)
        x_VTE = rearrange(x_VTE, 'b (j c) f -> b f j c', j=j).contiguous()

        return out_pose, out_vis, x_VTE

### Auxiliary Heatmap Model not up to date!!!
### Code not used currently
class AuxiliaryModel(nn.Module):
    def __init__(self, adj, in_dim=2, hid_dim=64, num_layers=2, n_head=8, dropout=0.1, n_pts=15,
                 heatmap_res_out=32, heatmap_res_hid=32, num_classes=1):
        super(AuxiliaryModel, self).__init__()
        self.n_layers = num_layers
        self.adj = adj
        self.n_pts = n_pts
        self.heatmap_res = heatmap_res_out

        self.gconv_input = ChebConv(in_c=in_dim, out_c=hid_dim, K=2)
        _gconv_layers = []
        _attention_layer = []

        self.dim_model = hid_dim
        c = copy.deepcopy
        attn = MultiHeadedAttention(n_head, self.dim_model)
        gcn = GraphNet(in_features=self.dim_model, out_features=self.dim_model, n_pts=n_pts)

        for i in range(num_layers):
            _gconv_layers.append(_ResChebGC(adj=self.adj, input_dim=hid_dim, output_dim=hid_dim,
                                            hid_dim=hid_dim, p_dropout=0.1))
            _attention_layer.append(GraAttenLayer(self.dim_model, c(attn), c(gcn), dropout))

        self.gconv_layers = nn.ModuleList(_gconv_layers)
        self.atten_layers = nn.ModuleList(_attention_layer)

        self.vis_layer = ChebConv(in_c=self.dim_model, out_c=num_classes, K=2)
        self.visibility_class_head = nn.Sequential(
            nn.BatchNorm1d(self.n_pts),
            nn.Sigmoid()
        )

        # Code taken and adjusted from
        # https://github.com/chrdiller/characteristic3dposes/blob/085151778aafff5bfc3e5e38c4d474c8cf154a4e/characteristic3dposes/model/main.py
        self.heatmap_regression_head = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.n_pts, out_channels=heatmap_res_hid, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(heatmap_res_hid),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=heatmap_res_hid, out_channels=heatmap_res_hid, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(heatmap_res_hid),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=heatmap_res_hid, out_channels=self.n_pts, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.n_pts),
            nn.Sigmoid()
        )

    def forward(self, x, mask=None):
        b, f = x.shape[0], x.shape[1]
        x = rearrange(x, 'b f j c  -> (b f) j c', )

        x = self.gconv_input(x, self.adj)
        for i in range(self.n_layers):
            x = self.atten_layers[i](x, mask)
            x = self.gconv_layers[i](x)

        out_vis = self.vis_layer(x, self.adj)
        out_vis = self.visibility_class_head(out_vis)
        out_vis = rearrange(out_vis, '(b f) j c -> b f j c', f=f).contiguous()

        d = math.floor(math.sqrt(self.dim_model))
        out_heatmap = x.view(b * f, self.n_pts, d, d)
        out_heatmap = self.heatmap_regression_head(out_heatmap)
        out_heatmap = rearrange(out_heatmap, '(b f) j d1 d2 -> b f j d1 d2', f=f).contiguous()

        return out_vis, out_heatmap


