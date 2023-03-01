from __future__ import absolute_import

import torch.nn as nn
import torch
import copy, math
import torch.nn.functional as F
from einops import rearrange
from torch.nn.parameter import Parameter
from model.block.chebconv import ChebConv, _ResChebGC


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        # features=layer.size=512
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class GraAttenLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(GraAttenLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


def attention(Q, K, V, mask=None, dropout=None):
    # Query=Key=Value: [batch_size, 8, max_len, 64]
    d_k = Q.size(-1)
    # Q * K.T = [batch_size, 8, max_len, 64] * [batch_size, 8, 64, max_len]
    # scores: [batch_size, 8, max_len, max_len]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    # padding mask
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, V), p_attn


class MultiHeadedAttention(nn.Module):

    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        Q, K, V = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                   for l, x in zip(self.linears, (query, key, value))]

        x, self.attn = attention(Q, K, V, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)

        return self.linears[-1](x)


src_mask = torch.tensor([[[True, True, True, True, True, True, True, True, True, True, True,
                           True, True, True, True, True, True, True, True, True, True]]])


class LAM_Gconv(nn.Module):

    def __init__(self, in_features, out_features, activation=nn.ReLU(inplace=True)):
        super(LAM_Gconv, self).__init__()
        self.fc = nn.Linear(in_features=in_features, out_features=out_features)
        self.activation = activation

    def laplacian(self, A_hat):
        D_hat = (torch.sum(A_hat, 0) + 1e-5) ** (-0.5)
        L = D_hat * A_hat * D_hat
        return L

    def laplacian_batch(self, A_hat):
        batch, N = A_hat.shape[:2]
        D_hat = (torch.sum(A_hat, 1) + 1e-5) ** (-0.5)
        L = D_hat.view(batch, N, 1) * A_hat * D_hat.view(batch, 1, N)
        return L

    def forward(self, X, A):
        batch = X.size(0)
        A_hat = A.unsqueeze(0).repeat(batch, 1, 1)
        X = self.fc(torch.bmm(self.laplacian_batch(A_hat), X))
        if self.activation is not None:
            X = self.activation(X)
        return X


class GraphNet(nn.Module):

    def __init__(self, in_features=2, out_features=2, n_pts=21):
        super(GraphNet, self).__init__()

        self.A_hat = Parameter(torch.eye(n_pts).float(), requires_grad=True)
        self.gconv1 = LAM_Gconv(in_features, in_features * 2)
        # self.gconv2 = LAM_Gconv(in_features * 2, out_features, activation=None)
        self.gconv2 = LAM_Gconv(in_features * 2, out_features)

    def forward(self, X):
        X_0 = self.gconv1(X, self.A_hat)
        X_1 = self.gconv2(X_0, self.A_hat)
        return X_1


class SpatialGraphTransformer(nn.Module):
    def __init__(self, adj, in_dim=2, hid_dim=64, num_layers=2,
                 n_head=8, dropout=0.1, n_pts=17, out_dim=3):
        super(SpatialGraphTransformer, self).__init__()
        self.n_layers = num_layers
        self.adj = adj

        self.gconv_input = ChebConv(in_c=in_dim, out_c=hid_dim, K=2)
        _gconv_layers = []
        _attention_layer = []

        dim_model = hid_dim
        c = copy.deepcopy
        attn = MultiHeadedAttention(n_head, dim_model)
        gcn = GraphNet(in_features=dim_model, out_features=dim_model, n_pts=n_pts)

        for i in range(num_layers):
            _gconv_layers.append(_ResChebGC(adj=self.adj, input_dim=hid_dim, output_dim=hid_dim,
                                            hid_dim=hid_dim, p_dropout=0.1))
            _attention_layer.append(GraAttenLayer(dim_model, c(attn), c(gcn), dropout))

        self.gconv_layers = nn.ModuleList(_gconv_layers)
        self.atten_layers = nn.ModuleList(_attention_layer)

        self.last_gconv_layer = ChebConv(in_c=hid_dim, out_c=out_dim, K=2)

    def forward(self, x, mask=None):
        x = self.gconv_input(x, self.adj)
        for i in range(self.n_layers):
            x = self.atten_layers[i](x, mask)
            x = self.gconv_layers[i](x)
        x = self.last_gconv_layer(x, self.adj)
        return x, self.atten_layers[0].self_attn.attn, self.atten_layers[1].self_attn.attn


class SpatialVisGraphTransformer(nn.Module):
    def __init__(self, adj, in_dim=2, hid_dim=64, num_layers=2, n_head=8, dropout=0.1, n_pts=17,
                 pose_embed_dim=16, vis_embed_dim=8):
        super(SpatialVisGraphTransformer, self).__init__()
        self.n_layers = num_layers
        self.adj = adj

        self.gconv_input = ChebConv(in_c=in_dim, out_c=hid_dim, K=2)
        _gconv_layers = []
        _attention_layer = []

        dim_model = hid_dim
        c = copy.deepcopy
        attn = MultiHeadedAttention(n_head, dim_model)
        gcn = GraphNet(in_features=dim_model, out_features=dim_model, n_pts=n_pts)

        for i in range(num_layers):
            _gconv_layers.append(_ResChebGC(adj=self.adj, input_dim=hid_dim, output_dim=hid_dim,
                                            hid_dim=hid_dim, p_dropout=0.1))
            _attention_layer.append(GraAttenLayer(dim_model, c(attn), c(gcn), dropout))

        self.gconv_layers = nn.ModuleList(_gconv_layers)
        self.atten_layers = nn.ModuleList(_attention_layer)

        self.last_gconv_layer = ChebConv(in_c=hid_dim, out_c=pose_embed_dim, K=2)

        self.vis_branch = nn.Sequential(
            nn.Linear(dim_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, vis_embed_dim),
            nn.ReLU()
        )

    def forward(self, x, mask=None):
        x = self.gconv_input(x, self.adj)
        for i in range(self.n_layers):
            x = self.atten_layers[i](x, mask)
            x = self.gconv_layers[i](x)

        pose_embedding = self.last_gconv_layer(x, self.adj)
        vis_embedding = self.vis_branch(x)

        # return torch.concat([pose_embedding, vis_embedding], dim=2)
        return pose_embedding, vis_embedding, self.atten_layers[0].self_attn.attn, self.atten_layers[1].self_attn.attn
