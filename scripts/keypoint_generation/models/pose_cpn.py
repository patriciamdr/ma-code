import math

import torch
from torch import nn

from demo.lib.hrnet.lib.models.pose_hrnet import Bottleneck as ResNetBottleneck

# https://github.com/lmb633/cpn-pytorch/blob/ed0cbf6e8b0c09a4388f440cc14735dee6191869/models.py
# https://github.com/GengDavid/pytorch-cpn
from torch.utils import model_zoo


class GlobalNet(nn.Module):
    def __init__(self, channel_sets, out_shape, num_class):
        super(GlobalNet, self).__init__()
        self.channel_sets = channel_sets
        laterals, upsamples, predicts = [], [], []
        self.layers = len(channel_sets)
        for i in range(self.layers):
            laterals.append(self._lateral(channel_sets[i]))
            predicts.append(self._predict(out_shape, num_class))
            if i != self.layers - 1:
                upsamples.append(self._upsample())
        self.laterals = nn.ModuleList(laterals)
        self.upsamples = nn.ModuleList(upsamples)
        self.predict = nn.ModuleList(predicts)

    def _lateral(self, in_channel):
        return nn.Sequential(
            nn.Conv2d(in_channel, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

    def _upsample(self):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256)
        )

    def _predict(self, out_shape, num_class):
        return nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_class, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Upsample(size=out_shape, mode='bilinear', align_corners=True),
            nn.BatchNorm2d(num_class)
        )

    def forward(self, x):
        features, predicts = [], []
        for i in range(self.layers):
            feature = self.laterals[i](x[i])
            if i > 0:
                feature = feature + up
            features.append(feature)
            if i < self.layers - 1:
                up = self.upsamples[i](feature)
            predicts.append(self.predict[i](feature))
        return features, predicts


class Bottleneck(nn.Module):
    def __init__(self, in_channel, planes, stride=1, expansion=2):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channel, planes * 2, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * expansion)
        )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class RefineNet(nn.Module):
    def __init__(self, lateral_channel, out_shape, num_class):
        super(RefineNet, self).__init__()
        self.cascade_num = 4
        cascades = []
        for i in range(self.cascade_num):
            cascades.append(self._maker_layer(lateral_channel, self.cascade_num - i - 1, out_shape))
        self.cascade = nn.ModuleList(cascades)
        self.final_predict = self._predict(lateral_channel * self.cascade_num, num_class)

    def _maker_layer(self, in_channel, num, output_shape):
        layers = []
        for i in range(num):
            layers.append(Bottleneck(in_channel, 128, expansion=2))
        layers.append(nn.Upsample(size=output_shape, mode='bilinear', align_corners=True))
        return nn.Sequential(*layers)

    def _predict(self, in_channel, num_class):
        return nn.Sequential(
            Bottleneck(in_channel, 128, expansion=2),
            nn.Conv2d(256, num_class, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_class)
        )

    def forward(self, x):
        refine_feature = []
        for i in range(self.cascade_num):
            refine_feature.append((self.cascade[i](x[i])))
        out = torch.cat(refine_feature, dim=1)
        out = self.final_predict(out)
        return out

model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return [x4, x3, x2, x1]


def resnet50(pretrained=False, **kwargs):
    model = ResNet(ResNetBottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        state_dict = model.state_dict()
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet50'])
        for k, v in pretrained_state_dict.items():
            if k not in state_dict:
                continue
            state_dict[k] = v
        print('successfully load ' + str(len(state_dict.keys())) + ' keys')
        model.load_state_dict(state_dict)
    return model


class CPN(nn.Module):
    def __init__(self, channel_sets=None, out_shape=(96, 72), n_class=17, pretrained=True):
        super(CPN, self).__init__()
        if channel_sets is None:
            channel_sets = [2048, 1024, 512, 256]
        self.resnet = resnet50(pretrained)
        self.global_net = GlobalNet(channel_sets=channel_sets, out_shape=out_shape, num_class=n_class)
        self.refine_net = RefineNet(lateral_channel=channel_sets[-1], out_shape=out_shape, num_class=n_class)

    def forward(self, x):
        feature = self.resnet(x)
        global_f, global_pred = self.global_net(feature)
        refine_pred = self.refine_net(global_f)
        return global_pred, refine_pred


def get_cpn():
    model = CPN().cuda()
    model = torch.nn.DataParallel(model)
    checkpoint = torch.load('/home/patricia/dev/StridedTransformer-Pose3D/scripts/keypoint_generation/checkpoint/CPN50_384x288.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])
    return model


