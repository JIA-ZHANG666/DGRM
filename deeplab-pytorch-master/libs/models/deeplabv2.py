#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-11-19

from __future__ import absolute_import, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from .sgr_layer import  GRMLayer
from .sgr_layer import  DualGCNHead
from .resnet import _ConvBnReLU, _ResLayer, _Stem
from collections import OrderedDict

class _ASPP(nn.Module):
    """
    Atrous spatial pyramid pooling (ASPP)
    """

    def __init__(self, in_ch, out_ch, rates):
        super(_ASPP, self).__init__()
        for i, rate in enumerate(rates):
            self.add_module(
                "c{}".format(i),
                nn.Conv2d(in_ch, out_ch, 3, 1, padding=rate, dilation=rate, bias=True),
            )

        for m in self.children():
            nn.init.normal_(m.weight, mean=0, std=0.01)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return sum([stage(x) for stage in self.children()])


class DeepLabV2_SGR(nn.Sequential):
    """
    DeepLab v2: Dilated ResNet + ASPP
    Output stride is fixed at 8
    """

    def __init__(self, n_classes, n_blocks, atrous_rates, input_feature_channels,
                 visual_feature_channels, num_symbol_node,
                 fasttest_embeddings, fasttest_dim, graph_adj_mat):
        super(DeepLabV2_SGR, self).__init__()
        ch = [64 * 2 ** p for p in range(6)]
        self.add_module("layer1", _Stem(ch[0]))
        self.add_module("layer2", _ResLayer(n_blocks[0], ch[0], ch[2], 1, 1))
       
        self.add_module("layer3", _ResLayer(n_blocks[1], ch[2], ch[3], 2, 1))
       
        self.add_module("layer4", _ResLayer(n_blocks[2], ch[3], ch[4], 1, 2))
        self.add_module("layer5", _ResLayer(n_blocks[3], ch[4], ch[5], 1, 4))
       
        self.add_module("aspp", _ASPP(ch[5],visual_feature_channels, atrous_rates))
        self.add_module("relu", nn.ReLU(inplace=False))
        self.add_module("sgr", GRMLayer(input_feature_channels,  visual_feature_channels, num_symbol_node,
                                       fasttest_embeddings, fasttest_dim, graph_adj_mat) ),
        #self.add_module("sgr", DualGCNHead(input_feature_channels, visual_feature_channels, num_symbol_node,
                                        #fasttest_embeddings, fasttest_dim, graph_adj_mat) ),
        #self.add_module("relu", nn.ReLU(inplace=False))
       
        #self.add_module("aspp", _ASPP(ch[5], n_classes, atrous_rates))
        

        self.add_module("convs", nn.Conv2d(visual_feature_channels, n_classes,
                                         1, 1, bias=False))

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, _ConvBnReLU.BATCH_NORM):
                m.eval()

    def forward(self, x):
       res = super(DeepLabV2_SGR, self).forward(x)
       return res

###########################
class DeepLabV2_SGR_1(nn.Module):
    def __init__(self, n_classes, n_blocks, atrous_rates, input_feature_channels,
                 visual_feature_channels, num_symbol_node,
                 fasttest_embeddings, fasttest_dim, graph_adj_mat):
        super(DeepLabV2_SGR_2, self).__init__()
        # [64, 128, 256, 512, 1024, 2048]
        ch = [64 * 2 ** p for p in range(6)]
        #  _Stem(64)
        self.features_extractor = nn.Sequential(
            OrderedDict(
                [   ("layer1", _Stem(ch[0])),
                    ("layer2", _ResLayer(n_blocks[0], ch[0], ch[2], 1, 1)),
                    ("layer3", _ResLayer(n_blocks[1], ch[2], ch[3], 2, 1)),
                    ("layer4", _ResLayer(n_blocks[2], ch[3], ch[4], 1, 2)),
                    ("layer5", _ResLayer(n_blocks[3], ch[4], ch[5], 1, 4)),
                    
                ]
            )
        )

        self.predict1 = nn.Sequential(
            OrderedDict([
                ("aspp", _ASPP(ch[5], n_classes, atrous_rates)),
                # ("bn", nn.BatchNorm2d(n_classes)),
                # ("relu", nn.ReLU(inplace=False))
            ])
        )

        self.predict2 = nn.Sequential(
            OrderedDict([
                ("aspp", _ASPP(ch[5], input_feature_channels, atrous_rates))
                ("sgr", SGRLayer(input_feature_channels, visual_feature_channels, num_symbol_node,
                                         fasttest_embeddings, fasttest_dim, graph_adj_mat) ),
                ("conv", nn.Conv2d(input_feature_channels, n_classes,
                                          1, 1, bias=False)),
                # ("bn", nn.BatchNorm2d(n_classes)),
                # ("relu", nn.ReLU(inplace=False))
            ])
        )

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, _ConvBnReLU.BATCH_NORM):
                m.eval()

    def forward(self, x):
       # x为图片【？，C, H，W】
       self.feats = self.features_extractor(x)
       #print(torch.cuda.empty_cache())
       #print('Model Input shape:', x.shape)
       #self.upsampled_feats = F.interpolate(self.feats, 
                #size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)
       #print(torch.cuda.empty_cache())
       predictions1 = self.predict1(self.feats)
       predictions2 = self.predict2(self.feats)
       print('predictions shape:', self.predictions.shape)

       return predictions1,predictions2
#################################



###########################
class DeepLabV2_SGR_2(nn.Module):
    def __init__(self, n_classes, n_blocks, atrous_rates, input_feature_channels,
                 visual_feature_channels, num_symbol_node,
                 fasttest_embeddings, fasttest_dim, graph_adj_mat):
        super(DeepLabV2_SGR_2, self).__init__()
        # [64, 128, 256, 512, 1024, 2048]
        ch = [64 * 2 ** p for p in range(6)]
        #  _Stem(64)
        self.features_extractor = nn.Sequential(
            OrderedDict(
                [   ("layer1", _Stem(ch[0])),
                    ("layer2", _ResLayer(n_blocks[0], ch[0], ch[2], 1, 1)),
                    ("layer3", _ResLayer(n_blocks[1], ch[2], ch[3], 2, 1)),
                    ("layer4", _ResLayer(n_blocks[2], ch[3], ch[4], 1, 2)),
                    ("layer5", _ResLayer(n_blocks[3], ch[4], ch[5], 1, 4)),
                    ("aspp", _ASPP(ch[5], input_feature_channels, atrous_rates))
                ]
            )
        )

        self.predict = nn.Sequential(
            OrderedDict([
                ("sgr", SGRLayer(input_feature_channels, visual_feature_channels, num_symbol_node,
                                         fasttest_embeddings, fasttest_dim, graph_adj_mat) ),
                ("conv", nn.Conv2d(input_feature_channels, n_classes,
                                          1, 1, bias=False)),
                # ("bn", nn.BatchNorm2d(n_classes)),
                # ("relu", nn.ReLU(inplace=False))
            ])
        )

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, _ConvBnReLU.BATCH_NORM):
                m.eval()

    def forward(self, x):
       # x为图片【？，C, H，W】
       self.feats = self.features_extractor(x)
       #print(torch.cuda.empty_cache())
       #print('Model Input shape:', x.shape)
       #self.upsampled_feats = F.interpolate(self.feats, 
                #size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)
       #print(torch.cuda.empty_cache())
       self.predictions = self.predict(self.upsampled_feats)
       #print('predictions shape:', self.predictions.shape)

       return self.predictions
#################################

if __name__ == "__main__":
    model = DeepLabV2(
        n_classes=21, n_blocks=[3, 4, 23, 3], atrous_rates=[6, 12, 18, 24]
    )
    model.eval()
    image = torch.randn(1, 3, 513, 513)

    print(model)
    print("input:", image.shape)
    print("output:", model(image).shape)
