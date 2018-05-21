# from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import resnet

############################################################
#  FPN Graph
############################################################

class FPN(nn.Module):  # xavier_fill as default
    def __init__(self, out_channels):
        super(FPN, self).__init__()
        self.out_channels = out_channels 
        self.P6 = nn.MaxPool2d(kernel_size=1, stride=2, padding=0, ceil_mode=False)
        self.P5_conv1 = nn.Conv2d(512, self.out_channels, kernel_size=1, stride=1)
        # *4
        self.P4_conv1 = nn.Conv2d(256, self.out_channels, kernel_size=1, stride=1)
        self.P4_conv2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1)

        self.P3_conv1 = nn.Conv2d(128, self.out_channels, kernel_size=1, stride=1)
        self.P3_conv2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
        
        self.P2_conv1 = nn.Conv2d(64, self.out_channels, kernel_size=1, stride=1)
        self.P2_conv2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, C1, C2, C3, C4, C5):
        p5_out = self.P5_conv1(C5)

        p4_out = torch.add(self.P4_conv1(C4), F.upsample(p5_out, scale_factor=2))
        p3_out = torch.add(self.P3_conv1(C3), F.upsample(p4_out, scale_factor=2))
        p2_out = torch.add(self.P2_conv1(C2), F.upsample(p3_out, scale_factor=2))

        p4_out = self.P4_conv2(p4_out)
        p3_out = self.P3_conv2(p3_out)
        p2_out = self.P2_conv2(p2_out)

        # P6 is used for the 5th anchor scale in RPN. Generated by
        # subsampling from P5 with stride of 2.
        p6_out = self.P6(p5_out)

        return p2_out, p3_out, p4_out, p5_out, p6_out


############################################################
#  Pose Estimation Graph
############################################################

class pose_estimation(nn.Module):
    def __init__(self, class_num=21, pretrain=False):
        super(pose_estimation, self).__init__()
        self.resnet = resnet.resnet18(pretrained=pretrain)
        # self.apply_fix()
        
        self.out_channels = 256
        self.fpn = FPN(self.out_channels)
        
        self.predict = nn.Sequential(nn.Conv2d(256, 128, 3, 1, 1), nn.ReLU(inplace=True),
                                     nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU(inplace=True),
                                     nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU(inplace=True),
                                     nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU(inplace=True),
                                     nn.Conv2d(128, class_num+1, 3, 1, 1))
        if not pretrain:
            self._init_weights(self.resnet)
            self._init_weights(self.fpn)
            self._init_weights(self.predict)


    def _init_weights(self, model):
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
#                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.001)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def load_weights(self, model_path=None):
        model_dict = self.state_dict()
        print('loading model from {}'.format(model_path))
        try:
            pretrained_dict = torch.load(model_path)
            from collections import OrderedDict
            tmp = OrderedDict()
            for k,v in pretrained_dict.items():
                if k in model_dict:
                    tmp[k] = v
            model_dict.update(tmp)
            self.load_state_dict(model_dict)
        except:
            print ('loading model failed, {} may not exist'.format(model_path))
            
    def apply_fix(self, model):
#        for param in self.resnet.conv1.parameters():
        for param in model.parameters():    
            param.requires_grad = False
#        for param in self.resnet.layer1.parameters():
        for param in model.parameters():
            param.requires_grad = False

    def forward(self, x):
        C1, C2, C3, C4, C5 = self.resnet(x)
        P2, P3, P4, P5, P6 = self.fpn(C1, C2, C3, C4, C5)
        # P3_x2 = F.upsample(P3, scale_factor=2)
        # out = self.predict(P3_x2)
#        P4_x2 = F.upsample(P4, scale_factor=2, mode='bilinear') # 
        out = self.predict(P3)

        return out