import torch
import torch.nn as nn
import math

class Block(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, dwise_stride=1, relu=True):
        super(Block, self).__init__()
        
        self.conv1=nn.Conv2d(in_channels, mid_channels, kernel_size=1, 
                             stride=1, padding=0)
        self.bn1=nn.BatchNorm2d(mid_channels)
        self.relu1=nn.ReLU(inplace=True)
        self.conv2=nn.Conv2d(mid_channels, mid_channels, kernel_size=3, 
                             stride=dwise_stride, padding=1, groups=mid_channels)
        self.bn2=nn.BatchNorm2d(mid_channels)
        self.relu2=nn.ReLU(inplace=True)
        self.conv3=nn.Conv2d(mid_channels, out_channels, kernel_size=1,
                             stride=1, padding=0)
        self.bn3=nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu1(x)
        x=self.conv2(x)
        x=self.bn2(x)
        x=self.relu2(x)
        x=self.conv3(x)
        x=self.bn3(x)
        return x
        
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, dwise_stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1=nn.Conv2d(in_channels, mid_channels, kernel_size=1, 
                             stride=1, padding=0)
        self.bn1=nn.BatchNorm2d(mid_channels)
        self.relu1=nn.ReLU(inplace=True)
        self.conv2=nn.Conv2d(mid_channels, mid_channels, kernel_size=3, 
                             stride=dwise_stride, padding=1, groups=mid_channels)
        self.bn2=nn.BatchNorm2d(mid_channels)
        self.relu2=nn.ReLU(inplace=True)
        self.conv3=nn.Conv2d(mid_channels, out_channels, kernel_size=1,
                             stride=1, padding=0)
        self.bn3=nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        res=x
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu1(x)
        x=self.conv2(x)
        x=self.bn2(x)
        x=self.relu2(x)
        x=self.conv3(x)
        x=self.bn3(x)
        x+=res
        return x
    
class Cell(nn.Module):
    def __init__(self, channels=None, dwise_stride=1):
        super(Cell, self).__init__()
        
        self.block=Block(channels[0][0],channels[0][1],channels[0][2],
                         dwise_stride=dwise_stride)
        blocks=[]
        for i in range(1,len(channels)):
            blocks.append(ResidualBlock(channels[i][0],channels[i][1], channels[i][2]))
        self.residual_blocks=nn.Sequential(*blocks)  
    
    def forward(self, x):
        x=self.block(x)
        x=self.residual_blocks(x)
        return x
    
class Mobilenet(nn.Module):
    def __init__(self, num_classes=21, pretrain=False):
        super(Mobilenet, self).__init__()
        self.channels=[[32,32,16],  #conv2_1
                       [16,96,24],  #conv2_2  /2
                       [24,144,24], #conv3_1
                       [24,144,32], #conv3_2  /2
                       [32,192,32], #conv4_1
                       [32,192,32], #conv4_2
                       [32,192,64], #conv4_3
                       [64,384,64], #conv4_4
                       [64,384,64], #conv4_5
                       [64,384,64], #conv4_6
                       [64,384,96], #conv4_7
                       [96,576,96], #conv5_1
                       [96,576,96], #conv5_2
                       [96,576,160], #conv5_3
                       [160,960,160], #conv6_1
                       [160,960,160], #conv6_2
                       [160,960,320], #conv6_3
                        ]
        self.prefix=nn.Conv2d(3,32,kernel_size=3,stride=2,padding=1)    #/2
        self.body_blocks=[self.prefix]
        self.body_blocks.append(self.make_layer(0,0))
        self.body_blocks.append(self.make_layer(1,2,stride=2))
        self.body_blocks.append(self.make_layer(3,5,stride=2))
        self.body_blocks.append(self.make_layer(6,9))
        self.body_blocks.append(self.make_layer(10,12))
        self.body_blocks.append(self.make_layer(13,15))
        self.body_blocks.append(self.make_layer(16,16))
        
        self.task=nn.Sequential(nn.Conv2d(320,128,kernel_size=3,padding=1),
                                nn.BatchNorm2d(128),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(128,128,kernel_size=3,padding=1),
                                nn.BatchNorm2d(128),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(128,128,kernel_size=1,padding=0),
                                nn.BatchNorm2d(128),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(128,num_classes+1,kernel_size=1),
                                nn.BatchNorm2d(num_classes+1))
        self.body_blocks.append(self.task)
        self.net=nn.Sequential(*self.body_blocks)
        
        if not pretrain:
            self._init_weights(self.net)

    def make_layer(self, start, end, stride=1):
        return Cell(self.channels[start:end+1],dwise_stride=stride)
        
    def _init_weights(self, model):
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
#                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.001)
                m.bias.data.zero_()
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
        
    def forward(self, x):
        return self.net(x)
    