from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
from submodule import *
import pdb

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.02)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        m.weight.data.normal_(0, 0.02)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

def conv2d_bn_relu(in_channels, out_channels, kernel_size, \
        stride=1, padding=0, bn=True, relu=True):
    bias = not bn
    layers = []
    layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride,
        padding, bias=bias))
    if bn:
        layers.append(nn.BatchNorm2d(out_channels))
    if relu:
        layers.append(nn.LeakyReLU(0.2, inplace=True))
    layers = nn.Sequential(*layers)

    for m in layers.modules():
        init_weights(m)

    return layers

class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, dilation=1, downsample=None):
        super(ResBlock, self).__init__()

        # To keep the shape of input and output same when dilation conv, we should compute the padding:
        # Reference:
        #   https://discuss.pytorch.org/t/how-to-keep-the-shape-of-input-and-output-same-when-dilation-conv/14338
        # padding = [(o-1)*s+k+(k-1)*(d-1)-i]/2, here the i is input size, and o is output size.
        # set o = i, then padding = [i*(s-1)+k+(k-1)*(d-1)]/2 = [k+(k-1)*(d-1)]/2      , stride always equals 1
        # if dilation != 1:
        #     padding = (3+(3-1)*(dilation-1))/2
        padding = dilation

        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride,
                     padding=padding, dilation=dilation, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channel)
        self.relu1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=stride,
                               padding=padding, dilation=dilation, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channel)
        self.relu2 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.downsample = downsample
        self.stride = stride
        self.in_ch  = in_channel
        self.out_ch = out_channel
        self.p = padding
        self.d = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)

        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu2(out)
        return out

class MetricBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride = 1):
        super(MetricBlock, self).__init__()
        self.conv3d_1 = nn.Conv3d(in_channel, out_channel, 3, 1, 1)
        self.bn1 = nn.BatchNorm3d(out_channel)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        out = self.conv3d_1(x)
        out = self.bn1(out)
        out = self.relu(out)
        return out   

class siamese_network(nn.Module):
    def __init__(self):
        super(siamese_network, self).__init__()
        self.conv0   = conv2d_bn_relu(1,  32, kernel_size=3, stride=1, padding=1, bn=False, relu=False)
        self.convres = ResBlock(32, 32, stride=1, dilation=1 )
        self.conv1   = conv2d_bn_relu(32, 32, kernel_size=3, stride=2, padding=1)
        self.conv2   = conv2d_bn_relu(32, 32, kernel_size=3, stride=1, padding=1, bn=False, relu=False)
        

    def forward(self, x):
    	x1 = self.conv0(x)
    	x2 = self.convres(self.convres(self.convres(x1)))
    	x3 = self.conv1(self.conv1(self.conv1(x2)))
    	x4 = self.conv2(x3)  
    	return x4 

class refine_disp_network(nn.Module):
    """docstring for refinement_network"""
    def __init__(self):
        super(refine_disp_network, self).__init__()
        self.conv0 = conv2d_bn_relu(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv1 = ResBlock(16, 16, stride=1, dilation=1)
        self.conv2 = ResBlock(16, 16, stride=1, dilation=2)
        self.conv3 = ResBlock(32, 32, stride=1, dilation=4)
        self.conv4 = ResBlock(32, 32, stride=1, dilation=8)
        self.conv5 = ResBlock(32, 32, stride=1, dilation=1)
        self.last  = conv2d_bn_relu(32, 1, kernel_size=3, stride=1, padding=1, bn=False, relu=False)

    def forward(self, disp, img): 
        x1 = self.conv0(disp)
        y1 = self.conv0(img)
        x2 = self.conv1(x1)
        y2 = self.conv1(y1)
        x3 = self.conv2(x2)
        y3 = self.conv2(y2)

        add = torch.cat((x3,y3),1)

        out = self.conv3(add)
        out = self.conv4(out)
        out = self.conv5(self.conv5(out))
        out = self.last(out)
        return out 

class invalidation_network(nn.Module):
    def __init__(self):
        super(invalidation_network, self).__init__()
        self.res0 =  ResBlock(2, 64, stride=1, dilation=1 )
        self.res  =  ResBlock(64, 64, stride=1, dilation=1 )
        self.last =  conv2d_bn_relu(64, 1, kernel_size=3, stride=1, padding=1, bn=False, relu=False)

    def forward(self, left,right):
        out = torch.cat((left,right),1)
        out = self.res0(out)
        for i in range(4):
          out += self.res(out)

        out = self.last(out) 
        return out    

class refine_inva_network(nn.Module):
    """docstring for refinement_network"""
    def __init__(self):
        super(refine_inva_network, self).__init__()
        self.conv0 = conv2d_bn_relu(32, 32, kernel_size=3, stride=1, padding=1)
        self.res   = ResBlock(32, 32, stride=1, dilation=1 )
        self.last  = conv2d_bn_relu(32, 1, kernel_size=3, stride=1, padding=1, bn=False, relu=False)

    def forward(self, x): 
        out = self.conv0(x)

        for i in range(4):
          out += self.res(out)

        out = self.last(out) 
        return out   

class ActiveStereoNet(nn.Module):
    def __init__(self, maxdisp):
        super(ActiveStereoNet, self).__init__()
        self.maxdisp = maxdisp
        self.siamese_net = siamese_network()
    	self.cost_volume = nn.Sequential(
            MetricBlock(64, 32),
            MetricBlock(32, 32),
            MetricBlock(32, 32),
            MetricBlock(32, 32),
            nn.Conv3d(32, 1, 3, padding=1),
            )


    	self.refine_disp_net = refine_disp_network()
    	self.invalidation_net = invalidation_network()
    	self.refine_inva_net = refine_inva_network()      

    def forward(self, left, right):
    	left_fea  = self.siamese_net(left)
    	right_fea = self.siamese_net(right)

        #matching
        costl = Variable(torch.FloatTensor(left_fea.size()[0], left_fea.size()[1]*2, self.maxdisp/8,  left_fea.size()[2],  left_fea.size()[3]).zero_()).cuda()
        costr = Variable(torch.FloatTensor(right_fea.size()[0], right_fea.size()[1]*2, self.maxdisp/8,  right_fea.size()[2],  right_fea.size()[3]).zero_()).cuda()

        for i in range(int(self.maxdisp/8)):
            if i > 0 :
             costl[:, :left_fea.size()[1], i, :,i:] = left_fea[:,:,:,i:]
             costl[:, left_fea.size()[1]:, i, :,i:] = right_fea[:,:,:,:-i]
            else:
             costl[:, :left_fea.size()[1], i, :,:]  = left_fea
             costl[:, left_fea.size()[1]:, i, :,:]  = right_fea
        costl = costl.contiguous()
        
        for i in range(int(self.maxdisp/8)):
            if i > 0 :
             costr[:, :right_fea.size()[1], i, :,:(right_fea.size()[3]-i)] = right_fea[:,:,:,:-i]
             costr[:, right_fea.size()[1]:, i, :,:(right_fea.size()[3]-i)] = left_fea[:,:,:,i:]
            else:
             costr[:, :right_fea.size()[1], i, :,:right_fea.size()[3]]   = right_fea
             costr[:, right_fea.size()[1]:, i, :,:right_fea.size()[3]]   = left_fea
        costr = costr.contiguous()        

    	costl0 = self.cost_volume(costl)
        costr0 = self.cost_volume(costr)

        costl0 = F.interpolate(costl0, [self.maxdisp,left.size()[2],left.size()[3]], mode='trilinear',align_corners=True)
        costr0 = F.interpolate(costr0, [self.maxdisp,left.size()[2],left.size()[3]], mode='trilinear',align_corners=True)

        costl0 = torch.squeeze(costl0,1)
        predl  = F.softmax(costl0,dim=1)
        predl  = disparityregression(self.maxdisp)(predl)
        costr0 = torch.squeeze(costr0,1)
        predr  = F.softmax(costr0,dim=1)
        predr  = disparityregression(self.maxdisp)(predr)

        predl_ = torch.unsqueeze(predl,1)
        predr_ = torch.unsqueeze(predr,1)

        displ_out = self.refine_disp_net(predl_, left) + predl_
        dispr_out = self.refine_disp_net(predr_, right) + predr_

    	return  predl, predr, torch.squeeze(displ_out,1), torch.squeeze(dispr_out,1)  #, low_invalidation_l, low_invalidation_r, mask_out_l, mask_out_r

