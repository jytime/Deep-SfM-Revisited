# This is a Debug version and a bit different from the released one https://github.com/jytime/DICL-Flow
# would convert this to be consistent with the released one

import torch
import torch.nn as nn
from torch.autograd import Variable
import os
import sys
import torch.nn.functional as F
import numpy as np
from lib.config import cfg
import pdb
try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass


__all__ = ['diclflow_net_shallow']

def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation, dilation = dilation, bias=False), nn.BatchNorm2d(out_planes))

class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, dilation=1, stride=1, downsample=None):
        super(ResBlock, self).__init__()

        # To keep the shape of input and output same when dilation conv, we should compute the padding:
        # Reference:
        #   https://discuss.pytorch.org/t/how-to-keep-the-shape-of-input-and-output-same-when-dilation-conv/14338
        # padding = [(o-1)*s+k+(k-1)*(d-1)-i]/2, here the i is input size, and o is output size.
        # set o = i, then padding = [i*(s-1)+k+(k-1)*(d-1)]/2 = [k+(k-1)*(d-1)]/2      , stride always equals 1
        # if dilation != 1:
        #     padding = (3+(3-1)*(dilation-1))/2
        padding = dilation

        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride,padding=padding, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=stride,padding=padding, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.downsample = downsample
        self.stride = stride
        self.in_ch = in_channel
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

class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, deconv=False, bn=True, relu=True, **kwargs):
        super(BasicConv, self).__init__()
        self.relu = relu
        self.use_bn = bn
        if self.use_bn: self.bn = nn.BatchNorm2d(out_channels)
        if deconv:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        
    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

class Conv2x(nn.Module):
    def __init__(self, in_channels, out_channels, deconv=False, concat=True, bn=True, relu=True):
        super(Conv2x, self).__init__()
        self.concat = concat
        self.deconv = deconv
        if deconv:
            kernel = 4
        else:
            kernel = 3

        if self.deconv and cfg.NO_DECONV:
            self.conv_r = BasicConv(in_channels, out_channels, False, bn, relu, kernel_size=3, stride=1, padding=1)
        else:
            self.conv1 = BasicConv(in_channels, out_channels, deconv, bn=False, relu=True, kernel_size=kernel, stride=2, padding=1)

        if self.concat: 
            self.conv2 = BasicConv(out_channels*2, out_channels, False, bn, relu, kernel_size=3, stride=1, padding=1)
        else:
            self.conv2 = BasicConv(out_channels, out_channels, False, bn, relu, kernel_size=3, stride=1, padding=1)

    def forward(self, x, rem):
        if self.deconv and cfg.NO_DECONV:
            x = self.conv_r(x)
            x = F.interpolate(x, [rem.shape[2],rem.shape[3]], mode='bilinear',align_corners=True)
        else:
            x = self.conv1(x)

        assert(x.size() == rem.size())

        if self.concat:
            x = torch.cat((x, rem), 1)
        else: 
            x = x + rem
        x = self.conv2(x)
        return x
      
def predict_flow(in_planes):
    return nn.Conv2d(in_planes,2,kernel_size=3,stride=1,padding=1,bias=True)


class FeatureGA(nn.Module):
    def __init__(self):
        super(FeatureGA, self).__init__()

        self.conv_start = nn.Sequential(
            BasicConv(3, 32, kernel_size=3, padding=1),
            BasicConv(32, 32, kernel_size=3, stride=2, padding=1),
            BasicConv(32, 32, kernel_size=3, padding=1))
        self.conv1a = BasicConv(32, 48, kernel_size=3, stride=2, padding=1)
        self.conv2a = BasicConv(48, 64, kernel_size=3, stride=2, padding=1)
        self.conv3a = BasicConv(64, 96, kernel_size=3, stride=2, padding=1)
        self.conv4a = BasicConv(96, 128, kernel_size=3, stride=2, padding=1)
        self.conv5a = BasicConv(128, 160, kernel_size=3, stride=2, padding=1)
        self.conv6a = BasicConv(160, 192, kernel_size=3, stride=2, padding=1)

        self.deconv6a = Conv2x(192, 160, deconv=True)
        self.deconv5a = Conv2x(160, 128, deconv=True)
        self.deconv4a = Conv2x(128, 96, deconv=True)
        self.deconv3a = Conv2x(96, 64, deconv=True)
        self.deconv2a = Conv2x(64, 48, deconv=True)
        self.deconv1a = Conv2x(48, 32, deconv=True)

        self.conv1b = Conv2x(32, 48)
        self.conv2b = Conv2x(48, 64)
        self.conv3b = Conv2x(64, 96)
        self.conv4b = Conv2x(96, 128)
        self.conv5b = Conv2x(128, 160)
        self.conv6b = Conv2x(160, 192)

        self.deconv6b = Conv2x(192,160, deconv=True)
        
        self.outconv_6 = BasicConv(160, 32, kernel_size=3,  padding=1)

        if cfg.SEP_LEVEL>1:
            self.deconv5b = Conv2x(160,128, deconv=True)
            self.outconv_5 = BasicConv(128, 32, kernel_size=3,  padding=1)

        if cfg.SEP_LEVEL>2:
            self.deconv4b = Conv2x(128, 96, deconv=True)
            self.outconv_4 = BasicConv(96, 32, kernel_size=3,  padding=1)

        if cfg.SEP_LEVEL>3:
            self.deconv3b = Conv2x(96, 64, deconv=True)
            self.outconv_3 = BasicConv(64, 32, kernel_size=3,  padding=1)

        if cfg.SEP_LEVEL>4:
            self.deconv2b = Conv2x(64, 48, deconv=True)
            self.outconv_2 = BasicConv(48, 32, kernel_size=3,  padding=1)


    def forward(self, x):
        x = self.conv_start(x)
        rem0 = x
        x = self.conv1a(x)
        rem1 = x
        x = self.conv2a(x)
        rem2 = x
        x = self.conv3a(x)
        rem3 = x
        x = self.conv4a(x)
        rem4 = x
        x = self.conv5a(x)
        rem5 = x
        x = self.conv6a(x)
        rem6 = x

        x = self.deconv6a(x,rem5)
        rem5 = x
        x = self.deconv5a(x,rem4)
        rem4 = x
        x = self.deconv4a(x, rem3)
        rem3 = x
        x = self.deconv3a(x, rem2)
        rem2 = x
        x = self.deconv2a(x, rem1)
        rem1 = x
        x = self.deconv1a(x, rem0)
        rem0 = x

        x = self.conv1b(x, rem1)
        rem1 = x
        x = self.conv2b(x, rem2)
        rem2 = x
        x = self.conv3b(x, rem3)
        rem3 = x
        x = self.conv4b(x, rem4)
        rem4 = x
        x = self.conv5b(x, rem5)
        rem5 = x
        x = self.conv6b(x, rem6)

        x = self.deconv6b(x, rem5)
        x6 = self.outconv_6(x)

        if cfg.SEP_LEVEL>1:
            x = self.deconv5b(x, rem4)
            x5 = self.outconv_5(x)
        if cfg.SEP_LEVEL>2:
            x = self.deconv4b(x, rem3)
            x4 = self.outconv_4(x)
        if cfg.SEP_LEVEL>3:
            x = self.deconv3b(x, rem2)
            x3 = self.outconv_3(x)
        if cfg.SEP_LEVEL>4:
            x = self.deconv2b(x, rem1)
            x2 = self.outconv_2(x)

        if cfg.SEP_LEVEL==1:
            return None, None, None, None,None,x6
        elif cfg.SEP_LEVEL==2:
            return None, None, None, None,x5,x6
        elif cfg.SEP_LEVEL==3:
            return None, None, None, x4,x5,x6
        elif cfg.SEP_LEVEL==4:
            return None, None, x3, x4,x5,x6
        elif cfg.SEP_LEVEL==5:
            return None, x2, x3, x4,x5,x6
        else:
            raise NotImplementedError
      

      
class FlowEntropy(nn.Module):
    def __init__(self):
        super(FlowEntropy, self).__init__()

    def forward(self, x):
        x = torch.squeeze(x, 1)
        B,U,V,H,W = x.shape
        x = x.view(B,-1,H,W)

        ###############
        x = F.softmax(x,dim=1).view(B,U,V,H,W)
        global_entropy = (-x*torch.clamp(x,1e-9,1-1e-9).log()).sum(1).sum(1)[:,np.newaxis]
        global_entropy /= np.log(x.shape[1]*x.shape[2])
        return global_entropy

class FlowRegression(nn.Module):
    def __init__(self, maxU, maxV,ratio=None):
        super(FlowRegression, self).__init__()
        self.maxU = maxU
        if ratio is not None: maxV = int(maxV//ratio)
        self.maxV = maxV
        if cfg.TRUNCATED:
            self.trunsize = cfg.TRUNCATED_SIZE
            self.pool3d = nn.MaxPool3d((self.trunsize*2+1,self.trunsize*2+1,1),stride=1,padding=(self.trunsize,self.trunsize,0))

    def forward(self, x):
        assert(x.is_contiguous() == True)
        sizeU = 2*self.maxU+1
        sizeV = 2*self.maxV+1
        x = x.squeeze(1)
        B,_,_,H,W = x.shape

        if cfg.TRUNCATED:
            b,u,v,h,w = x.shape
            oldx = x            
            x = x.view(b,u*v,h,w)
            idx = x.argmax(1)[:,np.newaxis]
            with torch.no_grad():
                mask = torch.zeros((b,u*v,h,w)).type_as(x)
                mask.scatter_(1,idx,1)
                mask = mask.view(b,1,u,v,-1)
                mask = self.pool3d(mask)[:,0].view(b,u,v,h,w)

            ninf = torch.zeros(1).fill_(-np.inf).type_as(oldx).view(1,1,1,1,1).expand(b,u,v,h,w)
            x = torch.where(mask.byte(),oldx,ninf)

        with torch.cuda.device_of(x):
            dispU = torch.reshape(torch.arange(-self.maxU, self.maxU+1,device=torch.cuda.current_device(), dtype=torch.float32),[1,sizeU,1,1,1])
            dispU = dispU.expand(B, -1, sizeV, H,W).contiguous()
            dispU = dispU.view(B,sizeU*sizeV , H,W)

            dispV = torch.reshape(torch.arange(-self.maxV, self.maxV+1,device=torch.cuda.current_device(), dtype=torch.float32),[1,1,sizeV,1,1])
            dispV = dispV.expand(B,sizeU, -1, H,W).contiguous()
            dispV = dispV.view(B,sizeU*sizeV,H,W)
            
        x = x.view(B,sizeU*sizeV,H,W)
        if cfg.FLOW_REG_BY_MAX:
            x = F.softmax(x,dim=1)
        else:
            x = F.softmin(x,dim=1)


        flowU = (x*dispU).sum(dim=1)
        flowV = (x*dispV).sum(dim=1)
        flow  = torch.cat((flowU.unsqueeze(1),flowV.unsqueeze(1)),dim=1)
        return flow


class smooth_cost(nn.Module):
    def __init__(self, md=3):
        super(smooth_cost, self).__init__()
        dimC = (2*md+1)**2
        if cfg.SMOOTH_COST_WITH_THREEMLP:
            self.smooth_layer = nn.Sequential(
            BasicConv(dimC, dimC, kernel_size=1, padding=0),
            BasicConv(dimC, dimC, kernel_size=1, padding=0),
            BasicConv(dimC, dimC, kernel_size=1, padding=0, stride=1, bn=False, relu=False))
        else:
            self.smooth_layer = BasicConv(dimC, dimC, kernel_size=1, padding=0, stride=1, bn=False, relu=False)
        if cfg.SMOOTH_BY_TEMP:
            self.smooth_layer = BasicConv(dimC, 1, kernel_size=1, padding=0, stride=1, bn=False, relu=False)

    def forward(self, x):
        x = x.squeeze(1)
        bs,du,dv,h,w = x.shape
        x = x.view(bs,du*dv,h,w)
        if cfg.SMOOTH_BY_TEMP:
            temp = self.smooth_layer(x)+1e-6
            x = x*temp
        else:
            x = self.smooth_layer(x)
        return x.view(bs,du,dv,h,w).unsqueeze(1)  


class DICL_shallow(nn.Module):
    def __init__(self):
        super(DICL_shallow,self).__init__()

        if cfg.PSP_FEATURE:
            from models.submodule import pspnet_sgm
            self.feature = pspnet_sgm(is_proj=True)
        else:
            self.feature = FeatureGA()

        if cfg.SMOOTH_COST:
            if cfg.SMOOTH_SHARE:
                self.cost_smooth6 = smooth_cost(md=cfg.SEATCH_RANGE[4])
                self.cost_smooth5 = self.cost_smooth6
                self.cost_smooth4 = self.cost_smooth6
                self.cost_smooth3 = self.cost_smooth6
                self.cost_smooth2 = self.cost_smooth6
            else:
                self.cost_smooth6 = smooth_cost(md=cfg.SEATCH_RANGE[4])
                self.cost_smooth5 = smooth_cost(md=cfg.SEATCH_RANGE[3])
                self.cost_smooth4 = smooth_cost(md=cfg.SEATCH_RANGE[2])
                self.cost_smooth3 = smooth_cost(md=cfg.SEATCH_RANGE[1])
                self.cost_smooth2 = smooth_cost(md=cfg.SEATCH_RANGE[0])
        else:
            self.cost_smooth6 = self.cost_smooth5 = self.cost_smooth4 = self.cost_smooth3 = self.cost_smooth2 = None

        self.entropy = FlowEntropy()
        #########


        self.flow6 = FlowRegression(cfg.SEATCH_RANGE[4],cfg.SEATCH_RANGE[4],ratio=cfg.COST6_RATIO)
        self.flow5 = FlowRegression(cfg.SEATCH_RANGE[3],cfg.SEATCH_RANGE[3])
        self.flow4 = FlowRegression(cfg.SEATCH_RANGE[2],cfg.SEATCH_RANGE[2])
        self.flow3 = FlowRegression(cfg.SEATCH_RANGE[1],cfg.SEATCH_RANGE[1])
        self.flow2 = FlowRegression(cfg.SEATCH_RANGE[0],cfg.SEATCH_RANGE[0])

        if cfg.USE_CORR:
            self.matching6 = None
            self.matching5 = None
            self.matching4 = None
            self.matching3 = None
            self.matching2 = None
        else:
            if cfg.SHALLOW_SHARE:
                self.matching6 = MatchingShallow() if not cfg.SHALLOW_Down else MatchingShallow_down()
                self.matching5 = self.matching6
                self.matching4 = self.matching6
                self.matching3 = self.matching6
                self.matching2 = self.matching6
            else:
                self.matching6 = MatchingShallow() if not cfg.SHALLOW_Down else MatchingShallow_down()
                self.matching5 = MatchingShallow() if not cfg.SHALLOW_Down else MatchingShallow_down()
                self.matching4 = MatchingShallow() if not cfg.SHALLOW_Down else MatchingShallow_down()
                self.matching3 = MatchingShallow() if not cfg.SHALLOW_Down else MatchingShallow_down()
                self.matching2 = MatchingShallow() if not cfg.SHALLOW_Down else MatchingShallow_down()
            

        if cfg.CTF_CONTEXT or cfg.CTF_CONTEXT_ONLY_FLOW2:
            self.context2 = nn.Sequential(
                        BasicConv(38,  64, kernel_size=3, padding=1,   dilation=1),
                        BasicConv(64, 128, kernel_size=3, padding=2,   dilation=2),
                        BasicConv(128, 128, kernel_size=3, padding=4,   dilation=4),
                        BasicConv(128, 96 , kernel_size=3, padding=8,   dilation=8),
                        BasicConv(96,  64 , kernel_size=3, padding=16,  dilation=16),
                        BasicConv(64,  32 , kernel_size=3, padding=1,   dilation=1),
                        nn.Conv2d(32,  2  , kernel_size=3, stride=1, padding=1, bias=True))
        if cfg.CTF_CONTEXT:
            self.context3 = nn.Sequential(
                        BasicConv(38,  64, kernel_size=3, padding=1,   dilation=1),
                        BasicConv(64,  128, kernel_size=3, padding=2,   dilation=2),
                        BasicConv(128, 128, kernel_size=3, padding=4,   dilation=4),
                        BasicConv(128, 96 , kernel_size=3, padding=8,   dilation=8),
                        BasicConv(96,  64 , kernel_size=3, padding=16,  dilation=16),
                        BasicConv(64,  32 , kernel_size=3, padding=1,   dilation=1),
                        nn.Conv2d(32,  2  , kernel_size=3, stride=1, padding=1, bias=True))
            self.context4 = nn.Sequential(
                        BasicConv(38,  64, kernel_size=3, padding=1,   dilation=1),
                        BasicConv(64,  128, kernel_size=3, padding=2,   dilation=2),
                        BasicConv(128, 128, kernel_size=3, padding=4,   dilation=4),
                        BasicConv(128, 96 , kernel_size=3, padding=8,   dilation=8),
                        BasicConv(96,  64 , kernel_size=3, padding=16,  dilation=16),
                        BasicConv(64,  32 , kernel_size=3, padding=1,   dilation=1),
                        nn.Conv2d(32,  2  , kernel_size=3, stride=1, padding=1, bias=True))
            self.context5 = nn.Sequential(
                        BasicConv(38,  64, kernel_size=3, padding=1,   dilation=1),
                        BasicConv(64,  128, kernel_size=3, padding=2,   dilation=2),
                        BasicConv(128, 128, kernel_size=3, padding=4,   dilation=4),
                        BasicConv(128, 96 , kernel_size=3, padding=8,   dilation=8),
                        BasicConv(96,  64 , kernel_size=3, padding=16,  dilation=16),
                        BasicConv(64,  32 , kernel_size=3, padding=1,   dilation=1),
                        nn.Conv2d(32,  2  , kernel_size=3, stride=1, padding=1, bias=True))

        if cfg.USE_CONTEXT6:
            self.context6 = nn.Sequential(
                        BasicConv(38,  64, kernel_size=3, padding=1,   dilation=1),
                        BasicConv(64 , 128, kernel_size=3, padding=2,   dilation=2),
                        BasicConv(128, 128, kernel_size=3, padding=4,   dilation=4),
                        BasicConv(128, 96 , kernel_size=3, padding=8,   dilation=8),
                        BasicConv(96,  64 , kernel_size=3, padding=16,  dilation=16),
                        BasicConv(64,  32 , kernel_size=3, padding=1,   dilation=1),
                        nn.Conv2d(32,  2  , kernel_size=3, stride=1, padding=1, bias=True))

        if cfg.FLOW_MASK:
            self.flow_mask_layer = nn.Conv2d(38,2,kernel_size=3,stride=1,padding=1,bias=True)


                 
        for m in self.modules():
            if isinstance(m, (nn.Conv2d)) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        if cfg.SMOOTH_INIT_BY_ID:
            if self.cost_smooth6 is not None:
                nn.init.eye_(self.cost_smooth6.smooth_layer.conv.weight.squeeze(-1).squeeze(-1))
            if self.cost_smooth5 is not None:
                nn.init.eye_(self.cost_smooth5.smooth_layer.conv.weight.squeeze(-1).squeeze(-1))
            if self.cost_smooth4 is not None:
                nn.init.eye_(self.cost_smooth4.smooth_layer.conv.weight.squeeze(-1).squeeze(-1))
            if self.cost_smooth3 is not None:
                nn.init.eye_(self.cost_smooth3.smooth_layer.conv.weight.squeeze(-1).squeeze(-1))
            if self.cost_smooth2 is not None:
                nn.init.eye_(self.cost_smooth2.smooth_layer.conv.weight.squeeze(-1).squeeze(-1))


    def warp(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow

        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow

        """
        flo = flo.type_as(x)

        B, C, H, W = x.size()
        # mesh grid 
        xx = torch.arange(0, W).view(1,-1).repeat(H,1)
        yy = torch.arange(0, H).view(-1,1).repeat(1,W)
        xx = xx.view(1,1,H,W).repeat(B,1,1,1)
        yy = yy.view(1,1,H,W).repeat(B,1,1,1)
        grid = torch.cat((xx,yy),1).type_as(flo)

        if x.is_cuda:
            grid = grid.cuda()
        vgrid = Variable(grid) + flo

        # scale grid to [-1,1] 
        vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:]/max(W-1,1)-1.0
        vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:]/max(H-1,1)-1.0
        vgrid = vgrid.permute(0,2,3,1)        

        
        output = nn.functional.grid_sample(x, vgrid,align_corners=True)
        mask = torch.autograd.Variable(torch.ones(x.size())).type_as(flo).cuda()
        mask = nn.functional.grid_sample(mask, vgrid,align_corners=True)

        
        mask[mask<0.9999] = 0
        mask[mask>0] = 1
        
        return output*mask, mask

    def forward(self, images):
        x = images[:,:3,:,:]
        y = images[:,3:,:,:]
        


        _,x2,x3,x4,x5,x6 = self.feature(x)       
        _,y2,y3,y4,y5,y6 = self.feature(y)

        if not cfg.CTF:
            cost6 = self.compute_cost(x6,y6,self.matching6,cfg.SEATCH_RANGE[4],cfg.SEATCH_RANGE[4],ratio=cfg.COST6_RATIO)

            if self.cost_smooth6 is not None: cost6 = self.cost_smooth6(cost6)
            flow6 = self.flow6(cost6)

            if cfg.SEP_LEVEL>1:
                cost5 = self.compute_cost(x5,y5,self.matching5,cfg.SEATCH_RANGE[3],cfg.SEATCH_RANGE[3])
                if self.cost_smooth5 is not None: cost5 = self.cost_smooth5(cost5)
                flow5 = self.flow5(cost5)  

            if cfg.SEP_LEVEL>2:
                cost4 = self.compute_cost(x4,y4,self.matching4,cfg.SEATCH_RANGE[2],cfg.SEATCH_RANGE[2])
                if self.cost_smooth4 is not None: cost4 = self.cost_smooth4(cost4)
                flow4 = self.flow4(cost4)

            if cfg.SEP_LEVEL>3:
                cost3 = self.compute_cost(x3,y3,self.matching3,cfg.SEATCH_RANGE[1],cfg.SEATCH_RANGE[1])
                if self.cost_smooth3 is not None: cost3 = self.cost_smooth3(cost3)
                flow3 = self.flow3(cost3)

            if cfg.SEP_LEVEL>4:
                cost2 = self.compute_cost(x2,y2,self.matching2,cfg.SEATCH_RANGE[0],cfg.SEATCH_RANGE[0])
                if self.cost_smooth2 is not None: cost2 = self.cost_smooth2(cost2)
                flow2 = self.flow2(cost2)

            if cfg.SMOOTH_LOSS and self.training:
                bs,_,du,dv,ph,pw = cost6.shape
                sm_loss6 = 1-F.softmax(cost6.squeeze(1).view(bs,du*dv,ph,pw),dim=1).max(dim=1)[0].mean()

                bs,_,du,dv,ph,pw = cost5.shape
                sm_loss5 = 1-F.softmax(cost5.squeeze(1).view(bs,du*dv,ph,pw),dim=1).max(dim=1)[0].mean()

                bs,_,du,dv,ph,pw = cost4.shape
                sm_loss4 = 1-F.softmax(cost4.squeeze(1).view(bs,du*dv,ph,pw),dim=1).max(dim=1)[0].mean()

                bs,_,du,dv,ph,pw = cost3.shape
                sm_loss3 = 1-F.softmax(cost3.squeeze(1).view(bs,du*dv,ph,pw),dim=1).max(dim=1)[0].mean()

                bs,_,du,dv,ph,pw = cost2.shape
                sm_loss2 = 1-F.softmax(cost2.squeeze(1).view(bs,du*dv,ph,pw),dim=1).max(dim=1)[0].mean()

                sm_loss = sm_loss6+sm_loss5+sm_loss4+sm_loss3+sm_loss2

                return flow2,flow3,flow4,flow5,flow6,sm_loss


            if cfg.SEP_LEVEL==1:
                return flow6
            elif cfg.SEP_LEVEL==2:
                return flow5, flow6
            elif cfg.SEP_LEVEL==3:
                return flow4,flow5, flow6            
            elif cfg.SEP_LEVEL==4:
                return flow3,flow4,flow5, flow6
            elif cfg.SEP_LEVEL==5:
                return flow2,flow3,flow4,flow5,flow6
            else:
                raise NotImplementedError

        else:
            cost6 = self.compute_cost(x6,y6,self.matching6,cfg.SEATCH_RANGE[4],cfg.SEATCH_RANGE[4],ratio=cfg.COST6_RATIO)
            if self.cost_smooth6 is not None: cost6 = self.cost_smooth6(cost6)
            flow6 = self.flow6(cost6)
            if cfg.CTF_CONTEXT and cfg.USE_CONTEXT6:
                if cfg.SUP_RAW_FLOW: raw_flow6 = flow6
                entro6 = self.entropy(cost6)
                g6 = F.interpolate(images[:,:3,:,:],scale_factor=(1/64), mode='bilinear',align_corners=True,recompute_scale_factor=True)
                feat6 = torch.cat((flow6.detach(),entro6.detach(),x6,g6),dim=1)
                flow6 = flow6 + self.context6(feat6)*cfg.SCALE_CONTEXT6
            up_flow6 = 2.0*F.interpolate(flow6, [x5.shape[2],x5.shape[3]], mode='bilinear',align_corners=True)
            up_flow6 = up_flow6.detach()


            warp5,_ = self.warp(y5,up_flow6)
            cost5 = self.compute_cost(x5,warp5,self.matching5,cfg.SEATCH_RANGE[3],cfg.SEATCH_RANGE[3])
            if self.cost_smooth5 is not None: cost5 = self.cost_smooth5(cost5)
            flow5 = self.flow5(cost5) + up_flow6
            if cfg.CTF_CONTEXT:
                if cfg.SUP_RAW_FLOW: raw_flow5 = flow5
                entro5 = self.entropy(cost5)
                g5 = F.interpolate(images[:,:3,:,:],scale_factor=(1/32), mode='bilinear',align_corners=True,recompute_scale_factor=True)
                feat5 = torch.cat((flow5.detach(),entro5.detach(),x5,g5),dim=1) 
                flow5 = flow5 + self.context5(feat5)*cfg.SCALE_CONTEXT5
            up_flow5 = 2.0*F.interpolate(flow5, [x4.shape[2],x4.shape[3]], mode='bilinear',align_corners=True)
            up_flow5 = up_flow5.detach()

            warp4,_ = self.warp(y4,up_flow5)
            cost4 = self.compute_cost(x4,warp4,self.matching4,cfg.SEATCH_RANGE[2],cfg.SEATCH_RANGE[2])
            if self.cost_smooth4 is not None: cost4 = self.cost_smooth4(cost4)
            flow4 = self.flow4(cost4) + up_flow5
            if cfg.CTF_CONTEXT:
                if cfg.SUP_RAW_FLOW: raw_flow4 = flow4
                entro4 = self.entropy(cost4)
                g4 = F.interpolate(images[:,:3,:,:],scale_factor=(1/16), mode='bilinear',align_corners=True,recompute_scale_factor=True)
                feat4 = torch.cat((flow4.detach(),entro4.detach(),x4,g4),dim=1) 
                flow4 = flow4 + self.context4(feat4)*cfg.SCALE_CONTEXT4
            up_flow4 = 2.0*F.interpolate(flow4, [x3.shape[2],x3.shape[3]], mode='bilinear',align_corners=True)
            up_flow4 = up_flow4.detach()

            warp3,_ = self.warp(y3,up_flow4)
            cost3 = self.compute_cost(x3,warp3,self.matching3,cfg.SEATCH_RANGE[1],cfg.SEATCH_RANGE[1])
            if self.cost_smooth3 is not None: cost3 = self.cost_smooth3(cost3)
            flow3 = self.flow3(cost3) + up_flow4
            if cfg.CTF_CONTEXT:
                if cfg.SUP_RAW_FLOW: raw_flow3 = flow3
                entro3 = self.entropy(cost3)
                g3 = F.interpolate(images[:,:3,:,:],scale_factor=(1/8), mode='bilinear',align_corners=True,recompute_scale_factor=True)
                feat3 = torch.cat((flow3.detach(),entro3.detach(),x3,g3),dim=1) 
                flow3 = flow3 + self.context3(feat3)*cfg.SCALE_CONTEXT3
            up_flow3 = 2.0*F.interpolate(flow3, [x2.shape[2],x2.shape[3]], mode='bilinear',align_corners=True)
            up_flow3 = up_flow3.detach()


            warp2,_ = self.warp(y2,up_flow3)
            cost2 = self.compute_cost(x2,warp2,self.matching2,cfg.SEATCH_RANGE[0],cfg.SEATCH_RANGE[0])
            if self.cost_smooth2 is not None: cost2 = self.cost_smooth2(cost2)
            flow2 = self.flow2(cost2) + up_flow3

            if cfg.CTF_CONTEXT or cfg.CTF_CONTEXT_ONLY_FLOW2:
                if cfg.SUP_RAW_FLOW: raw_flow2 = flow2
                entro2 = self.entropy(cost2)
                g2 = F.interpolate(images[:,:3,:,:],scale_factor=(1/4), mode='bilinear',align_corners=True,recompute_scale_factor=True)
                feat2 = torch.cat((flow2.detach(),entro2.detach(),x2,g2),dim=1) 
                flow2 = flow2 + self.context2(feat2)*cfg.SCALE_CONTEXT2

            if cfg.FLOW_MASK:
                feat_mask = torch.cat((flow2.detach(),entro2.detach(),x2,g2),dim=1) 
                pred_mask = self.flow_mask_layer(feat_mask).sigmoid()

            if cfg.TRAIN_FLOW and self.training:
                if cfg.FLOW_MASK:
                    if cfg.SUP_RAW_FLOW: 
                        return pred_mask, flow2, raw_flow2, flow3, raw_flow3, flow4, raw_flow4, flow5, raw_flow5, flow6,raw_flow6
                    return pred_mask, flow2, flow3, flow4, flow5, flow6                    

                if cfg.SUP_RAW_FLOW: 
                    return flow2, raw_flow2, flow3, raw_flow3, flow4, raw_flow4, flow5, raw_flow5, flow6,raw_flow6
                return flow2, flow3, flow4, flow5, flow6

                
            flow0 = 4.0*F.interpolate(flow2,scale_factor=4, mode='bilinear',align_corners=True,recompute_scale_factor=True)
            entro0 = F.interpolate(entro2,scale_factor=4, mode='bilinear',align_corners=True,recompute_scale_factor=True)
            
            return flow0, entro0


        
    def compute_cost(self, x,y,matchnet,maxU,maxV,ratio=None):
        if ratio is not None: maxV = int(maxV//ratio)


        if cfg.COST_COMP_METHOD =='compute_cost_vcn':
            b,c,height,width = x.shape
            with torch.cuda.device_of(x):
                cost = x.new().resize_(x.size()[0], 1, 2*maxU+1,2*maxV+1, height,  width).zero_()
            for i in range(2*maxU+1):
                ind = i-maxU
                for j in range(2*maxV+1):
                    indd = j-maxV
                    with torch.cuda.device_of(x):
                        temp = x.new().resize_(x.size()[0], c*2, height,  width).zero_()
                    temp[:,:c,max(0,-indd):height-indd,max(0,-ind):width-ind] = x[:,:,max(0,-indd):height-indd,max(0,-ind):width-ind]
                    temp[:,c:,max(0,-indd):height-indd,max(0,-ind):width-ind] = y[:,:,max(0,+indd):height+indd,max(0,ind):width+ind]
                    cost[:, :, i,j,:,:] = matchnet(temp)
        elif cfg.COST_COMP_METHOD =='compute_cost_vcn_together':
            sizeU = 2*maxU+1
            sizeV = 2*maxV+1
            b,c,height,width = x.shape

            if cfg.USE_CORR:
                with torch.cuda.device_of(x):
                    cost = x.new().resize_(x.size()[0], c, 2*maxU+1,2*maxV+1, height,  width).zero_()
                for i in range(2*maxU+1):
                    ind = i-maxU
                    for j in range(2*maxV+1):
                        indd = j-maxV
                        left = x[:,:,max(0,-indd):height-indd,max(0,-ind):width-ind]
                        right = y[:,:,max(0,+indd):height+indd,max(0,ind):width+ind]
                        cost[:, :, i,j,max(0,-indd):height-indd,max(0,-ind):width-ind]   = left*right
                cost = cost.mean(1)
                cost = cost.unsqueeze(1)
            else:
                with torch.cuda.device_of(x):
                    cost = x.new().resize_(x.size()[0], 2*c, 2*maxU+1,2*maxV+1, height,  width).zero_()
                for i in range(2*maxU+1):
                    ind = i-maxU
                    for j in range(2*maxV+1):
                        indd = j-maxV
                        cost[:,:c,i,j,max(0,-indd):height-indd,max(0,-ind):width-ind] = x[:,:,max(0,-indd):height-indd,max(0,-ind):width-ind]
                        cost[:,c:,i,j,max(0,-indd):height-indd,max(0,-ind):width-ind] = y[:,:,max(0,+indd):height+indd,max(0,ind):width+ind]

                if cfg.CTF and cfg.REMOVE_WARP_HOLE:
                    valid_mask = cost[:,c:,...].sum(dim=1)!=0
                    valid_mask = valid_mask.detach()
                    cost = cost*valid_mask.unsqueeze(1).float()

                cost = cost.permute([0,2,3,1,4,5]).contiguous() 
                cost = cost.view(x.size()[0]*sizeU*sizeV,c*2, x.size()[2], x.size()[3])
                cost = matchnet(cost)
                cost = cost.view(x.size()[0],sizeU,sizeV,1, x.size()[2],x.size()[3])
                cost = cost.permute([0,3,1,2,4,5]).contiguous() 
        else:
            raise NotImplementedError
        return cost

def diclflow_net_shallow(data=None):
    model = DICL_shallow()
    if data is not None:
        model.load_state_dict(data['state_dict'], strict=False)
    return model


class MatchingShallow(nn.Module):
    def __init__(self):
        super(MatchingShallow, self).__init__()

        self.match = nn.Sequential(
                        BasicConv(64, 96, kernel_size=3, padding=1,   dilation=1),
                        BasicConv(96, 64, kernel_size=3, padding=2,   dilation=2),
                        BasicConv(64, 32 , kernel_size=3, padding=1,  dilation=1),
                        nn.Conv2d(32, 1  , kernel_size=3, stride=1, padding=1, bias=True))
    def forward(self, x):
        x = self.match(x)
        return x




class MatchingShallow_down(nn.Module):
    def __init__(self):
        super(MatchingShallow_down, self).__init__()

        if cfg.SHALLOW_DOWN_SMALL:
            self.match = nn.Sequential(
                            BasicConv(64, 32, kernel_size=1),
                            BasicConv(32, 64, kernel_size=3, stride=2,    padding=1),
                            BasicConv(64, 64, kernel_size=3, padding=1,   dilation=1),
                            BasicConv(64, 64, kernel_size=3, padding=1,   dilation=1),
                            BasicConv(64, 32, kernel_size=4, padding=1, stride=2, deconv=True),
                            nn.Conv2d(32, 1  , kernel_size=3, stride=1, padding=1, bias=True),)
        else:
            self.match = nn.Sequential(
                            BasicConv(64, 96, kernel_size=3, padding=1,   dilation=1),
                            BasicConv(96, 128, kernel_size=3, stride=2,    padding=1),
                            BasicConv(128, 128, kernel_size=3, padding=1,   dilation=1),
                            BasicConv(128, 64, kernel_size=3, padding=1,   dilation=1),
                            BasicConv(64, 32, kernel_size=4, padding=1, stride=2, deconv=True),
                            nn.Conv2d(32, 1  , kernel_size=3, stride=1, padding=1, bias=True),)

    def forward(self, x):
        x = self.match(x)
        return x

