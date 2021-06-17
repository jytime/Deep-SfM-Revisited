from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
from models.submodule import *
from models.inverse_warp import inverse_warp
import pdb
from lib.config import cfg, cfg_from_file, save_config_to_file
import utils

def convtext(in_planes, out_planes, kernel_size = 3, stride = 1, dilation = 1):

    if cfg.CONTEXT_BN:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size = kernel_size, stride = stride, dilation = dilation, padding = ((kernel_size - 1) * dilation) // 2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True))
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size = kernel_size, stride = stride, dilation = dilation, padding = ((kernel_size - 1) * dilation) // 2, bias=False),
            nn.ReLU(inplace=True))


def convbn(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):   
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, 
                        padding=padding, dilation=dilation, bias=True),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1,inplace=True))

class REG2D(nn.Module):
    def __init__(self, nlabel, mindepth):
        super(REG2D, self).__init__()
        self.nlabel = nlabel
        self.mindepth = cfg.MIN_DEPTH

        self.feature_extraction = feature_extraction()

        dd = np.cumsum([128,128,96,64,32])

        od = self.nlabel
        self.conv0 = convbn(od,      128, kernel_size=3, stride=1)
        self.conv1 = convbn(od+dd[0],128, kernel_size=3, stride=1)
        self.conv2 = convbn(od+dd[1],96,  kernel_size=3, stride=1)
        self.conv3 = convbn(od+dd[2],64,  kernel_size=3, stride=1)
        self.conv4 = convbn(od+dd[3],32,  kernel_size=3, stride=1) 
        self.predict = nn.Conv2d(od+dd[4],1,kernel_size=3,stride=1,padding=1,bias=True)


        self.context = nn.Sequential(
                    convbn(135, 128, kernel_size=3, padding=1,   dilation=1),
                    convbn(128, 128, kernel_size=3, padding=1,   dilation=1),
                    convbn(128, 128, kernel_size=3, padding=1,   dilation=1),
                    convbn(128, 128, kernel_size=3, padding=2,   dilation=2),
                    convbn(128, 128, kernel_size=3, padding=4,   dilation=4),
                    convbn(128, 96 , kernel_size=3, padding=8,   dilation=8),
                    convbn(96,  64 , kernel_size=3, padding=16,  dilation=16),
                    convbn(64,  32 , kernel_size=3, padding=1,   dilation=1),
                    nn.Conv2d(32,  1  , kernel_size=3, stride=1, padding=1, bias=True))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1]*m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()


    def forward(self, ref, targets, pose, intrinsics, intrinsics_inv):

        intrinsics4 = intrinsics.clone()
        intrinsics_inv4 = intrinsics_inv.clone()
        intrinsics4[:,:2,:] = intrinsics4[:,:2,:] / 4
        intrinsics_inv4[:,:2,:2] = intrinsics_inv4[:,:2,:2] * 4

        refimg_fea     = self.feature_extraction(ref)

        disp2depth = Variable(torch.ones(refimg_fea.size(0), refimg_fea.size(2), refimg_fea.size(3))).cuda() * self.mindepth * self.nlabel
        

        batch_size  = ref.size(0)

        assert len(targets) == 1
        for j, target in enumerate(targets):
            targetimg_fea  = self.feature_extraction(target)
            cost = Variable(torch.FloatTensor(batch_size, int(self.nlabel),  refimg_fea.size()[2],  refimg_fea.size()[3]).zero_()).cuda()
            for i in range(int(self.nlabel)):
                depth = torch.div(disp2depth, i+1+1e-16)
                targetimg_fea_t = inverse_warp(targetimg_fea, depth, pose[:,j], intrinsics4, intrinsics_inv4)
                cost[:,i] = (refimg_fea*targetimg_fea_t).mean(dim=1)

            cost = F.leaky_relu(cost, 0.1,inplace=True)


            x = torch.cat((self.conv0(cost), cost),1)
            x = torch.cat((self.conv1(x), x),1)
            x = torch.cat((self.conv2(x), x),1)
            x = torch.cat((self.conv3(x), x),1)
            x = torch.cat((self.conv4(x), x),1)
            depth_init = self.predict(x).squeeze(1)

            pose_to_sample = pose[:,j]

            scales = torch.from_numpy(np.arange(0.5, 1.6,0.1)).view(1,-1,1,1).cuda()
            num_sampled_p = scales.shape[1]
            sampled_poses = pose_to_sample.unsqueeze(1).expand(-1,num_sampled_p,-1,-1).contiguous()
            sampled_poses[...,-1:] = sampled_poses[...,-1:]*scales
            
            offset_num = 9
            delta = (offset_num-1)/2
            std = 0.5

            cost = Variable(torch.FloatTensor(batch_size, offset_num*num_sampled_p,  refimg_fea.size()[2],  refimg_fea.size()[3]).zero_()).cuda()

            temp_num = 0
            for offset in range(offset_num):
                depth_offset = (offset-delta)*std
                depth_hypo = depth_init.detach()+depth_offset
                for sampled_p in range(num_sampled_p):
                    targetimg_fea_t = inverse_warp(targetimg_fea, depth_hypo, sampled_poses[:,sampled_p], intrinsics4, intrinsics_inv4)
                    cost[:,temp_num] = (refimg_fea*targetimg_fea_t).mean(dim=1)
                    temp_num = temp_num+1

            ref_down = F.interpolate(ref,scale_factor=(1/4), mode='bilinear',align_corners=True,recompute_scale_factor=True)
            x = torch.cat((cost,refimg_fea,depth_init.unsqueeze(1).detach(),ref_down),dim=1)
            depth_init = depth_init.unsqueeze(1)
            depth = self.context(x) + depth_init.detach()



            depth = F.interpolate(depth,scale_factor=4, mode='bilinear',align_corners=True,recompute_scale_factor=True)
            depth_init = F.interpolate(depth_init,scale_factor=4, mode='bilinear',align_corners=True,recompute_scale_factor=True)
        
        return depth_init, depth



