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


def sample_pose_by_scale(pose_mat,scale):
    # pose_mat: Bx3x4
    cur_angle = utils.matrix2angle(pose_mat[:,:3,:3])
    cur_angle = cur_angle*scale
    cur_rot = utils.angle2matrix(cur_angle)    

    cur_trans = pose_mat[:,:3,-1].unsqueeze(-1)
    cur_trans[:,2,:] = cur_trans[:,2,:]*scale

    pose_sampled = torch.cat((cur_rot,cur_trans),dim=-1)
    return pose_sampled




class PANet(nn.Module):
    def __init__(self, nlabel, mindepth):
        super(PANet, self).__init__()
        self.nlabel = nlabel
        self.mindepth = cfg.MIN_DEPTH

        self.feature_extraction = feature_extraction()

        self.convs = nn.Sequential(
            convtext(33, 128, 3, 1, 1),
            convtext(128, 128, 3, 1, 2),
            convtext(128, 128, 3, 1, 4),
            convtext(128, 96, 3, 1, 8),
            convtext(96, 64, 3, 1, 16),
            convtext(64, 32, 3, 1, 1),
            convtext(32, 1, 3, 1, 1)
        )

        self.dres00 = nn.Sequential(convbn_3d(64, 8, 3, 1, 1),
                                     nn.ReLU(inplace=True),
                                     convbn_3d(8, 16, 3, 1, 1),
                                     nn.ReLU(inplace=True))

        self.dres01 = nn.Sequential(convbn_3d(16, 16, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(16, 16, 1, 1, 0)) 

        self.dres02 = nn.Sequential(convbn_3d(16, 16, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(16, 16, 1, 1, 0))
 
        self.dres03 = nn.Sequential(convbn_3d(16, 16, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(16, 16, 1, 1, 0)) 

        self.classify0 = nn.Sequential(convbn_3d(16, 8, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(8, 1, kernel_size=3, padding=1, stride=1,bias=False))



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
        

        for j, target in enumerate(targets):
            targetimg_fea  = self.feature_extraction(target)
            num_sampled_p = len(np.arange(cfg.SCALE_MIN, cfg.SCALE_MAX,cfg.SCALE_STEP))
            cost = Variable(torch.FloatTensor(num_sampled_p,refimg_fea.size()[0], refimg_fea.size()[1]*2, int(self.nlabel),  refimg_fea.size()[2],  refimg_fea.size()[3]).zero_()).cuda()
            for i in range(int(self.nlabel)):
                depth = torch.div(disp2depth, i+1+1e-16)
                pose_to_sample = pose[:,j]
                feature_list = []
                for scale in np.arange(cfg.SCALE_MIN, cfg.SCALE_MAX,cfg.SCALE_STEP):
                    cur_pose = sample_pose_by_scale(pose_to_sample,scale)
                    targetimg_fea_cur = inverse_warp(targetimg_fea, depth, cur_pose, intrinsics4, intrinsics_inv4)
                    feature_list.append(targetimg_fea_cur.unsqueeze(0))
                targetimg_fea_t = torch.cat(feature_list,dim=0)
                cost[:,:, :refimg_fea.size()[1], i, :,:] = refimg_fea.unsqueeze(0).expand(num_sampled_p,-1,-1,-1,-1)
                cost[:,:, refimg_fea.size()[1]:, i, :,:] = targetimg_fea_t

            cost0 = Variable(torch.FloatTensor(num_sampled_p,refimg_fea.size()[0], 1, int(self.nlabel),  refimg_fea.size()[2],  refimg_fea.size()[3]).zero_()).cuda()
            for num_p in range(num_sampled_p):
                cost_temp = self.dres00(cost[num_p])
                cost_temp = self.dres01(cost_temp) + cost_temp
                cost_temp = self.dres02(cost_temp) + cost_temp 
                cost_temp = self.dres03(cost_temp) + cost_temp 
                cost_temp = self.classify0(cost_temp) 
                cost0[num_p] = cost_temp

            cost0,_ = cost0.max(dim=0)

            if j == 0:
                costs = cost0
            else:
                costs = costs + cost0

        costs = costs/len(targets)

        costss = Variable(torch.FloatTensor(refimg_fea.size()[0], 1, int(self.nlabel),  refimg_fea.size()[2],  refimg_fea.size()[3]).zero_()).cuda()
        for i in range(int(self.nlabel)):
            costt = costs[:, :, i, :, :]
            costss[:, :, i, :, :] = self.convs(torch.cat([refimg_fea, costt],1)) + costt

        costss = F.interpolate(costss, [self.nlabel,ref.size()[2],ref.size()[3]], mode='trilinear')
        costss = torch.squeeze(costss,1)
        pred = F.softmax(costss,dim=1)
        pred = disparityregression(self.nlabel)(pred)

        depth = self.mindepth*self.nlabel/(pred.unsqueeze(1)+1e-16)

        return None, depth
