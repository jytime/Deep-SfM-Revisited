# Credit: https://github.com/sunghoonim/DPSNet

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

class PSNet(nn.Module):
    def __init__(self, nlabel, mindepth):
        super(PSNet, self).__init__()
        self.nlabel = nlabel
        self.mindepth = cfg.MIN_DEPTH

        self.number = 0
        self.feature_extraction = feature_extraction()

        if cfg.IND_CONTEXT:
            self.context_net = feature_extraction()

        if cfg.PSNET_DEP_CONTEXT:
            self.dep_convs = nn.Sequential(
                convtext(36, 128, 3, 1, 1),
                convtext(128, 128, 3, 1, 2),
                convtext(128, 128, 3, 1, 4),
                convtext(128, 96, 3, 1, 8),
                convtext(96, 64, 3, 1, 16),
                convtext(64, 32, 3, 1, 1),
                convtext(32, 1, 3, 1, 1))


        self.convs = nn.Sequential(
            convtext(33, 128, 3, 1, 1),
            convtext(128, 128, 3, 1, 2),
            convtext(128, 128, 3, 1, 4),
            convtext(128, 96, 3, 1, 8),
            convtext(96, 64, 3, 1, 16),
            convtext(64, 32, 3, 1, 1),
            convtext(32, 1, 3, 1, 1))

        if cfg.COST_BY_COLOR:
            self.COST_BY_COLOR_layer = convtext(4, 128, 3, 1, 1)
        if cfg.COST_BY_COLOR_WITH_FEAT:
            self.COST_BY_COLOR_layer = convtext(36, 128, 3, 1, 1)


        self.dres0 = nn.Sequential(convbn_3d(64, 32, 3, 1, 1),
                                     nn.ReLU(inplace=True),
                                     convbn_3d(32, 32, 3, 1, 1),
                                     nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1)) 

        self.dres2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1))
 
        self.dres3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1)) 

        self.dres4 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1)) 
 
        self.classify = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1,bias=False))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

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


    def forward(self, ref, targets, pose, intrinsics, intrinsics_inv,pose_gt=None,depth_gt=None,E_mat=None):

        intrinsics4 = intrinsics.clone()
        intrinsics_inv4 = intrinsics_inv.clone()
        intrinsics4[:,:2,:] = intrinsics4[:,:2,:] / 4
        intrinsics_inv4[:,:2,:2] = intrinsics_inv4[:,:2,:2] * 4
        
        if cfg.RESCALE_DEPTH:
            pose[:,0,:,-1:] = pose[:,0,:,-1:]*cfg.NORM_TARGET

        refimg_fea     = self.feature_extraction(ref)

        ones_vec = Variable(torch.ones(refimg_fea.size(0), refimg_fea.size(2), refimg_fea.size(3))).cuda()
        
        disp2depth =  ones_vec * self.mindepth * self.nlabel

        for j, target in enumerate(targets):
            targetimg_fea  = self.feature_extraction(target)
            cost = Variable(torch.FloatTensor(refimg_fea.size()[0], refimg_fea.size()[1]*2, int(self.nlabel),  refimg_fea.size()[2],  refimg_fea.size()[3]).zero_()).cuda()
            

            for i in range(int(self.nlabel)):
                if cfg.PREDICT_BY_DEPTH:
                    depth = ones_vec*(i+1)*self.mindepth
                else:
                    depth = torch.div(disp2depth, i+1+1e-16)

                targetimg_fea_t = inverse_warp(targetimg_fea, depth, pose[:,j], intrinsics4, intrinsics_inv4)
                cost[:, :refimg_fea.size()[1], i, :,:] = refimg_fea
                cost[:, refimg_fea.size()[1]:, i, :,:] = targetimg_fea_t

            cost = cost.contiguous()
            cost0 = self.dres0(cost)
            cost0 = self.dres1(cost0) + cost0
            cost0 = self.dres2(cost0) + cost0 
            cost0 = self.dres3(cost0) + cost0 
            cost0 = self.dres4(cost0) + cost0
            cost0 = self.classify(cost0)
            if j == 0:
                costs = cost0
            else:
                costs = costs + cost0

        costs = costs/len(targets)

        ########

        if cfg.PSNET_CONTEXT:
            costss = Variable(torch.FloatTensor(refimg_fea.size()[0], 1, int(self.nlabel),  refimg_fea.size()[2],  refimg_fea.size()[3]).zero_()).cuda()
            if cfg.IND_CONTEXT:
                refimg_fea     = self.context_net(ref)
            for i in range(int(self.nlabel)):
                costt = costs[:, :, i, :, :]

                if cfg.COST_BY_COLOR or cfg.COST_BY_COLOR_WITH_FEAT:
                    g = F.interpolate(ref,[refimg_fea.shape[2],refimg_fea.shape[3]], mode='bilinear',align_corners=True)
                    if cfg.COST_BY_COLOR_WITH_FEAT:
                        temp = self.COST_BY_COLOR_layer(torch.cat([refimg_fea,costt,g],1))
                    else:
                        temp = self.COST_BY_COLOR_layer(torch.cat([costt,g],1))
                    costss[:, :, i, :, :] = self.convs[2:](temp) + costt
                else:
                    costss[:, :, i, :, :] = self.convs(torch.cat([refimg_fea, costt],1)) + costt
        else:
            costss = costs

        costs = F.interpolate(costs, [self.nlabel,ref.size()[2],ref.size()[3]], mode='trilinear')
        costs = torch.squeeze(costs,1)
        pred0 = F.softmax(costs,dim=1)
        
        # predict depth or disparity?
        # the answer: similar performance
        if cfg.PREDICT_BY_DEPTH:
            pred0 = depthregression(self.nlabel)(pred0)
            depth_init = pred0.unsqueeze(1)
        else:
            pred0 = disparityregression(self.nlabel)(pred0)
            depth_init = self.mindepth*self.nlabel/(pred0.unsqueeze(1)+1e-16)

        costss = F.interpolate(costss, [self.nlabel,ref.size()[2],ref.size()[3]], mode='trilinear')
        costss = torch.squeeze(costss,1)
        pred = F.softmax(costss,dim=1)
        
        if cfg.PREDICT_BY_DEPTH:
            pred = depthregression(self.nlabel)(pred)
            depth = pred.unsqueeze(1)*self.mindepth
        else:
            pred = disparityregression(self.nlabel)(pred)
            depth = self.mindepth*self.nlabel/(pred.unsqueeze(1)+1e-16)

        if cfg.PSNET_DEP_CONTEXT:
            up_feat = F.interpolate(refimg_fea,[ref.size()[2],ref.size()[3]] , mode='bilinear',align_corners=True)
            dep_feat = torch.cat((depth.detach(),up_feat,ref),dim=1)
            depth_out = self.dep_convs(dep_feat) + depth
            return depth, depth_out

        return depth_init, depth



