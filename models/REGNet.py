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


def sample_pose(ref_pose, sample_num):

    std_rot = 0.12
    std_tr = 0.27
    
    batch_size = ref_pose.size()[0]
    ref_trans = ref_pose[:, :3, 3].float()
    trans = Variable(torch.Tensor(np.array(range(int(-sample_num / 2), 1+int(sample_num / 2)))).cuda(),requires_grad=False)
    
    trans = - (trans) / trans[0] * std_tr
    trans1 = trans.view(1, sample_num, 1, 1, 1).repeat(batch_size, 1, sample_num, sample_num,1)  # b * n * n * n * 1
    trans2 = trans.view(1, 1, sample_num, 1, 1).repeat(batch_size, sample_num, 1, sample_num, 1)
    trans3 = trans.view(1, 1, 1, sample_num, 1).repeat(batch_size, sample_num, sample_num, 1, 1)
    trans_vol = torch.cat((trans1, trans2, trans3), 4)  # b * n * n * n * 3

    trans_volume = ref_trans.view(batch_size, 1, 1, 1, 3) + trans_vol

    rot = Variable(torch.Tensor(np.array(range(int(-sample_num / 2), 1+int(sample_num / 2)))).cuda(),requires_grad=False)
    rot = - rot  / rot[0] * std_rot
    rot1 = rot.view(1, sample_num, 1, 1, 1).repeat(batch_size, 1, sample_num, sample_num,
                                                    1)  # b * n * n * n * 1
    rot2 = rot.view(1, 1, sample_num, 1, 1).repeat(batch_size, sample_num, 1, sample_num, 1)
    rot3 = rot.view(1, 1, 1, sample_num, 1).repeat(batch_size, sample_num, sample_num, 1, 1)
    angle_vol = torch.cat((rot1, rot2, rot3), 4)  # b * n * n * n * 3
    angle_matrix = utils.angle2matrix(angle_vol)  # b * n * n * n * 3 * 3
    rot_volume = torch.matmul(angle_matrix,ref_pose[:, :3, :3].view(batch_size, 1, 1, 1, 3,3).repeat(1, sample_num, sample_num, sample_num, 1, 1))
    
    rot_volume = rot_volume.view(batch_size,-1,3,3)
    trans_volume = trans_volume.view(batch_size,-1,3).unsqueeze(-1)
    sampled_poses = torch.cat((rot_volume,trans_volume),dim=-1)
    return sampled_poses


class REGNet(nn.Module):
    def __init__(self, nlabel, mindepth):
        super(REGNet, self).__init__()
        self.nlabel = nlabel
        self.mindepth = cfg.MIN_DEPTH

        self.sample_num = 5


        self.feature_extraction = feature_extraction()

        if cfg.IND_CONTEXT:
            self.context_net = feature_extraction()

        self.convs = nn.Sequential(
            convtext(33, 128, 3, 1, 1),
            convtext(128, 128, 3, 1, 2),
            convtext(128, 128, 3, 1, 4),
            convtext(128, 96, 3, 1, 8),
            convtext(96, 64, 3, 1, 16),
            convtext(64, 32, 3, 1, 1),
            convtext(32, 1, 3, 1, 1)
        )

        self.num_sampled_p = (self.sample_num)**3

        self.posecnn0 = nn.Sequential(convbn_3d(self.num_sampled_p, 32, 3, 1, 1),
                                     nn.ReLU(inplace=True),
                                     convbn_3d(32, 32, 3, 1, 1),
                                     nn.ReLU(inplace=True))

        self.posecnn1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1)) 

        self.posecnn2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1))
 
        self.posecnn3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1)) 

        self.posecnn4 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1)) 
 
        self.predict = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1,bias=False))

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
        num_sampled_p = self.num_sampled_p
        intrinsics4 = intrinsics4.unsqueeze(1).expand(-1,num_sampled_p,-1,-1).reshape(batch_size*num_sampled_p,intrinsics4.shape[-2],intrinsics4.shape[-1])
        intrinsics_inv4 = intrinsics_inv4.unsqueeze(1).expand(-1,num_sampled_p,-1,-1).reshape(batch_size*num_sampled_p,intrinsics_inv4.shape[-2],intrinsics_inv4.shape[-1])


        for j, target in enumerate(targets):
            targetimg_fea  = self.feature_extraction(target)
            targetimg_fea = targetimg_fea.unsqueeze(1).expand(-1,num_sampled_p,-1,-1,-1).reshape(batch_size*num_sampled_p,targetimg_fea.shape[-3],targetimg_fea.shape[-2],targetimg_fea.shape[-1])
            cost = Variable(torch.FloatTensor(refimg_fea.size()[0],num_sampled_p, int(self.nlabel),  refimg_fea.size()[2],  refimg_fea.size()[3]).zero_()).cuda()
            for i in range(int(self.nlabel)):
                depth = torch.div(disp2depth, i+1+1e-16)
                pose_to_sample = pose[:,j]
                sampled_poses = sample_pose(pose_to_sample,self.sample_num)
                sampled_poses = sampled_poses.reshape(batch_size*num_sampled_p,sampled_poses.shape[-2],sampled_poses.shape[-1])
                depth = depth.unsqueeze(1).expand(-1,num_sampled_p,-1,-1).reshape(batch_size*num_sampled_p,depth.shape[-2],depth.shape[-1])
                
                targetimg_fea_t = inverse_warp(targetimg_fea, depth, sampled_poses, intrinsics4, intrinsics_inv4)
                
                targetimg_fea_t = targetimg_fea_t.reshape(batch_size,num_sampled_p,targetimg_fea_t.shape[-3],targetimg_fea_t.shape[-2],targetimg_fea_t.shape[-1])

                cost[:,:,i,:,:] = (refimg_fea.unsqueeze(1) * targetimg_fea_t).mean(dim=2)

            cost = F.leaky_relu(cost, 0.1,inplace=True)
            cost0 = self.posecnn0(cost)
            cost0 = self.posecnn1(cost0) + cost0
            cost0 = self.posecnn2(cost0) + cost0 
            cost0 = self.posecnn3(cost0) + cost0 
            cost0 = self.posecnn4(cost0) + cost0
            cost0 = self.predict(cost0)

            if j == 0:
                costs = cost0
            else:
                costs = costs + cost0

        costs = costs/len(targets)

        ########


        costss = Variable(torch.FloatTensor(refimg_fea.size()[0], 1, int(self.nlabel),  refimg_fea.size()[2],  refimg_fea.size()[3]).zero_()).cuda()
        
        if cfg.IND_CONTEXT:
            refimg_fea     = self.context_net(ref)

        for i in range(int(self.nlabel)):
            costt = costs[:, :, i, :, :]
            costss[:, :, i, :, :] = self.convs(torch.cat([refimg_fea, costt],1)) + costt
        ########

        costs = F.interpolate(costs, [self.nlabel,ref.size()[2],ref.size()[3]], mode='trilinear')
        costs = torch.squeeze(costs,1)
        pred0 = F.softmax(costs,dim=1)
        pred0 = disparityregression(self.nlabel)(pred0)
        depth_init = self.mindepth*self.nlabel/(pred0.unsqueeze(1)+1e-16)

        costss = F.interpolate(costss, [self.nlabel,ref.size()[2],ref.size()[3]], mode='trilinear')
        costss = torch.squeeze(costss,1)
        pred = F.softmax(costss,dim=1)
        pred = disparityregression(self.nlabel)(pred)

        depth = self.mindepth*self.nlabel/(pred.unsqueeze(1)+1e-16)

        return depth_init, depth



