from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
from models.submodule import *
from models.inverse_warp import *
import pdb
from lib.config import cfg, cfg_from_file, save_config_to_file
import utils

import cv2 as cv
import cv2
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




class DISPNET(nn.Module):
    def __init__(self, nlabel, mindepth):
        super(DISPNET, self).__init__()
        self.nlabel = nlabel
        self.mindepth = cfg.MIN_DEPTH

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


    def projection_warp(self,refimg_fea,depth_hypo,intrinsics,intrinsics_inv,E_mat,pose_mat):

        b, _, h, w = refimg_fea.size()

        # construct current_pixel_coords: (B, 3, HW); for each pixel, (u,v,1)
        i_range = Variable(torch.arange(0, h).view(1, h, 1).expand(1,h,w)).type_as(depth_hypo)  # [1, H, W]
        j_range = Variable(torch.arange(0, w).view(1, 1, w).expand(1,h,w)).type_as(depth_hypo)  # [1, H, W]
        ones = Variable(torch.ones(1,h,w)).type_as(depth_hypo)
        pixel_coords = torch.stack((j_range, i_range, ones), dim=1)  # [1, 3, H, W]
        current_pixel_coords = pixel_coords[:,:,:h,:w].expand(b,3,h,w).contiguous().view(b, 3, -1).cuda()


        # compute Fundamental matrix and Epipolar lines
        if E_mat is not None:
            Kinv_E = torch.bmm(intrinsics_inv.transpose(1,2),E_mat)
            F_mat = torch.bmm(Kinv_E,intrinsics_inv)
            epi_line = torch.bmm(F_mat, current_pixel_coords).transpose(1,2)


        # [X,Y,Z] = dK-1[u1,v1,1]
        cam_coords_unit = intrinsics_inv.bmm(current_pixel_coords).view(b, 3, h, w)
        cam_coords = cam_coords_unit*depth_hypo     # 
        cam_coords_flat = cam_coords.view(b, 3, -1)

        # R, T
        proj_cam_to_src_pixel = intrinsics.bmm(pose_mat)
        proj_c2p_rot = proj_cam_to_src_pixel[:,:,:3]
        proj_c2p_tr = proj_cam_to_src_pixel[:,:,-1:]

        # R[X,Y,Z] +T = lambda K-1 [u2,v2,1]
        pcoords = proj_c2p_rot.bmm(cam_coords_flat)
        pcoords = pcoords + proj_c2p_tr

        # normalize the results
        X = pcoords[:, 0];Y = pcoords[:, 1];Z = pcoords[:, 2].clamp(min=1e-3)
        X_norm = (X / Z); Y_norm = (Y / Z)

        # check_error = epi_line[:,:,0]*init_anchors[:,:,0]+epi_line[:,:,1]*init_anchors[:,:,1] + epi_line[:,:,2]
        init_anchors = torch.stack([X_norm, Y_norm], dim=2)   #.view(b,h,w,2)

        if E_mat is not None:
            slope = -(epi_line[...,0]/epi_line[...,1])
            ones_vector =  torch.ones(slope.shape).cuda().type_as(slope)
            move = torch.cat((ones_vector.unsqueeze(-1),slope.unsqueeze(-1)),dim=-1)
            move_norm = torch.norm(move, p=2, dim=-1).type_as(move)
            move_normalized = move/move_norm.unsqueeze(-1)
        else:
            move_normalized = None


        return current_pixel_coords, init_anchors, move_normalized

    def triangulation(self,left_uv1, rigth_uv,intrinsics,intrinsics_inv,pose_mat,u_base=True,depth_hypo=None):
        # left_uv1: (B,3,HW)
        # rigth_uv: (B,HW,2)
        R_mat = pose_mat[:,:,:3]
        T_mat = pose_mat[:,:,-1:]
        KT = torch.bmm(intrinsics,T_mat)

        Rp = torch.bmm(R_mat,left_uv1)
        Rp_with_K = torch.bmm(intrinsics,R_mat).bmm(intrinsics_inv).bmm(left_uv1)

        if u_base:
            # u2*[kt]2-[kt0]
            # --------------
            # [Rp]0 - u2 *[Rp]2
            numerator = rigth_uv[...,0]*KT[:,2]-KT[:,0]
            denominator = Rp_with_K[:,0,:] - rigth_uv[...,0]*Rp_with_K[:,2,:]
            depth  = numerator/(denominator+1e-5)

        else:
            numerator = rigth_uv[...,1]*KT[:,2]-KT[:,1]
            denominator = Rp_with_K[:,1,:] - rigth_uv[...,1]*Rp_with_K[:,2,:]
            depth  = numerator/(denominator+1e-5)

        
        depth = depth.clamp(min=0.0,max=80)
        return depth


    def forward(self, ref, targets, pose, intrinsics, intrinsics_inv,pose_gt=None,depth_gt=None,E_mat=None):

        intrinsics4 = intrinsics.clone()
        intrinsics_inv4 = intrinsics_inv.clone()
        intrinsics4[:,:2,:] = intrinsics4[:,:2,:] / 4
        intrinsics_inv4[:,:2,:2] = intrinsics_inv4[:,:2,:2] * 4

        refimg_fea     = self.feature_extraction(ref)

        disp_hypo = Variable(torch.ones(refimg_fea.size(0), refimg_fea.size(2), refimg_fea.size(3))).cuda().type_as(refimg_fea)
        
        depth_hypo = (disp_hypo.unsqueeze(1))*20
        ###############
        # compute initialized anchors with depth = 20
        assert pose.shape[1] == 1
        pose_mat = pose[:,0].cuda()

        raw_pixel_coords, init_anchors, move_normalized = self.projection_warp(refimg_fea,depth_hypo,intrinsics4,intrinsics_inv4,E_mat,pose_mat)

        b, _, h, w = refimg_fea.size()

        ###############

        num_disp = int(2*self.nlabel+1)

        for j, target in enumerate(targets):
            targetimg_fea  = self.feature_extraction(target)
            cost = Variable(torch.FloatTensor(refimg_fea.size()[0], refimg_fea.size()[1]*2, num_disp,  refimg_fea.size()[2],  refimg_fea.size()[3]).zero_()).cuda()
            for i in range(num_disp):
                cur_disp = i-self.nlabel
                cur_uv = init_anchors + move_normalized*cur_disp
                cur_u_norm = 2*(cur_uv[...,0])/(w-1) - 1  # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
                cur_v_norm = 2*(cur_uv[...,1])/(h-1) - 1  # Idem [B, H*W]
                X_mask = ((cur_u_norm > 1)+(cur_u_norm < -1)).detach(); cur_u_norm[X_mask] = 2
                Y_mask = ((cur_v_norm > 1)+(cur_v_norm < -1)).detach(); cur_v_norm[Y_mask] = 2
                cur_pixel_coords = torch.stack([cur_u_norm, cur_v_norm], dim=2).view(b,h,w,2)
                targetimg_fea_t = torch.nn.functional.grid_sample(targetimg_fea, cur_pixel_coords, padding_mode='zeros',align_corners=True)
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

        costss = Variable(torch.FloatTensor(refimg_fea.size()[0], 1, num_disp,  refimg_fea.size()[2],  refimg_fea.size()[3]).zero_()).cuda()
        
        for i in range(num_disp):
            costt = costs[:, :, i, :, :]
            costss[:, :, i, :, :] = self.convs(torch.cat([refimg_fea, costt],1)) + costt

        costss = torch.squeeze(costss,1)
        pred = F.softmax(costss,dim=1)
        pred = disparityregression_lam(self.nlabel)(pred)

        pred_right_uv = init_anchors + move_normalized*pred.view(b,-1).unsqueeze(-1)
        
        depth = self.triangulation(raw_pixel_coords,pred_right_uv,intrinsics4,intrinsics_inv4,pose_mat)
        
        check_nan = torch.isnan(depth)
        depth[check_nan] = 0
        depth = depth.view(b,1,h,w)
        
        if cfg.PSNET_DEP_CONTEXT:
            # up_feat = F.interpolate(refimg_fea,[ref.size()[2],ref.size()[3]] , mode='bilinear',align_corners=True)
            ref_g = F.interpolate(ref,scale_factor=(1/4), mode='bilinear',align_corners=True,recompute_scale_factor=True)
            dep_feat = torch.cat((depth.detach(),refimg_fea,ref_g),dim=1)
            depth_out = self.dep_convs(dep_feat).type_as(depth) + depth

            depth = F.interpolate(depth,[ref.size()[2],ref.size()[3]] , mode='bilinear',align_corners=True)
            depth_out = F.interpolate(depth_out,[ref.size()[2],ref.size()[3]] , mode='bilinear',align_corners=True)

            return depth, depth_out

        depth = F.interpolate(depth,[ref.size()[2],ref.size()[3]] , mode='bilinear',align_corners=True)

        return None, depth

def drawlines_only(img1,lines,pts):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    
    r,c,_ = img1.shape
    pts = pts[:,:2]
    # img1 = cv.cvtColor(img1,cv.COLOR_GRAY2BGR)
    # img2 = cv.cvtColor(img2,cv.COLOR_GRAY2BGR)
    for r,pt1 in zip(lines,pts):
        color = tuple(np.random.randint(0,255,3).tolist())
        # import pdb;pdb.set_trace()
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv.circle(img1,tuple(pt1),3,color,-1)
    return img1

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    # import pdb;pdb.set_trace()
    r,c,_ = img1.shape
    # img1 = cv.cvtColor(img1,cv.COLOR_GRAY2BGR)
    # img2 = cv.cvtColor(img2,cv.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv.line(img1, (x0,y0), (x1,y1), color,1)
        # img1 = cv.circle(img1,tuple(pt1),5,color,-1)
        # img2 = cv.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2
