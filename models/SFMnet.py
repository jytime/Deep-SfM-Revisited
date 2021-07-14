from __future__ import print_function
import torch
import numpy as np
import cv2
import utils
import time
from models import DICL_shallow
from models.RAFT.core.raft import RAFT

from models import PSNet as PSNet
import essential_matrix
from epipolar_utils import *
from models.PoseNet import ResNet,Bottleneck, PlainPose
from lib.config import cfg, cfg_from_file, save_config_to_file

# for speed analysis
global time_dict
time_dict = {}

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

class SFMnet(torch.nn.Module):
    def __init__(self, nlabel=64, min_depth=0.5):
        super(SFMnet,self).__init__()
        ##### Hyperparameters #####
        self.delta = 0.001
        self.alpha = 0.0 
        self.maxreps = 200 
        self.min_matches = cfg.min_matches
        self.ransac_iter = cfg.ransac_iter
        self.ransac_threshold = cfg.ransac_threshold

        self.nlabel = nlabel

        # set minimum depth, to avoid numerical errors
        self.min_depth = min_depth

        # choose your flow estimator, default as DICL_shallow
        if cfg.FLOW_EST =='RAFT':
            self.flow_estimator = RAFT()
        elif cfg.FLOW_EST =='DICL':
            self.flow_estimator = DICL_shallow()
        else:
            raise NotImplementedError

        # choose your depth estimator, default as PSNet
        if cfg.DEPTH_EST=='PSNET':
            self.depth_estimator = PSNet(nlabel,min_depth)
        elif cfg.DEPTH_EST=='CVP':
            from models.CVPMVS import CVPMVS
            self.depth_estimator = CVPMVS()
        elif cfg.DEPTH_EST=='PANET':
            from models.PANet import PANet
            self.depth_estimator = PANet(nlabel,min_depth)
        elif cfg.DEPTH_EST=='REGNET':
            from models.REGNet import REGNet
            self.depth_estimator = REGNet(nlabel,min_depth)
        elif cfg.DEPTH_EST=='REG2D':
            from models.REG2D import REG2D
            self.depth_estimator = REG2D(nlabel,min_depth)
        elif cfg.DEPTH_EST=='DISPNET':
            from models.DISPNET import DISPNET
            self.depth_estimator = DISPNET(nlabel,min_depth)
        else:
            raise NotImplementedError

        # sift feature extraction
        self.sift = cv2.xfeatures2d.SIFT_create()
        self.surf = cv2.xfeatures2d.SURF_create()

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

        if cfg.POSE_EST =='POSENET':
            if cfg.POSE_NET_TYPE  == 'plain':
                self.posenet = PlainPose()
            elif cfg.POSE_NET_TYPE == 'res':
                self.posenet = ResNet(Bottleneck,[3, 4, 6, 3])
            else:
                raise NotImplementedError


    def forward(self, ref, target, intrinsic, pose_gt=None, pred_pose=None, use_gt_pose=False, 
                    h_side=None,w_side=None,logger=None,depth_gt=None,img_path=None):
        
        # if TRAIN_FLOW, we only conduct flow estimation
        if self.training and cfg.TRAIN_FLOW:
            flow_outputs  = self.flow_estimator(torch.cat((ref, target),dim=1)) 
            return flow_outputs

        intrinsic_gpu = intrinsic.float().cuda()                # Bx3x3
        intrinsic_inv_gpu = torch.inverse(intrinsic_gpu)        # Bx3x3

        # Default, if do not use ground truth poses for training 
        if use_gt_pose == False:

            # if predict relative poses online, or use pre-saved poses
            if cfg.PRED_POSE_ONLINE:

                # flow estimation
                with autocast(enabled=cfg.MIXED_PREC):
                    flow_start = time.time()
                    flow_2D, conf  = self.flow_estimator(torch.cat((ref, target),dim=1)) 

                # recover image shape, to avoid meaningless flow matches
                if h_side is not None or w_side is not None:
                    flow_2D = flow_2D[:,:,:h_side,:w_side]
                    try:  
                        conf = conf[:,:,:h_side,:w_side]   
                    except:
                        conf = conf

                # choose how to estimate pose, by RANSAC or deep regression
                if cfg.POSE_EST =='RANSAC':
                    # some inputs are left for possible visualization or debug, plz ignore them if not
                    # return:   Pose matrix             Bx3x4
                    #           Essential matrix        Bx3x3
                    P_mat,E_mat = self.pose_by_ransac(flow_2D,ref,target,intrinsic_inv_gpu,
                                                        h_side,w_side,pose_gt =pose_gt,img_path=img_path)
                    rot_and_trans = None
                elif cfg.POSE_EST =='POSENET':
                    rot_and_trans = self.posenet(flow_2D,conf,ref,target)
                    P_mat = RT2Pose(rot_and_trans)
                else:
                    raise NotImplementedError
            else:
                # use ground truth poses, for oracle experiments
                P_mat = pred_pose; E_mat = None; flow_2D = None

            # if only use gt scales, for oracle experiments                
            if cfg.PRED_POSE_GT_SCALE:
                scale = torch.norm(pose_gt[:,:3, 3],dim=1, p=2).unsqueeze(1).unsqueeze(1)
                P_mat[:,:,-1:] = P_mat[:,:,-1:]*scale

            P_mat.unsqueeze_(1)
        else:
            E_mat = None
            P_mat = pose_gt.clone()
            if cfg.GT_POSE_NORMALIZED:
                scale = torch.norm(P_mat[:,:3, 3],dim=1, p=2).unsqueeze(1).unsqueeze(1)
                P_mat[:,:,-1:] = P_mat[:,:,-1:]/scale
            P_mat.unsqueeze_(1)
            flow_2D = torch.zeros([ref.shape[0],2,ref.shape[2],ref.shape[3]]).cuda().type_as(ref)
        
        if cfg.RECORD_POSE or (cfg.RECORD_POSE_EVAL and not self.training):
            return P_mat, flow_2D

        if h_side is not None or w_side is not None:
            ref = ref[:,:,:h_side,:w_side]; target = target[:,:,:h_side,:w_side]

        # depth prediction
        with autocast(enabled=cfg.MIXED_PREC):
            depth_start = time.time()
            depth_init, depth = self.depth_estimator(ref, [target], P_mat, intrinsic_gpu, intrinsic_inv_gpu,pose_gt=pose_gt,depth_gt=depth_gt,E_mat=E_mat)

        if self.training:
            # rot_and_trans is only used for pose deep regression
            # otherwise, it is None
            return flow_2D, P_mat, depth, depth_init, rot_and_trans
        return flow_2D, P_mat, depth, time_dict



    def pose_by_ransac(self, flow_2D, ref, target, intrinsic_inv_gpu,
                            h_side, w_side, pose_gt=False, img_path=None):

        b, _, h, w = flow_2D.size()
        coord1_flow_2D, coord2_flow_2D = flow2coord(flow_2D)    # Bx3x(H*W) 
        coord1_flow_2D = coord1_flow_2D.view(b,3,h,w)        
        coord2_flow_2D = coord2_flow_2D.view(b,3,h,w)    
        margin = 10                 # avoid corner case


        E_mat = torch.zeros(b, 3, 3).cuda()                     # Bx3x3
        P_mat = torch.zeros(b, 3, 4).cuda()                     # Bx3x4

        PTS1=[]; PTS2=[];                                       # point list

        # process the frames of each batch
        for b_cv in range(b):
            # convert images to cv2 style
            if h_side is not None or w_side is not None:
                ref_cv =ref[b_cv,:,:h_side,:w_side].cpu().numpy().transpose(1,2,0)[:,:,::-1]
                tar_cv =target[b_cv,:,:h_side,:w_side].cpu().numpy().transpose(1,2,0)[:,:,::-1]
            else:
                ref_cv =ref[b_cv].cpu().numpy().transpose(1,2,0)[:,:,::-1]
                tar_cv =target[b_cv].cpu().numpy().transpose(1,2,0)[:,:,::-1]
            ref_cv = (ref_cv*0.5+0.5)*255; tar_cv = (tar_cv*0.5+0.5)*255

            # detect key points           
            kp1, des1 = self.sift.detectAndCompute(ref_cv.astype(np.uint8),None)
            kp2, des2 = self.sift.detectAndCompute(tar_cv.astype(np.uint8),None)
            if len(kp1)<self.min_matches or len(kp2)<self.min_matches:
                # surf generally has more kps than sift
                kp1, des1 = self.surf.detectAndCompute(ref_cv.astype(np.uint8),None)
                kp2, des2 = self.surf.detectAndCompute(tar_cv.astype(np.uint8),None)

            try:
                # filter out some key points
                matches = self.flann.knnMatch(des1,des2,k=2)
                good = []; pts1 = []; pts2 = []
                for i,(m,n) in enumerate(matches):
                    if m.distance < 0.8*n.distance: good.append(m); pts1.append(kp1[m.queryIdx].pt); pts2.append(kp2[m.trainIdx].pt)
                
                # degengrade if not existing good matches
                if len(good)<self.min_matches:
                    good = [];pts1 = [];pts2 = []
                    for i,(m,n) in enumerate(matches):
                        good.append(m); pts1.append(kp1[m.queryIdx].pt); pts2.append(kp2[m.trainIdx].pt)
                pts1 = np.array(pts1); PTS1.append(pts1);pts2 = np.array(pts2); PTS2.append(pts2);
            except:
                # if cannot find corresponding pairs, ignore this sift mask 
                PTS1.append([None]); PTS2.append([None])

        assert len(PTS1)==b

        for batch in range(b):
            if cfg.SIFT_POSE:
                # if directly use SIFT matches
                pts1 = PTS1[batch]; pts2 = PTS2[batch]
                coord1_sift_2D = torch.FloatTensor(pts1)
                coord2_sift_2D = torch.FloatTensor(pts2)
                coord1_flow_2D_norm_i = torch.cat((coord1_sift_2D,torch.ones(len(coord1_sift_2D),1)),dim=1).unsqueeze(0).to(coord1_flow_2D.device).permute(0,2,1)
                coord2_flow_2D_norm_i = torch.cat((coord2_sift_2D,torch.ones(len(coord2_sift_2D),1)),dim=1).unsqueeze(0).to(coord1_flow_2D.device).permute(0,2,1)
            else:
                # check the number of matches
                if len(PTS1[batch])<self.min_matches or len(PTS2[batch])<self.min_matches:
                    coord1_flow_2D_norm_i = coord1_flow_2D[batch,:,margin:-margin,margin:-margin].contiguous().view(3,-1).unsqueeze(0)
                    coord2_flow_2D_norm_i = coord2_flow_2D[batch,:,margin:-margin,margin:-margin].contiguous().view(3,-1).unsqueeze(0)                
                else:
                    if cfg.SAMPLE_SP:
                        # conduct interpolation
                        pts1 = torch.from_numpy(PTS1[batch]).to(coord1_flow_2D.device).type_as(coord1_flow_2D)
                        B, C, H, W = coord1_flow_2D.size()
                        pts1[:,0] = 2.0*pts1[:,0]/max(W-1,1)-1.0;pts1[:,1] = 2.0*pts1[:,1]/max(H-1,1)-1.0
                        coord1_flow_2D_norm_i = F.grid_sample(coord1_flow_2D[batch].unsqueeze(0), pts1.unsqueeze(0).unsqueeze(-2),align_corners=True).squeeze(-1)
                        coord2_flow_2D_norm_i = F.grid_sample(coord2_flow_2D[batch].unsqueeze(0), pts1.unsqueeze(0).unsqueeze(-2),align_corners=True).squeeze(-1)
                    else:
                        # default choice
                        pts1 = np.int32(np.round(PTS1[batch]))
                        coord1_flow_2D_norm_i = coord1_flow_2D[batch,:,pts1[:,1],pts1[:,0]].unsqueeze(0)
                        coord2_flow_2D_norm_i = coord2_flow_2D[batch,:,pts1[:,1],pts1[:,0]].unsqueeze(0)

            intrinsic_inv_gpu_i = intrinsic_inv_gpu[batch].unsqueeze(0)

            # projection by intrinsic matrix
            coord1_flow_2D_norm_i = torch.bmm(intrinsic_inv_gpu_i, coord1_flow_2D_norm_i) 
            coord2_flow_2D_norm_i = torch.bmm(intrinsic_inv_gpu_i, coord2_flow_2D_norm_i) 
            # reshape coordinates            
            coord1_flow_2D_norm_i = coord1_flow_2D_norm_i.transpose(1,2)[0,:,:2].contiguous()
            coord2_flow_2D_norm_i = coord2_flow_2D_norm_i.transpose(1,2)[0,:,:2].contiguous()
            
            with autocast(enabled=False):
                # GPU-accelerated RANSAC five-point algorithm
                E_i, P_i, F_i,inlier_num = compute_P_matrix_ransac(coord1_flow_2D_norm_i.detach(), coord2_flow_2D_norm_i.detach(), 
                                                                intrinsic_inv_gpu[batch,:,:], self.delta, self.alpha, self.maxreps, 
                                                                len(coord1_flow_2D_norm_i), len(coord1_flow_2D_norm_i), 
                                                                self.ransac_iter, self.ransac_threshold) 

            E_mat[batch, :, :] = E_i.detach(); P_mat[batch, :, :] = P_i.detach()

        return P_mat, E_mat




#############################################  Utility #############################################

def check_tensor(tensor):
    return torch.isinf(tensor).any() or torch.isnan(tensor).any()

def Pose2RT(pose_mat):
    # pose_mat [B,3,4]
    # return : (d1,d2,d3,t1,t2,t3)
    cur_angle = utils.matrix2angle(pose_mat[:,:3,:3])
    cur_trans = pose_mat[:,:3,-1]
    return torch.cat((cur_angle,cur_trans),dim=-1)

def RT2Pose(RT):
    # RT (d1,d2,d3,t1,t2,t3)
    # return : [B,3,4]
    cur_rot = utils.angle2matrix(RT[:,:3])   
    cur_trans = RT[:,3:].unsqueeze(-1)
    return torch.cat((cur_rot,cur_trans),dim=-1)

def flow2coord(flow):
    """
    Generate flat homogeneous coordinates 1 and 2 from optical flow. 
    Args:
        flow: bx2xhxw, torch.float32
    Output:
        coord1_hom: bx3x(h*w)
        coord2_hom: bx3x(h*w)
    """
    b, _, h, w = flow.size()
    coord1 = torch.zeros_like(flow)
    coord1[:,0,:,:] += torch.arange(w).float().cuda()
    coord1[:,1,:,:] += torch.arange(h).float().cuda()[:, None]
    coord2 = coord1 + flow
    coord1_flat = coord1.reshape(b, 2, h*w)
    coord2_flat = coord2.reshape(b, 2, h*w)

    ones = torch.ones((b, 1, h*w), dtype=torch.float32).cuda()
    coord1_hom = torch.cat((coord1_flat, ones), dim=1)
    coord2_hom = torch.cat((coord2_flat, ones), dim=1)
    return coord1_hom, coord2_hom

def coord2flow(coord1, coord2, b, h, w):
    """
    Convert flat homogeneous coordinates 1 and 2 to optical flow. 
    Args:
        coord1: bx3x(h*w)
        coord2: bx3x(h*w)
    Output:
        flow: bx2xhxw, torch.float32
    """
    coord1 = coord1[:, :2, :] # bx2x(h*w)
    coord2 = coord2[:, :2, :] # bx2x(h*w)
    flow = coord2 - coord1
    flow = flow.reshape(b, 2, h, w)
    return flow

