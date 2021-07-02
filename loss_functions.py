import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from scipy.stats import norm
import numpy as np
import cv2
import matplotlib.image as mpimg
from models.pose2flow import inverse_warp_im, pose2flow
from lib.config import cfg, cfg_from_file, save_config_to_file


# For Flow Training

def MultiScale_UP(output,target,loss_type='L1',weight=[1.,0.5,0.25],valid_range=None,removezero=False,extra_mask=None):
    loss = 0
    loss_list = []
    b, _, h, w = target.size()

    if (type(output) is not tuple) and (type(output) is not set):
        output = {output}
    
    for i, cur_output in enumerate(output):
        realflow = F.interpolate(cur_output, (h,w), mode='bilinear', align_corners=True)
        realflow[:,0,:,:] = realflow[:,0,:,:]*(w/cur_output.shape[3])
        realflow[:,1,:,:] = realflow[:,1,:,:]*(h/cur_output.shape[2])

        with torch.no_grad():
            if i==0: epe = realEPE(realflow,target,extra_mask=extra_mask)

        if loss_type=='L2':
            lossvalue = torch.norm(realflow-target,p=2,dim=1)
        elif loss_type=='robust':
            lossvalue = ((realflow-target).abs().sum(dim=1)+1e-8)
            lossvalue = lossvalue**0.4
        elif loss_type=='L1':
            lossvalue = (realflow-target).abs().sum(dim=1)
        else:
            raise NotImplementedError

        if cfg.USE_VALID_RANGE and valid_range is not None:
            with torch.no_grad():
                mask = (target[:,0,:,:].abs()<=valid_range[i][1]) & (target[:,1,:,:].abs()<=valid_range[i][0])
        else:
            with torch.no_grad():
                mask = torch.ones(target[:,0,:,:].shape).type_as(target)

        lossvalue = lossvalue*mask.float() 
        if extra_mask is not None:
            val = extra_mask > 0
            lossvalue = lossvalue[val]
            cur_loss = lossvalue.mean()*weight[i]
            assert lossvalue.shape[0] == extra_mask.sum()
        else:
            cur_loss = lossvalue.mean()*weight[i]

        loss+=cur_loss
        loss_list.append(cur_loss)

    loss = loss/len(output)

    return loss,loss_list,epe


def photometric_reconstruction_loss_Charbonnier(flo_fw_est, flo_bw_est, I1, I2,t):  
    left_est  = warp(I2, flo_fw_est)  
    right_est = warp(I1, flo_bw_est)
    left_occl,right_occl = compute_occlusion(flo_fw_est,flo_bw_est,t)
    left_select,right_select = compute_occlusion(flo_fw_est,flo_bw_est,1)

    left_occl.detach_();right_occl.detach_();left_select.detach_();right_select.detach_()
    
    left_occl_gy = left_occl[:,:,:, 1:]
    left_occl_gx = left_occl[:,:,1:, :]
    
    right_occl_gy = right_occl[:,:,:, 1:]
    right_occl_gx = right_occl[:,:,1:, :]
    
    left_est_gx, left_est_gy = gradient_im(left_est)
    right_est_gx, right_est_gy = gradient_im(right_est)
    left_gx, left_gy = gradient_im(I1)
    right_gx, right_gy = gradient_im(I2)    
    
    leftl1_loss = (charbonnier_penalty(left_est - I1)*left_occl).mean()/(left_occl.mean()+1e-3)
    rightl1_loss = (charbonnier_penalty(right_est - I2)*right_occl).mean()/(right_occl.mean()+1e-3)
    
    left_gxl1_loss = (charbonnier_penalty(left_est_gx - left_gx)*left_occl_gx).mean()/(left_occl_gx.mean()+1e-3)
    left_gyl1_loss = (charbonnier_penalty(left_est_gy - left_gy)*left_occl_gy).mean()/(left_occl_gy.mean()+1e-3)
    right_gxl1_loss = (charbonnier_penalty(right_est_gx - right_gx)*right_occl_gx).mean()/(right_occl_gx.mean()+1e-3)
    right_gxl1_loss = (charbonnier_penalty(right_est_gy - right_gy)*right_occl_gy).mean()/(right_occl_gy.mean()+1e-3)
    
    census_loss = ternary_loss(I1, left_est, left_occl) + ternary_loss(I2, right_est, right_occl)
    
    reconstruction_loss = 0.5*census_loss + leftl1_loss + rightl1_loss + left_gxl1_loss + left_gyl1_loss + right_gxl1_loss + right_gxl1_loss
    return reconstruction_loss, left_est, right_est, left_occl.byte(), right_occl.byte(), left_select.byte(), right_select.byte()
  

def weighted_smooth_depth_loss(pred_disp,im):
    loss = 0
    weight = 0.5

    Ix, Iy = gradient_im(im)
    Ix2, IxIy = gradient_im(Ix)
    IyIx, Iy2 = gradient_im(Iy)
    
    weight_x = torch.exp(-weight*torch.mean(Ix.abs(),1))
    weight_y = torch.exp(-weight*torch.mean(Iy.abs(),1))
    weight_x2 = torch.exp(-weight*torch.mean(Ix2.abs(),1))
    weight_y2 = torch.exp(-weight*torch.mean(Iy2.abs(),1))
    weight_xy = torch.exp(-weight*torch.mean(IxIy.abs(),1))
    weight_yx = torch.exp(-weight*torch.mean(IyIx.abs(),1))

    dx, dy = gradient_depth(pred_disp)
    dx2, dxdy = gradient_depth(dx)
    dydx, dy2 = gradient_depth(dy)   
    
    dx = dx * weight_x
    dy = dy * weight_y
    dx2 = dx2 * weight_x2
    dy2 = dy2 * weight_y2
    dxdy = dxdy * weight_xy
    dydx = dydx * weight_yx
    
    loss = (dx.abs().mean() + dy.abs().mean() + dx2.abs().mean() + dxdy.abs().mean() + dydx.abs().mean() + dy2.abs().mean())
        
    return loss


def ternary_loss(img1, img2_warped, mask, max_distence=1):
    patch_size = 2 * max_distence +1
    def ternary_transform(img):
        intensities = (0.5*(img[:,0,:,:] + img[:,1,:,:] + img[:,2,:,:])/3 + 0.5)*255
        intensities = intensities.unsqueeze_(1)
        out_channels = patch_size * patch_size
        weights = torch.eye(out_channels).view((out_channels,1,patch_size, patch_size)).cuda()       
        patches = F.conv2d(intensities, weights, None, 1, patch_size//2)        
        transf = patches - intensities
        transf_norm = transf / torch.sqrt(0.81 + transf**2)
        return transf_norm
      
    def hamming_distance(t1,t2):
      dist = (t1 -t2)**2
      dist_norm = dist / (0.1 + dist)
      dist_sum = torch.sum(dist_norm, 1, keepdim=True)
      return dist_sum
    
    t1 = ternary_transform(img1)
    t2 = ternary_transform(img2_warped)
    dist = hamming_distance(t1,t2)
    
    transform_mask = create_mask(mask,[[max_distence, max_distence], [max_distence, max_distence]])    
    tmp = mask * transform_mask
    
    return (charbonnier_penalty(dist)*tmp).mean()/(tmp.mean()+1e-3)



############# Utils #############


def compute_occlusion(disp_left,disp_right,t):
    disp_right2left = warp(disp_right, disp_left) 
    disp_left2right = warp(disp_left, disp_right) 
    tmp_left = (disp_left + disp_right2left).abs()
    tmp_right = (disp_right + disp_left2right).abs()
    mask_left = (tmp_left[:,0,:,:] < t) & (tmp_left[:,1,:,:] < t)
    mask_left = mask_left.unsqueeze(1)
    mask_right = (tmp_right[:,0,:,:] < t) & (tmp_right[:,1,:,:] < t)
    mask_right = mask_right.unsqueeze(1)
    mask_left = mask_left.float()
    mask_right = mask_right.float()
    return mask_left, mask_right
  

def warp(x, flo):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow

    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow

    """
    B, C, H, W = x.size()
        # mesh grid 
    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    grid = torch.cat((xx,yy),1).float()

    if x.is_cuda:
        grid = grid.cuda()
    vgrid = Variable(grid) + flo

        # scale grid to [-1,1] 
    vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:]/max(W-1,1)-1.0
    vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:]/max(H-1,1)-1.0

    vgrid = vgrid.permute(0,2,3,1)        
    output = nn.functional.grid_sample(x, vgrid,align_corners=True)
    mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
    mask = nn.functional.grid_sample(mask, vgrid,align_corners=True)

    mask[mask<0.9999] = 0
    mask[mask>0] = 1
        
    return output*mask

def charbonnier_penalty(err):
    return torch.sqrt(err**2 + 0.001**2)


def gradient(pred):
    D_dy = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    D_dx = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    return D_dx, D_dy

def gradient_im(image):
    D_dy = image[:, :, :, 1:] - image[:,:,  :, :-1]
    D_dx = image[:, :, 1:, :] - image[:,:,  :-1, :]
    return D_dx, D_dy

def gradient_depth(pred):
    D_dy = pred[:, :, 1:] - pred[:, :, :-1]
    D_dx = pred[:, 1:, :] - pred[:, :-1, :]
    return D_dx, D_dy

def random_select_points(x,y,x_,y_,samples=10):
    idx=torch.randperm(x.shape[0])
    x=x[idx[:samples],:]
    y=y[idx[:samples],:]
    x_=x_[idx[:samples],:]
    y_=y_[idx[:samples],:]
    return x,y,x_,y_

def create_mask(tensor, paddings):
    shape = tensor.size()
    inner_width = shape[2] - (paddings[0][0] + paddings[0][1])
    inner_height = shape[3] - (paddings[1][0] + paddings[1][1])
    inner = torch.ones([inner_width, inner_height]).cuda()
    
    mask2d = F.pad(inner, [paddings[0][0],paddings[0][1],paddings[1][0],paddings[0][1]])
    mask3d = mask2d.unsqueeze_(0).repeat(shape[0],1,1)
    mask4d = mask3d.unsqueeze_(1)
    return mask4d.detach_()
  
def realEPE(output, target, sparse=False, valid_range=None,extra_mask=None):
    b, _, h, w = target.size()
    upsampled_output = output
    if cfg.USE_VALID_RANGE and valid_range is not None:
        mask = (target[:,0,:,:].abs()<=valid_range[1]) & (target[:,1,:,:].abs()<=valid_range[0])
        mask = mask.unsqueeze(1).expand(-1,2,-1,-1).float()
        upsampled_output = upsampled_output*mask
        target = target*mask
    return EPE(upsampled_output, target, sparse, mean=True,extra_mask=extra_mask)
  


def EPE(input_flow, target_flow, sparse=False, mean=True,extra_mask=None):
    EPE_map = torch.norm(target_flow-input_flow,2,1)
    batch_size = EPE_map.size(0)

    if sparse:
        # invalid flow is defined with both flow coordinates to be exactly 0
        mask = (target_flow[:,0] == 0) & (target_flow[:,1] == 0)
        EPE_map = EPE_map[~mask]

    if extra_mask is not None:
        EPE_map = EPE_map[extra_mask.bool()]

    if mean:
        return EPE_map.mean()
    else:
        return EPE_map.sum()/batch_size

def EPEd(input_depth, target_depth, sparse=False, mean=True):
    #EPE_map = torch.norm(input_depth-target_depth,2,1)
    EPE_map = torch.abs(input_depth - target_depth) / target_depth
    batch_size = EPE_map.size(0)
    if sparse:
        # invalid flow is defined with both flow coordinates to be exactly 0
        mask = (target_depth == 0)

        EPE_map = EPE_map[~mask]
    if mean:
        return EPE_map.mean()
    else:
        return EPE_map.sum()/batch_size



def realEPEd(output, target, sparse=False):
    return EPEd(output, target, sparse, mean=True)

def check_tuple(input_t):
    if type(input_t) not in [tuple, list]:
        input_t = [input_t]
    return input_t


