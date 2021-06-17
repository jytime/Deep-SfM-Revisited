import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class CorrelationLayer(nn.Module):
    def __init__(self, md=4):
        """
        this is the normal correlation layer used in PWC-Net 
        input: md --- maximum displacement (for correlation. default: 4)
        """
        super(CorrelationLayer, self).__init__()
        self.md = md
        self.out = (self.md*2+1)**2

    def forward(self, input0, input1):
        B,C,H,W = input0.shape
        output = torch.zeros((B,self.out,H,W), device = input0.device)
        big1 = torch.zeros((B,C,H+2*self.md,W+2*self.md), device = input0.device)
        big1[:,:,self.md:self.md+H,self.md:self.md+W] = input1
        index = 0
        for row_i in range(-1*self.md, self.md+1):
            for col_i in range(-1*self.md, self.md+1):
                dot = input0 * big1[:,:,self.md+row_i:self.md+row_i+H,self.md+col_i:self.md+col_i+W]
                output[:, index, :, :] = torch.sum(dot, dim = 1)
                index += 1
        output = output / C
        return output

class EpipolarCorrelationLayer(nn.Module):
    def __init__(self, maxd, mind, H, W):
        """
        this is the proposed correlation layer that takes epipolar constraints into account
        input: maxd --- displacement alone epipolar line (for correlation. default: 4)
               mind --- displacement perpendicular to epipolar line (for correlation. default: 4)
        """
        super(EpipolarCorrelationLayer, self).__init__()
        self.maxd = maxd
        self.mind = mind
        self.out = len(self.maxd) * len(self.mind)
        self.H = H
        self.W = W

        # the intrinsic is the same as DeMoN
        K = np.zeros((3,3))
        K[0,0] = 0.89115971 * self.W
        K[0,2] = 0.5 * self.W
        K[1,1] = 1.18821287 * self.H
        K[1, 2] = 0.5 * self.H
        K[2,2] = 1.0
        Ki = np.linalg.inv(K)

        pixel_dir = np.zeros((self.H, self.W, 3))
        for i in range(self.H):
            for j in range(self.W):
                pixel_dir[i, j, :] = np.dot(Ki, np.array([j, i, 1]))
        pixel_dir = pixel_dir.reshape(self.H * self.W, 3, 1).astype(np.float32)

        pixel_loc = np.zeros((self.H, self.W, 2))
        for i in range(self.H):
            for j in range(self.W):
                pixel_loc[i, j, :] = [j,i]
        pixel_loc = pixel_loc.reshape(1, self.H, self.W, 2).astype(np.float32)

        self.K = torch.from_numpy(K.reshape(1,3,3).astype(np.float32)).cuda()
        self.Ki = torch.from_numpy(Ki.reshape(1,3,3).astype(np.float32)).cuda()
        self.pixel_dir = torch.from_numpy(pixel_dir).cuda()
        self.pixel_loc = torch.from_numpy(pixel_loc).cuda()

    def forward(self, imgL, imgR, R, T, initial_flow):
        '''
        input R: N*3*3
        input T: N*3*1
        '''
        B, C, H, W = imgL.shape
        output = torch.zeros((B, self.out, H, W), device=imgL.device)
        first_part = torch.matmul(self.K, R).view(B, 1, 3, 3)
        first_part = torch.matmul(first_part, self.pixel_dir)
        second_part = torch.matmul(self.K, T).view(B, 1, 3, 1)
        if (first_part != first_part).any():
            print("first_part has nan!!")
            print("R:")
            print(R.detach().cpu().numpy())
        first_part_depth = first_part[:, :, 2:3, :].clone()
        first_part_depth[torch.abs(first_part_depth) < 1e-6] = 1e-6
        end_point = first_part[:, :, :2, :] / first_part_depth
        if (end_point != end_point).any():
            print("end_point has nan!!")
        sapce_point = first_part * 10.0 + second_part
        sapce_point_depth = sapce_point[:, :, 2:3, :].clone()
        sapce_point_depth[torch.abs(sapce_point_depth) < 1e-6] = 1e-6
        project_point = sapce_point[:, :, :2, :] / sapce_point_depth
        if (project_point != project_point).any():
            print("project_point has nan!!")
        para_dir = F.normalize(project_point - end_point, dim = 2)
        if (para_dir != para_dir).any():
            print("para_dir has nan!!")
        perp_dir = para_dir[:,:,[1,0],:]
        perp_dir[:,:,0,:] *= -1.0
        para_dir = para_dir.view(B, H, W, 2)
        perp_dir = perp_dir.view(B, H, W, 2)
        end_point = end_point.view(B, H, W, 2)
        flow_point = self.pixel_loc + initial_flow.permute(0,2,3,1)
        index = 0
        # get the initial point
        nearest_k = (flow_point - end_point) * para_dir
        nearest_k = torch.sum(nearest_k, dim = 3, keepdim = True)
        initial_loc = end_point + nearest_k * para_dir

        for para_i in self.maxd:
            for perp_i in self.mind:
                grid = initial_loc + para_i * para_dir + perp_i + perp_dir
                grid[:,:,:,0] = 2.0 * grid[:,:,:,0]/(self.W-1) - 1.0
                grid[:,:,:,1] = 2.0 * grid[:,:,:,1]/(self.H-1) - 1.0
                sampled = F.grid_sample(imgR, grid)
                dot = imgL * sampled
                output[:, index, :, :] = torch.sum(dot, dim = 1)
                index += 1
        output = output / C
        para_dir = para_dir.permute(0, 3, 1, 2)
        initial_loc = initial_loc.permute(0, 3, 1, 2)
        epipolar_flow = initial_loc - self.pixel_loc.permute(0, 3, 1, 2)
        if (output!=output).any():
            print("output has NaN")
        if (para_dir!=para_dir).any():
            print("para_dir has NaN")
        result = torch.cat([epipolar_flow, para_dir, output], dim = 1)
        return result