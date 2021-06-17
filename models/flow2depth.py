import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def Flow2Depth(R, T, initial_flow, K_mat):
    '''
    Convert optical flow to depth values, 
    with the help of rotation, translation, and intrinsic matrix
    '''
    B, _, H, W = initial_flow.shape

    K = K_mat.cpu().numpy()
    Ki = np.linalg.inv(K)

    pixel_dir = np.zeros((H, W, 3))
    for i in range(H):
        for j in range(W):
            pixel_dir[i, j, :] = np.dot(Ki, np.array([j, i, 1]))
    pixel_dir = pixel_dir.reshape(H * W, 3, 1).astype(np.float32)

    pixel_loc = np.zeros((H, W, 2))
    for i in range(H):
        for j in range(W):
            pixel_loc[i, j, :] = [j,i]
    pixel_loc = pixel_loc.reshape(1, H, W, 2).astype(np.float32)

    pixel_dir = torch.from_numpy(pixel_dir).cuda()
    pixel_loc = torch.from_numpy(pixel_loc).cuda()

    first_part = torch.matmul(K_mat, R).view(B, 1, 3, 3)
    first_part = torch.matmul(first_part, pixel_dir)

    second_part = torch.matmul(K_mat, T.unsqueeze(-1)).view(B, 1, 3, 1)
    second_part = second_part.expand(-1, H*W, -1, -1)
    
    output = first_part+second_part
    output = output.view(B,-1,H,W)

    return output[:,-1,:,:]
