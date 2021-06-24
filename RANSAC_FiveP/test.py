import torch
import numpy as np
from torch.autograd import Variable
import essential_matrix
from pdb import set_trace as st

import time

def euler2mat(angle):
    """Convert euler angles to rotation matrix.
     Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174
    Args:
        angle: rotation angle along 3 axis (in radians) -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the euler angles -- size = [B, 3, 3]
    """
    B = angle.size(0)
    x, y, z = angle[:,0], angle[:,1], angle[:,2]

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zeros = z.detach()*0
    ones = zeros.detach()+1
    zmat = torch.stack([cosz, -sinz, zeros,
                        sinz,  cosz, zeros,
                        zeros, zeros,  ones], dim=1).reshape(B, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zeros,  siny,
                        zeros,  ones, zeros,
                        -siny, zeros,  cosy], dim=1).reshape(B, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([ones, zeros, zeros,
                        zeros,  cosx, -sinx,
                        zeros,  sinx,  cosx], dim=1).reshape(B, 3, 3)

    rotMat = xmat @ ymat @ zmat
    return rotMat

def skew(x):
    """ Create a skew-symmetric matrix from vector
    Args:
      x: vector size - size = 3
    Returns:
      skew-symmetric matrix of vector x - size = 3 x 3 
    """
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])

def errorEMat(E1, E2):
    """ Compute L2 error between two E matrices.
    Args:
      E1: E matrix 1, 3x3 torch Tensor
      E2: E matrix 2, 3x3 torch Tensor
    Returns:
      error: scalar
    """
    E1_normalized = E1 / E1[2][2];
    E2_normalized = E2 / E2[2][2];
    return torch.norm(E1_normalized - E2_normalized)

def errorResidual(E, qs, qps):
    """ Computes residual error.
    Args:
      E: Essential matrix, 3x3
      qs: Point set 1, bxnx2, b=1 
      qps: Point set 2, bxnx2, b=1
    """
    nPoints = qs.size(0) * qs.size(1)
    ones = torch.ones((qs.size(0), qs.size(1), 1), dtype=torch.float64)
    qs = torch.cat((qs, ones), dim=2)
    qps = torch.cat((qps, ones), dim=2)
    error_sum = 0
    for i in range(nPoints):
        x = qs[0][i].unsqueeze(1)
        xp = qps[0][i].unsqueeze(1)
        Ex = torch.mm(E, x)
        error = torch.dot(xp.squeeze(), Ex.squeeze()) / torch.sqrt(Ex[0][0]**2 + Ex[1][0]**2)
        error_sum = error_sum + error.abs()
    result = error_sum/nPoints

    return result

##### Hyperparameters to be set #####
torch.manual_seed(1)
np.random.seed(1)
torch.set_default_dtype(torch.float64)


# input points
nPoints = 78600
outlier_frac = 0.1
noise_std = 0.0001 # not in pixels!
# ransac
num_test_points = 10
num_ransac_test_points = 1000
num_ransac_iterations = 1
inlier_threshold = 0.05
# iterative reweighted least-squares
delta = 0.01 
alpha = 0.0
maxreps = 10
#####################################

# Generate rotation and translation
angle = torch.rand(1, 3)
rotMat = euler2mat(angle).squeeze()
# R^T = R^{-1}
# print(torch.mm(rotMat.transpose(0, 1), rotMat))
# determinant == 1
# print(np.linalg.det(rotMat.numpy()))
trans = torch.rand(3, 1)
pMat = torch.cat((rotMat, trans), 1)
E_gt = torch.mm(torch.from_numpy(skew(trans.squeeze().numpy())), torch.squeeze(rotMat))
print("Ground-truth E matrix")
print(E_gt)

# Generate 3D points
Xs = torch.rand(3, nPoints) + 0.1
PXs = torch.mm(rotMat, Xs) + trans 

# Create 2D points
qs = torch.empty((2, 0))
qps = torch.empty((2, 0))

q = torch.zeros((2, 1))
qp = torch.zeros((2, 1))
for i in range(nPoints):
    q[0, :] = Xs[0, i] / Xs[2, i]
    q[1, :] = Xs[1, i] / Xs[2, i]
    qp[0, :] = PXs[0, i] / PXs[2, i]
    qp[1, :] = PXs[1, i] / PXs[2, i]

    if torch.norm(q) < 2 and PXs[2, i] > 0:
        qs = torch.cat((qs, q), dim=1)
        qps = torch.cat((qps, qp), dim=1)

# Add noise and create outliers
# noise - applied in camera coordinate space (not image space)
qs = qs.transpose(0, 1).unsqueeze(0) # add batch dimensions at beginning
qps = qps.transpose(0, 1).unsqueeze(0) 
noise = np.random.normal(loc=0.0, scale=noise_std, size=np.shape(qs.numpy()))
qps_n = qps + torch.from_numpy(noise)

# outliers - added to start
#outlier_end = int(qs.size(1) * outlier_frac)
#outlier_indices = torch.randperm(outlier_end)
#qps_n[:, 0:outlier_end, :] = qps_n[:, 0:outlier_end, :][:, outlier_indices, :]

# outliers - randomly distributed
outlier_end = int(qs.size(1) * outlier_frac)
outlier_indices = torch.randperm(qs.size(1))[0:outlier_end]
outlier_indices_sorted, indices = torch.sort(outlier_indices)
qps_n[:, outlier_indices_sorted, :] = qps_n[:, outlier_indices, :]

# convert to input format
qs = qs.cuda()
qps_n = qps_n.cuda()

print("Start essential matrix initialisation")
# ToDo: make a copy of qs and qps_n, ensure the first num_ransac_test_points are
# randomly selected from the entire dataset and pass this to the initialise function
Ematrix = essential_matrix.initialise(qs, qps_n, num_test_points, num_ransac_test_points, num_ransac_iterations, inlier_threshold)
Ematrix = Ematrix.cpu()
qs = qs.cpu()
qps_n = qps_n.cpu()

print("Initialized E matrix")
print(Ematrix)
print("Error")
print(errorEMat(Ematrix, E_gt))
error1 =  errorResidual(Ematrix, qs, qps)
print(error1)

print("Start essential matrix optimisation")
# ENSURE THAT ORIGINAL QS AND QPS_N IS USED FOR THIS FUNCTION
Eopt = essential_matrix.optimise(qs, qps_n, Ematrix, delta, alpha, maxreps)

print("E matrix after optimization (polishing)")
print(Eopt)
print("Error")
print(errorEMat(Eopt, E_gt))
error2 =  errorResidual(Eopt, qs, qps)
print(error2)



