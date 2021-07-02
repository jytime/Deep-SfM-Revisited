import torch
import torch.nn.functional as F
from torch.autograd import grad
import essential_matrix
import time
from pdb import set_trace as st

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



def compute_E_matrix(coord1_hom, coord2_hom, intrinsic_inv, delta, alpha, maxreps, num_test_points, 
        ransac_test_points, ransac_iter, ransac_threshold):
    """
    Compute essential matrix and fundamental matrix using our estimation algorithm.
    Args:
        coord1_hom: nx3, homogenous coordinates of first correspondences
        coord2_hom: nx3
        intrinsic_inv: 3x3
        delta: scalar, describes change between L2 and L1
        alpha: scalar, parameter between truncated L2 and Huber
        maxreps: scalr, reps for maximization
    Output:
        E_init, essential matrix after initialization
        E_opt, essential matrix after optimization
        F_init
        F_opt
    """
    coord1_norm = coord1_hom.mm(intrinsic_inv.transpose(0,1))
    coord2_norm = coord2_hom.mm(intrinsic_inv.transpose(0,1))
    coord1_norm = coord1_norm[:,:2]
    coord2_norm = coord2_norm[:,:2]

    coord1 = coord1_norm.unsqueeze(0).contiguous().cuda()
    coord2 = coord2_norm.unsqueeze(0).contiguous().cuda()
    E_init = essential_matrix.initialise(coord1.double(), coord2.double(), num_test_points, ransac_test_points, ransac_iter, ransac_threshold)
    # This is currently done on cpu
    # Maybe: it is good idea to just save a copy on the cpu, rather than passing back and forth. Or do this optimisation on gpu. 
    E_opt = essential_matrix.optimise(coord1.double().cpu(), coord2.double().cpu(), E_init.double().cpu(), delta, alpha, maxreps)

    E_init = E_init.float()
    E_opt = E_opt.float().cuda()
    F_init = intrinsic_inv.transpose(0,1).mm(E_init).mm(intrinsic_inv)
    F_opt = intrinsic_inv.transpose(0,1).mm(E_opt).mm(intrinsic_inv)
    if torch.isnan(torch.sum(F_opt)) or torch.isnan(torch.sum(F_init)):
        st()

    return E_init, E_opt, F_init, F_opt

def compute_E_matrix_ransac(coord1, coord2, intrinsic_inv, delta, alpha, maxreps, num_test_points, 
        ransac_test_points, ransac_iter, ransac_threshold):
    """
    Compute essential matrix and fundamental matrix using our estimation algorithm.
    Args:
        coord1: nx2, homogenous coordinates of first correspondences
        coord2: nx2
        intrinsic_inv: 3x3
        delta: scalar, describes change between L2 and L1
        alpha: scalar, parameter between truncated L2 and Huber
        maxreps: scalr, reps for maximization
    Output:
        E_init, essential matrix after initialization
        E_opt, essential matrix after optimization
        F_init
        F_opt
    """

    E_init = essential_matrix.initialise(coord1.double(), coord2.double(), num_test_points, ransac_test_points, ransac_iter, ransac_threshold)

    E_init = E_init.float()
    F_init = intrinsic_inv.transpose(0,1).mm(E_init).mm(intrinsic_inv)

    return E_init, F_init

def compute_P_matrix_ransac(coord1, coord2, intrinsic_inv, delta, alpha, maxreps, num_test_points, 
        ransac_test_points, ransac_iter, ransac_threshold):
    """
    Compute essential matrix and fundamental matrix using our estimation algorithm.
    Args:
        coord1: nx2, homogenous coordinates of first correspondences
        coord2: nx2
        intrinsic_inv: 3x3
        delta: scalar, describes change between L2 and L1
        alpha: scalar, parameter between truncated L2 and Huber
        maxreps: scalr, reps for maximization
    Output:
        E_init, essential matrix after initialization
        E_opt, essential matrix after optimization
        F_init
        F_opt
    """

    E_init, P_init, inlier_num = essential_matrix.computeP(coord1.double(), coord2.double(), num_test_points, ransac_test_points, ransac_iter, ransac_threshold)

    E_init = E_init.float()
    F_init = intrinsic_inv.transpose(0,1).mm(E_init).mm(intrinsic_inv)

    return E_init, P_init, F_init,inlier_num



############################################# Bilevel Optimization ##################################################
def double_derivative(f, wrt1, wrt2):
    # grads = grad(f, wrt1, create_graph=True)[0].view(-1) # Flatten gradient
    grads = grad(f, wrt1, create_graph=True)[0]
    for idx in range(len(grads)):
        f = grads[idx]
        if idx == 0:
            grad_double = grad(f, wrt2, create_graph=True)[0].contiguous().view(-1)[None]
            # grad_double = grad(f, wrt2, create_graph=True)[0].contiguous().unsqueeze(0)
        else:
            grad_double = torch.cat((grad_double, grad(f, wrt2, create_graph=True)[0].contiguous().view(-1)[None]), 0)
            # grad_double = torch.cat((grad_double, grad(f, wrt2, create_graph=True)[0].contiguous().unsqueeze(0)), dim=0)
    return grad_double

def parametric_huber(residual, delta, alpha):
    residual = torch.abs(residual)
    return torch.where(residual < delta, 0.5 * residual ** 2,\
      alpha * delta * (residual - delta) + 0.5 * delta**2)

def essential_matrix_from_euler_angles_batch(angles):
    '''
    Args:
      angles: bx5
    Output:
      E: bx3x3 

    U[0][0] =  cy*cz; U[0][1] = -cz*sx*sy + cx*sz; U[0][2] = cx*cz*sy + sx*sz;
    U[1][0] = -cy*sz; U[1][1] =  cx*cz + sx*sy*sz; U[1][2] = cz*sx - cx*sy*sz;
    U[2][0] =    -sy; U[2][1] =            -cy*sx; U[2][2] =            cx*cy;
    V[0][0] =     cv; V[0][1] =   0; V[0][2] =    sv;
    V[1][0] = -su*sv; V[1][1] =  cu; V[1][2] = cv*su;
    V[2][0] = -cu*sv; V[2][1] = -su; V[2][2] = cu*cv;
    E = U*I2*V'
    '''
    b = angles.size(0)
    sx = torch.sin(angles[:, 0])
    sy = torch.sin(angles[:, 1])
    sz = torch.sin(angles[:, 2])
    su = torch.sin(angles[:, 3])
    sv = torch.sin(angles[:, 4])
    cx = torch.cos(angles[:, 0])
    cy = torch.cos(angles[:, 1])
    cz = torch.cos(angles[:, 2])
    cu = torch.cos(angles[:, 3])
    cv = torch.cos(angles[:, 4])
    U = torch.stack((torch.stack((cy*cz, -cz*sx*sy + cx*sz, cx*cz*sy + sx*sz), dim=1),\
                     torch.stack((-cy*sz, cx*cz + sx*sy*sz, cz*sx - cx*sy*sz), dim=1),\
                     torch.stack((-sy, -cy*sx, cx*cy), dim=1)), dim=1)
    V = torch.stack((torch.stack((cv, torch.zeros(b, device='cuda'), sv), dim=1),\
                     torch.stack((-su*sv, cu, cv*su), dim=1),\
                     torch.stack((-cu*sv, -su, cu*cv), dim=1)), dim=1)
    E = torch.matmul(torch.matmul(U, torch.diag(torch.tensor([1., 1., 0.], device='cuda')).repeat(b, 1, 1)), V.transpose(1, 2))
    return E

def essential_matrix_from_euler_angles(angles):
    '''
    Args:
      angles: 5
    Output:
      E: 3x3

    U[0][0] =  cy*cz; U[0][1] = -cz*sx*sy + cx*sz; U[0][2] = cx*cz*sy + sx*sz;
    U[1][0] = -cy*sz; U[1][1] =  cx*cz + sx*sy*sz; U[1][2] = cz*sx - cx*sy*sz;
    U[2][0] =    -sy; U[2][1] =            -cy*sx; U[2][2] =            cx*cy;
    V[0][0] =     cv; V[0][1] =   0; V[0][2] =    sv;
    V[1][0] = -su*sv; V[1][1] =  cu; V[1][2] = cv*su;
    V[2][0] = -cu*sv; V[2][1] = -su; V[2][2] = cu*cv;
    E = U*I2*V'
    '''
    sx = torch.sin(angles[0])
    sy = torch.sin(angles[1])
    sz = torch.sin(angles[2])
    su = torch.sin(angles[3])
    sv = torch.sin(angles[4])
    cx = torch.cos(angles[0])
    cy = torch.cos(angles[1])
    cz = torch.cos(angles[2])
    cu = torch.cos(angles[3])
    cv = torch.cos(angles[4])
    U = torch.stack((torch.stack((cy*cz, -cz*sx*sy + cx*sz, cx*cz*sy + sx*sz)),\
                     torch.stack((-cy*sz, cx*cz + sx*sy*sz, cz*sx - cx*sy*sz)),\
                     torch.stack((-sy, -cy*sx, cx*cy))))
    V = torch.stack((torch.stack((cv, torch.tensor(0., device='cuda'), sv)),\
                     torch.stack((-su*sv, cu, cv*su)),\
                     torch.stack((-cu*sv, -su, cu*cv))))
    E = torch.mm(torch.mm(U, torch.diag(torch.tensor([1., 1., 0.], device='cuda'))), V.transpose(0, 1))
    return E

def fitting_loss_givens(angles, x, xp):
    """ Computes one-sided residual error.
    Args:
      angles: Euler angles parameterizing essential matrix, bx5
      x: Normalized point set 1, bxnx2 
      xp: Normalized point set 2, bxnx2
    Output:
      error: average single-sided residual error, in normalized coordinate space
    """
    x = x.transpose(1,2).contiguous()
    xp = xp.transpose(1,2).contiguous()
    ones = torch.ones((x.size(0), 1, x.size(2)), dtype=torch.float32, device='cuda')
    x  = torch.cat((x, ones), dim=1)
    xp = torch.cat((xp, ones), dim=1)

    E = essential_matrix_from_euler_angles_batch(angles)
    Ex = torch.matmul(E, x) # bx3xn
    numerator = torch.squeeze(torch.squeeze(torch.abs(torch.matmul(xp.transpose(1,2).unsqueeze(2), Ex.transpose(1,2).unsqueeze(3))), dim=3), dim=2) # bxn
    denominator = torch.sqrt(Ex[:, 0, :]**2 + Ex[:, 1, :]**2)
    error = numerator / denominator
    error = error.mean() 

    return error

def robust_epipolar_loss_givens(angles, x, xp, delta, alpha):
    ''' Computes a robust epipolar loss
    1/n sum rho(xp^T E x)
    Args:
      angles: essential matrix parameters, 5
      x: point-set 1, nx2
      xp: point-set 2, nx2
    Output:
      loss: average epipolar loss
    '''
    # Convert back to homogeneous
    x = x.transpose(0,1).contiguous()
    xp = xp.transpose(0,1).contiguous()
    ones = torch.ones((1, x.size(1)), dtype=torch.float32, device='cuda')
    x  = torch.cat((x, ones), dim=0)
    xp = torch.cat((xp, ones), dim=0)
    E = essential_matrix_from_euler_angles(angles)
    Ex = torch.mm(E, x) # 3xn
    residuals = torch.squeeze(\
      torch.matmul(xp.transpose(0,1).unsqueeze(1),\
      Ex.transpose(0,1).unsqueeze(2))) # n
    loss = parametric_huber(residuals, delta, alpha) # n
    loss = loss.mean() # scalar
    return loss

def robust_epipolar_loss(angles, x, xp, delta, alpha):
    ''' Computes a robust epipolar loss
    1/n sum rho(xp^T E x)
    Args:
      angles: essential matrix parameters, 5
      x: point-set 1, 3xn -> (V^T * x)
      xp: point-set 2, 3xn -> (U^T * xp)
    Output:
      loss: average epipolar loss
    '''
    # ToDo: Rewrite assuming identity E matrix
    E = essential_matrix_from_euler_angles(angles) # Should be diag([1, 1, 0])
    Ex = torch.mm(E, x) # 3xn
    residuals = torch.squeeze(\
      torch.matmul(xp.transpose(0,1).unsqueeze(1),\
      Ex.transpose(0,1).unsqueeze(2))) # n
    loss = parametric_huber(residuals, delta, alpha) # n
    loss = loss.mean() # scalar
    return loss

def analytical_gradient(x, xp, delta, alpha):
    '''
    Compute derivative of the optimal essential matrix with respect to the inputs
    by evaluating the second derivative with respect to the essential matrix parameters
    and the mixed derivative
    Args:
      x: point-set 1, 3xn -> (V^T * x)
      xp: point-set 2, 3xn -> (U^T * xp)
    Output:
    '''

    with torch.no_grad():
        # Compute elementwise products
        xxp = torch.mul(x[0, :], xp[0, :]) # torch.Size([n])
        xyp = torch.mul(x[0, :], xp[1, :])
        xzp = torch.mul(x[0, :], xp[2, :])
        yxp = torch.mul(x[1, :], xp[0, :])
        yyp = torch.mul(x[1, :], xp[1, :])
        yzp = torch.mul(x[1, :], xp[2, :])
        zxp = torch.mul(x[2, :], xp[0, :])
        zyp = torch.mul(x[2, :], xp[1, :])
        zzp = torch.mul(x[2, :], xp[2, :])

        zero_tensor = torch.zeros_like(xxp)

        r_at_0 = xxp + yyp # torch.Size([n])
        drdt_at_0 = torch.stack((-yzp, -xzp, yxp - xyp, -zyp, -zxp), dim=1)  # torch.Size([n, 5])
        # Is actually symmetric, so annoying to have to set all values
        d2rdt2_at_0 = torch.stack((torch.stack((-yyp, -yxp, zero_tensor, zzp, zero_tensor), dim=1),\
                                   torch.stack((-yxp, -xxp, zero_tensor, zero_tensor, zzp), dim=1),\
                                   torch.stack((zero_tensor, zero_tensor, -xxp - yyp, -zxp,  zyp), dim=1),\
                                   torch.stack((zzp, zero_tensor, -zxp, -yyp, -yxp), dim=1),\
                                   torch.stack((zero_tensor, zzp, zyp, -yxp, -xxp), dim=1)), dim=1) # torch.Size([n, 5, 5])

        drdt_at_0_outer_product = drdt_at_0.unsqueeze(2) * drdt_at_0.unsqueeze(1)
        d2fidt2_at_0 = torch.where((torch.abs(r_at_0) < delta).view(-1,1,1),\
                                   drdt_at_0_outer_product + r_at_0.view(-1,1,1) * d2rdt2_at_0,\
                                   alpha * delta * torch.sign(r_at_0).view(-1,1,1) * d2rdt2_at_0)
        d2fdt2_at_0 = d2fidt2_at_0.mean(dim=0)

        d2rdxpdt_at_0 = torch.stack((torch.stack((zero_tensor, zero_tensor, -x[1, :]), dim=1),\
                                     torch.stack((zero_tensor, zero_tensor, -x[0, :]), dim=1),\
                                     torch.stack((x[1, :], -x[0, :], zero_tensor), dim=1),\
                                     torch.stack((zero_tensor, -x[2, :], zero_tensor), dim=1),\
                                     torch.stack((-x[2, :], zero_tensor, zero_tensor), dim=1)), dim=1) # torch.Size([n, 5, 3])
        I_hat = torch.diag(torch.tensor([1.,1.,0.]))
        # Note unfortunate transpose here - is there a better way to formulate this?
        drdt_by_drdxp_at_0 = drdt_at_0.unsqueeze(2) * torch.mm(I_hat, x).t().unsqueeze(1) # torch.Size([n, 5, 3])
        d2fidxpdt_at_0 = torch.where((torch.abs(r_at_0) < delta).view(-1,1,1),\
                                     drdt_by_drdxp_at_0 + r_at_0.view(-1,1,1) * d2rdxpdt_at_0,\
                                     alpha * delta * torch.sign(r_at_0).view(-1,1,1) * d2rdxpdt_at_0)
        # Concatenate each point correspondence and divide by number of points
        # Rows: [x1, y1, z1, x2, y2, z2, ...]
        # Note: different to that found by double derivative [x1, x2, ..., y1, ..., z1, ...]
        d2fdxpdt_at_0 = d2fidxpdt_at_0.permute([1,0,2]).reshape([5, -1]) / x.size(1) # torch.Size([5, 3*n])

        # Compute derivative
        dEdxp, _ = torch.gesv(-d2fdxpdt_at_0, d2fdt2_at_0)

        # ToDo: currently w.r.t. U^T * xp

    return (d2fdt2_at_0, d2fdxpdt_at_0, dEdxp)
