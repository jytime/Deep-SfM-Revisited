from __future__ import division
import shutil
import numpy as np
import torch
from path import Path
import datetime
from collections import OrderedDict
from collections import namedtuple


def save_path_formatter(args, parser):
    def is_default(key, value):
        return value == parser.get_default(key)
    args_dict = vars(args)
    data_folder_name = str(Path(args_dict['data']).normpath().name)
    folder_string = [data_folder_name]
    if not is_default('epochs', args_dict['epochs']):
        folder_string.append('{}epochs'.format(args_dict['epochs']))
    keys_with_prefix = OrderedDict()
    keys_with_prefix['epoch_size'] = 'epoch_size'
    keys_with_prefix['batch_size'] = 'b'
    keys_with_prefix['lr'] = 'lr'

    for key, prefix in keys_with_prefix.items():
        value = args_dict[key]
        if not is_default(key, value):
            folder_string.append('{}{}'.format(prefix, value))
    save_path = Path(','.join(folder_string))
    timestamp = datetime.datetime.now().strftime("%m-%d-%H:%M")
    return save_path/timestamp


OxtsPacket = namedtuple('OxtsPacket',
                        'lat, lon, alt, ' +
                        'roll, pitch, yaw, ' +
                        'vn, ve, vf, vl, vu, ' +
                        'ax, ay, az, af, al, au, ' +
                        'wx, wy, wz, wf, wl, wu, ' +
                        'pos_accuracy, vel_accuracy, ' +
                        'navstat, numsats, ' +
                        'posmode, velmode, orimode')

# Bundle into an easy-to-access structure
OxtsData = namedtuple('OxtsData', 'packet, T_w_imu')



def load_oxts_packets_and_poses(oxts_files):
    """Generator to read OXTS ground truth data.
       Poses are given in an East-North-Up coordinate system 
       whose origin is the first GPS position.
    """
    # Scale for Mercator projection (from first lat value)
    scale = None
    # Origin of the global coordinate system (first GPS position)
    origin = None

    oxts = []

    for filename in oxts_files:
        with open(filename, 'r') as f:
            for line in f.readlines():
                line = line.split()
                # Last five entries are flags and counts
                line[:-5] = [float(x) for x in line[:-5]]
                line[-5:] = [int(float(x)) for x in line[-5:]]

                packet = OxtsPacket(*line)

                if scale is None:
                    scale = np.cos(packet.lat * np.pi / 180.)

                R, t = pose_from_oxts_packet(packet, scale)

                if origin is None:
                    origin = t

                T_w_imu = transform_from_rot_trans(R, t - origin)

                oxts.append(OxtsData(packet, T_w_imu))

    return oxts


def pose_from_oxts_packet(packet, scale):
    """Helper method to compute a SE(3) pose matrix from an OXTS packet.
    """
    er = 6378137.  # earth radius (approx.) in meters

    # Use a Mercator projection to get the translation vector
    tx = scale * packet.lon * np.pi * er / 180.
    ty = scale * er * \
        np.log(np.tan((90. + packet.lat) * np.pi / 360.))
    tz = packet.alt
    t = np.array([tx, ty, tz])

    # Use the Euler angles to get the rotation matrix
    Rx = rotx(packet.roll)
    Ry = roty(packet.pitch)
    Rz = rotz(packet.yaw)
    R = Rz.dot(Ry.dot(Rx))

    # Combine the translation and rotation into a homogeneous transform
    return R, t


def transform_from_rot_trans(R, t):
    """Transforation matrix from rotation matrix and translation vector."""
    R = R.reshape(3, 3)
    t = t.reshape(3, 1)
    return np.hstack([R, t])



def rotx(t):
    """Rotation about the x-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1,  0,  0],
                     [0,  c, -s],
                     [0,  s,  c]])



def rotz(t):
    """Rotation about the z-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])

def roty(t):
    """Rotation about the y-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  0,  s],
                     [0,  1,  0],
                     [-s, 0,  c]])


def read_calib_file(filepath):
    """Read in a calibration file and parse into a dictionary."""
    data = {}

    with open(filepath, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    return data

def kitti_readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines

def tensor2array(tensor, max_value=255, colormap='rainbow'):
    if max_value is None:
        max_value = tensor.max()
    if tensor.ndimension() == 2 or tensor.size(0) == 1:
        try:
            import cv2
            if cv2.__version__.startswith('2'): # 2.4
                color_cvt = cv2.cv.CV_BGR2RGB
            else:  
                color_cvt = cv2.COLOR_BGR2RGB
            if colormap == 'rainbow':
                colormap = cv2.COLORMAP_RAINBOW
            elif colormap == 'bone':
                colormap = cv2.COLORMAP_BONE
            array = (255*tensor.squeeze().numpy()/max_value).clip(0, 255).astype(np.uint8)
            colored_array = cv2.applyColorMap(array, colormap)
            array = cv2.cvtColor(colored_array, color_cvt).astype(np.float32)/255
            array = array.transpose(2, 0, 1)
        except ImportError:
            if tensor.ndimension() == 2:
                tensor.unsqueeze_(2)
            array = (tensor.expand(tensor.size(0), tensor.size(1), 3).numpy()/max_value).clip(0,1)

    elif tensor.ndimension() == 3:
        #assert(tensor.size(0) == 3)
        #array = 0.5 + tensor.numpy()*0.5
        array = 0.5 + tensor.numpy().transpose(1,2,0)*0.5
    return array


def save_checkpoint(save_path, dpsnet_state, epoch, filename='checkpoint.pth.tar'):
    file_prefixes = ['dpsnet']
    states = [dpsnet_state]
    for (prefix, state) in zip(file_prefixes, states):
        torch.save(state, save_path/'{}_{}_{}'.format(prefix,epoch,filename))


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.lr * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



def matrix2angle(matrix):
    """
    ref: https://github.com/matthew-brett/transforms3d/blob/master/transforms3d/euler.py
    input size: ... * 3 * 3
    output size:  ... * 3
    """
    i = 0
    j = 1
    k = 2
    dims = [dim for dim in matrix.shape]
    M = matrix.contiguous().view(-1, 3, 3)

    cy = torch.sqrt(M[:, i, i] * M[:, i, i] + M[:, j, i] * M[:, j, i])

    if torch.max(cy).item() > 1e-15 * 4:
        ax = torch.atan2(M[:, k, j], M[:, k, k])
        ay = torch.atan2(-M[:, k, i], cy)
        az = torch.atan2(M[:, j, i], M[:, i, i])
    else:
        ax = torch.atan2(-M[:, j, k], M[:, j, j])
        ay = torch.atan2(-M[:, k, i], cy)
        az = torch.zero(matrix.shape[:-1])
    return torch.cat([torch.unsqueeze(ax, -1), torch.unsqueeze(ay, -1), torch.unsqueeze(az, -1)], -1).view(dims[:-1])


def angle2matrix(angle):
    """
    ref: https://github.com/matthew-brett/transforms3d/blob/master/transforms3d/euler.py
    input size:  ... * 3
    output size: ... * 3 * 3
    """
    dims = [dim for dim in angle.shape]
    angle = angle.view(-1, 3)

    i = 0
    j = 1
    k = 2
    ai = angle[:, 0]
    aj = angle[:, 1]
    ak = angle[:, 2]
    si, sj, sk = torch.sin(ai), torch.sin(aj), torch.sin(ak)
    ci, cj, ck = torch.cos(ai), torch.cos(aj), torch.cos(ak)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk

    M = torch.eye(3)
    M = M.view(1, 3, 3)
    M = M.repeat(angle.shape[0], 1, 1).cuda()

    M[:, i, i] = cj * ck
    M[:, i, j] = sj * sc - cs
    M[:, i, k] = sj * cc + ss
    M[:, j, i] = cj * sk
    M[:, j, j] = sj * ss + cc
    M[:, j, k] = sj * cs - sc
    M[:, k, i] = -sj
    M[:, k, j] = cj * si
    M[:, k, k] = cj * ci

    return M.view(dims + [3])

