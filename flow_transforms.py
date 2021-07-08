from __future__ import division
import torch
import random
import numpy as np
import numbers
import types
import scipy.ndimage as ndimage
from scipy.misc import imresize
from scipy.ndimage.interpolation import zoom
from lib.config import cfg, cfg_from_file, save_config_to_file

'''Set of tranform random routines that takes both input and target as arguments,
in order to have random but coherent transformations.
inputs are PIL Image pairs and targets are ndarrays'''



class ComposeCo(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, images, targets, intrinsics):
        for t in self.transforms:
            images, targets, intrinsics = t(images, targets, intrinsics)
        return images, targets, intrinsics



class NormalizeCo(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, images):
        for tensor in images:
            for t, m, s in zip(tensor, self.mean, self.std):
                t.sub_(m).div_(s)
        return images


class ArrayToTensorCo(object):
    """Converts a list of numpy.ndarray (H x W x C) along with a intrinsics matrix to a list of torch.FloatTensor of shape (C x H x W) with a intrinsics tensor."""

    def __call__(self, images):
        tensors = []

        # if images is None: return None
        for im in images:
            assert(isinstance(im, np.ndarray))
            # put it from HWC to CHW format
            im = np.transpose(im, (2, 0, 1))
            # handle numpy array
            tensors.append(torch.from_numpy(im).float())
        return tensors

class CenterCropCo(object):
    """ Center crop image """
    def __init__(self, size):
        self.size = size

    def __call__(self, images, target, intrinsics):
        out_h, out_w = self.size
        in_h, in_w, _ = images[0].shape
        offset_y = (in_h - out_h) // 2
        offset_x = (in_w - out_w) // 2
        cropped_images = [im[offset_y:offset_y+out_h, offset_x:offset_x+out_w, :] for im in images]
        if target is None:
            cropped_target = None
        else:
            cropped_target = [tar[offset_y:offset_y + out_h, offset_x:offset_x + out_w, :] for tar in target]

        output_intrinsics = np.copy(intrinsics)
        output_intrinsics[0,2] -= offset_x
        output_intrinsics[1,2] -= offset_y
        return cropped_images, cropped_target, output_intrinsics

class RandomCropCo(object):
    """Randomly crop images"""
    def __init__(self, size):
        self.size = size

    def __call__(self, images, depths, intrinsics):
        assert intrinsics is not None
        output_intrinsics = np.copy(intrinsics)

        out_h, out_w = self.size 
        in_h, in_w, _ = images[0].shape

        if random.random()>0.5 and cfg.ZOOM_INPUT:
            x_scaling = np.random.uniform(1.0,1.15)
            y_scaling = np.random.uniform(1.0,1.15)
        else:
            x_scaling=1.0;y_scaling=1.0;

        scaled_h, scaled_w = round(in_h * y_scaling), round(in_w * x_scaling)

        output_intrinsics[0] *= x_scaling
        output_intrinsics[1] *= y_scaling

        scaled_images = [imresize(im, (scaled_h, scaled_w)) for im in images]
        scaled_depths = [resize_sparse_map(depth,depth>0, x_scaling,y_scaling) for depth in depths]

        offset_y = np.random.randint(scaled_h - out_h + 1)
        offset_x = np.random.randint(scaled_w - out_w + 1)

        cropped_images = [im[offset_y:offset_y + out_h, offset_x:offset_x + out_w, :] for im in scaled_images]
        cropped_depths = [de[offset_y:offset_y + out_h, offset_x:offset_x + out_w] for de in scaled_depths]

        output_intrinsics[0,2] -= offset_x
        output_intrinsics[1,2] -= offset_y

        return cropped_images, cropped_depths, output_intrinsics


def resize_sparse_map( flow, valid, fx=1.0, fy=1.0):
    ht, wd = flow.shape[:2]
    coords = np.meshgrid(np.arange(wd), np.arange(ht))
    coords = np.stack(coords, axis=-1)

    coords = coords.reshape(-1, 2).astype(np.float32)
    flow = flow.reshape(-1, 1).astype(np.float32)
    valid = valid.reshape(-1).astype(np.float32)

    coords0 = coords[valid>=1]
    flow0 = flow[valid>=1]

    ht1 = int(round(ht * fy))
    wd1 = int(round(wd * fx))

    coords1 = coords0 * [fx, fy]
    # flow1 = flow0 * [fx, fy]

    xx = np.round(coords1[:,0]).astype(np.int32)
    yy = np.round(coords1[:,1]).astype(np.int32)

    v = (xx > 0) & (xx < wd1) & (yy > 0) & (yy < ht1)
    xx = xx[v]
    yy = yy[v]
    flow0 = flow0[v]

    flow_img = np.zeros([ht1, wd1, 1], dtype=np.float32)
    # valid_img = np.zeros([ht1, wd1], dtype=np.int32)

    flow_img[yy, xx] = flow0
    # valid_img[yy, xx] = 1

    return flow_img #, valid_img

class Compose(object):
    """ Composes several co_transforms together.
    For example:
    >>> co_transforms.Compose([
    >>>     co_transforms.CenterCrop(10),
    >>>     co_transforms.ToTensor(),
    >>>  ])
    """

    def __init__(self, co_transforms):
        self.co_transforms = co_transforms

    def __call__(self, input):
        for t in self.co_transforms:
            input = t(input)
        return input


class ArrayToTensor(object):
    """Converts a numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W)."""

    def __call__(self, array):
        assert(isinstance(array, np.ndarray))
        array = np.transpose(array, (2, 0, 1))
        # handle numpy array
        tensor = torch.from_numpy(array)
        # put it from HWC to CHW format
        return tensor.float()


class Lambda(object):
    """Applies a lambda as a transform"""

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, input,target):
        return self.lambd(input,target)
      

class CenterCrop(object):
    """Crops the given inputs and target arrays at the center to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    Careful, img1 and img2 may not be the same size
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, inputs, target):
        h1, w1, _ = inputs[0].shape
        h2, w2, _ = inputs[1].shape
        th, tw = self.size
        x1 = int(round((w1 - tw) / 2.))
        y1 = int(round((h1 - th) / 2.))
        x2 = int(round((w2 - tw) / 2.))
        y2 = int(round((h2 - th) / 2.))

        inputs[0] = inputs[0][y1: y1 + th, x1: x1 + tw]
        inputs[1] = inputs[1][y2: y2 + th, x2: x2 + tw]
        target = target[y1: y1 + th, x1: x1 + tw]
        return inputs, target


class Scale(object):
    """ Rescales the inputs and target arrays to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation order: Default: 2 (bilinear)
    """

    def __init__(self, size, order=2):
        self.size = size
        self.order = order

    def __call__(self, inputs, target):
        h, w, _ = inputs[0].shape
        if (w <= h and w == self.size) or (h <= w and h == self.size):
            return inputs,target
        if w < h:
            ratio = self.size/w
        else:
            ratio = self.size/h

        inputs[0] = ndimage.interpolation.zoom(inputs[0], ratio, order=self.order)
        inputs[1] = ndimage.interpolation.zoom(inputs[1], ratio, order=self.order)

        target = ndimage.interpolation.zoom(target, ratio, order=self.order)
        target *= ratio
        return inputs, target


class RandomCrop(object):
    """Crops the given PIL.Image at a random location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, inputs,target):
        h, w, _ = inputs[0].shape
        th, tw = self.size
        if w == tw and h == th:
            return inputs,target

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        inputs[0] = inputs[0][y1: y1 + th,x1: x1 + tw]
        inputs[1] = inputs[1][y1: y1 + th,x1: x1 + tw]
        return inputs, target[y1: y1 + th,x1: x1 + tw]

class Padding(object):
    """Crops the given inputs and target arrays at the center to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    Careful, img1 and img2 may not be the same size
    """

    def __init__(self, size,scale=64):
        self.scale=scale
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
            
    def __call__(self, inputs, target):
        h1, w1, _ = inputs[0].shape
        h2, w2, _ = inputs[1].shape

        height_new = int(np.ceil(h1/self.scale)*self.scale)
        width_new  = int(np.ceil(w1/self.scale)*self.scale)
        pw = width_new-w1
        ph = height_new-h1
        padding = ((0,pw),(0,ph),(0,0))
        inputs[0] =np.pad(inputs[0], padding,  "constant",constant_values=0)
        inputs[1] =np.pad(inputs[1], padding,  "constant",constant_values=0)
        target=np.pad(target,padding,  "constant",constant_values=0)
        return inputs,target
    