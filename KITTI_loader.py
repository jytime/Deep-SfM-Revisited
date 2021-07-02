import torch.utils.data as data
import os
import os.path
from imageio import imread
import numpy as np
import glob
from pdb import set_trace as st
import time
import cv2
from lib.config import cfg, cfg_from_file, save_config_to_file
from torchvision.transforms import ColorJitter
from PIL import Image
from utils import kitti_readlines,read_calib_file
import utils
import random
from kitti_utils import generate_depth_map


def load_flow_from_png(png_path):
    # The -1 is here to specify not to change the image depth (16bit), and is compatible
    # with both OpenCV2 and OpenCV3
    flo_file = cv2.imread(png_path, -1)
    flo_img = flo_file[:,:,2:0:-1].astype(np.float32)
    invalid = (flo_file[:,:,0] == 0)
    flo_img = flo_img - 32768
    flo_img = flo_img / 64
    flo_img[np.abs(flo_img) < 1e-10] = 1e-10
    flo_img[invalid, :] = 0
    return(flo_img)

def disparity_loader_png(path):
    if os.path.exists(path):
        disp_file= cv2.imread(path, -1)
        disp = disp_file.astype(np.float32)
        disp = disp / 256
        disp = np.expand_dims(disp, 2)
    else:
        disp = None
    return disp

def mask_loader_png(path):
    if os.path.exists(path):
        disp_file= cv2.imread(path, -1)
        disp_file = disp_file > 0
        disp = disp_file.astype(np.float32)
        disp = np.expand_dims(disp, 2)
    else:
        disp = None
    return disp

def load_intrinsics(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
        data = {}
        for line in lines:
            key, value = line.split(':', 1)
            data[key] = np.array([float(x) for x in value.split()])
        p_mat = np.resize(data['P2'], (3, 4))
        intrinsics = p_mat[:,:3]
    return intrinsics

def load_poses(filepath):
    poses = []
    with open(filepath, 'r') as readfile:
        lines = readfile.readlines()
        for line in lines:
            line = line.strip()
            pose = np.fromstring(line, dtype=float, sep=' ')
            pose = pose.reshape(3, 4)
            #pose = np.vstack((pose, [0, 0, 0, 1]))
            pose = pose.astype(np.float32)
            poses.append(pose)
        return poses

def compute_deltaRT(ps1,ps2):   
    R1 = ps1[:3,:3]
    T1 = ps1[:,3:]  
    R2 = ps2[:3,:3]
    T2 = ps2[:,3:]
    Rf = R1.T.dot(R2)
    Tf = R1.T.dot(T2-T1)
    pose = np.concatenate((Rf, Tf), axis=1)
    return pose







class KITTIVOLoaderGT(data.Dataset):
    def __init__(self, root, transform=None, target_transform=None, co_transform=None, train=True):
        self.root = root

        # sequence for training and testing
        self.train_seqs = [0,1,2,3,4,5,6,7,8]
        self.test_seqs = [9,10]
        
        self.train = train

        if train:
            self._collect_train_frames()
            self._collect_train_frames_gt()
            self.path_list = self.train_frames
            self.path_list_gt = self.train_frames_gt
            self.path_list_gt_mask = self.train_frames_gt_mask
        else:
            self._collect_test_frames()
            self._collect_test_frames_gt()
            self.path_list = self.test_frames
            self.path_list_gt = self.test_frames_gt
            self.path_list_gt_mask = self.test_frames_gt_mask

        self.transform = transform
        self.target_transform = target_transform
        self.co_transform = co_transform
        self.pose_dict = {i:load_poses(os.path.join(self.root, 'poses', '{:02d}.txt'.format(i))) for i in range(11)}
        self.photo_aug = ColorJitter.get_params((0.8, 1.2), (0.8, 1.2), (0.8, 1.2), (-0.1, 0.1))
        self.asymmetric_color_aug_prob = 0.2

        
        self.pred_poses_fw = {i: np.load(os.path.join(self.root, 'pred_poses','{:02d}_fw.npy').format(i)) for i in range(11)}
        self.pred_poses_bw = {i: np.load(os.path.join(self.root, 'pred_poses','{:02d}_bw.npy').format(i)) for i in range(11)}

    def _collect_train_frames(self):
        self.train_frames = []
        self.seq_len = []
        for seq in self.train_seqs:
            img_dir = os.path.join(self.root, "sequences", "{:02d}".format(seq), "image_2")
            img_paths = glob.glob(os.path.join(img_dir, '*.png'))
            N = len(img_paths)
            self.train_frames.extend(img_paths)
            self.seq_len.append(N)
        self.train_frames = sorted(self.train_frames)

    def _collect_train_frames_gt(self):
        self.train_frames_gt = []
        self.train_frames_gt_mask = []
        for seq in self.train_seqs:
            img_dir = os.path.join(self.root, "Pseudo_depth", "{:02d}".format(seq), "velodyne")
            mask_dir = os.path.join(self.root, "RealDepth", "{:02d}".format(seq), "velodyne")
            img_paths = glob.glob(os.path.join(img_dir, '*.png'))
            mask_paths = glob.glob(os.path.join(mask_dir, '*.png'))
            self.train_frames_gt.extend(img_paths)
            self.train_frames_gt_mask.extend(mask_paths)
        self.train_frames_gt = sorted(self.train_frames_gt)
        self.train_frames_gt_mask = sorted(self.train_frames_gt_mask)

    def _collect_test_frames(self):
        self.test_frames = []
        self.seq_len = []
        for seq in self.test_seqs:
            img_dir = os.path.join(self.root, "sequences", "{:02d}".format(seq), "image_2")
            img_paths = glob.glob(os.path.join(img_dir, '*.png'))
            N = len(img_paths)
            self.test_frames.extend(img_paths)
            self.seq_len.append(N)
        self.test_frames = sorted(self.test_frames)

    def _collect_test_frames_gt(self):
        self.test_frames_gt = []
        self.test_frames_gt_mask = []
        for seq in self.test_seqs:
            img_dir = os.path.join(self.root, "Pseudo_depth", "{:02d}".format(seq), "velodyne")
            # img_dir = os.path.join(self.root, "RealDepth", "{:02d}".format(seq), "velodyne")
            mask_dir = os.path.join(self.root, "RealDepth", "{:02d}".format(seq), "velodyne")


            img_paths = glob.glob(os.path.join(img_dir, '*.png'))
            mask_paths = glob.glob(os.path.join(mask_dir, '*.png'))
            self.test_frames_gt.extend(img_paths)
            self.test_frames_gt_mask.extend(mask_paths)
        self.test_frames_gt = sorted(self.test_frames_gt)
        self.test_frames_gt_mask = sorted(self.test_frames_gt_mask)


    def __getitem__(self, index):
        # load gt
        gt1_path = self.path_list_gt[index]
        gt1_path_mask = self.path_list_gt_mask[index]

        # load image
        img1_path = self.path_list[index]
        path1_split = img1_path.split('/')
        seq_1 = int(path1_split[-3])
        img_id_1 = int(os.path.splitext(path1_split[-1])[0])
        skip = cfg.SKIP

        
        try:
            img2_path = self.path_list[index+skip]
            gt2_path = self.path_list_gt[index+skip]
            gt2_path_mask = self.path_list_gt_mask[index+skip]
        except:
            img2_path = self.path_list[index-skip]
            gt2_path = self.path_list_gt[index-skip]
            gt2_path_mask = self.path_list_gt_mask[index-skip]
        path2_split = img2_path.split('/')
        seq_2 = int(path2_split[-3])
        img_id_2 = int(os.path.splitext(path2_split[-1])[0])

        if seq_1 != seq_2:
            # in the case where img1 is the last frame in the sequence, take image 2 as the previous frame
            img2_path = self.path_list[index-skip]
            gt2_path = self.path_list_gt[index-skip]
            gt2_path_mask = self.path_list_gt_mask[index-skip]
            path2_split = img2_path.split('/')
            seq_2 = int(path2_split[-3])
            img_id_2 = int(os.path.splitext(path2_split[-1])[0])

        assert(seq_1 == seq_2)

        inputs = [img1_path, img2_path] 
        gt_depth = [gt1_path,gt2_path]
        gt_depth_mask = [gt1_path_mask,gt2_path_mask]

        # load intrinsic
        calib = os.path.join(self.root, "sequences", "{:02d}".format(seq_1), "calib.txt")

        # load 
        # if cfg.CHECK_WRONG_POSE:
        #     pose_1 = self.pose_dict[seq_1][img_id_1]
        #     try:
        #         pose_2 = self.pose_dict[seq_2][img_id_2+1]
        #     except:
        #         pose_2 = self.pose_dict[seq_2][img_id_2]
        # else:


        pose_1 = self.pose_dict[seq_1][img_id_1]
        pose_2 = self.pose_dict[seq_2][img_id_2]
        pose_bw = compute_deltaRT(pose_1,pose_2)
        pose_fw = compute_deltaRT(pose_2,pose_1)
        poses = [pose_fw, pose_bw]

        pred_pose_fw = self.pred_poses_fw[seq_1][img_id_1]
        pred_pose_bw = self.pred_poses_bw[seq_1][img_id_1]
        pred_poses = [pred_pose_fw, pred_pose_bw]


        # Write load images and intrinsics 
        imgs = [os.path.join(self.root, path) for path in inputs]
        intrinsic = os.path.join(self.root, calib)

        depth_gt_mask = [mask_loader_png(gt_mask) for gt_mask in gt_depth_mask]
        depth_gt = [disparity_loader_png(gt) for gt in gt_depth]
        depth_gt = [a*b for a,b in zip(depth_gt,depth_gt_mask)]
        inputs, depth_gt, calib = [cv2.imread(img)[:,:,::-1].astype(np.uint8) for img in imgs], depth_gt, load_intrinsics(intrinsic)
        # calib: intri

        if self.train:
            if random.random() > 0.5:
                image_stack = np.concatenate([inputs[0], inputs[1]], axis=0)
                image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
                img1, img2 = np.split(image_stack, 2, axis=0)
                inputs[0] = img1.astype(np.float32);
                inputs[1] = img2.astype(np.float32)

        if self.co_transform is not None:
            inputs, depth_gt, calib = self.co_transform(inputs, depth_gt, calib)
        if self.transform is not None:
            inputs = self.transform(inputs)  
        if self.target_transform is not None:
            depth_gt = self.target_transform(depth_gt)

        # if cfg.GENERATE_KITTI_POSE_TO_SAVE:
        #     return inputs, calib, poses, depth_gt,seq_1,img_id_1
        # else:
        return inputs, calib, poses,pred_poses, depth_gt, img1_path,img2_path

    def __len__(self):
        return len(self.path_list)





class KITTIRAWLoaderGT(data.Dataset):
    def __init__(self, root, transform=None, target_transform=None, co_transform=None, train=True):

        self.root = root
        
        self.train = train

        train_files = '/mnt/lustre/wangjianyuan/data/KITTI_RAW/kitti_raw_dataset/jy_split/train_files.txt'
        # test_files = '/mnt/lustre/wangjianyuan/data/KITTI_RAW/kitti_raw_dataset/jy_split/test_files.txt'
        # test_files = '/mnt/lustre/wangjianyuan/data/KITTI_RAW/kitti_raw_dataset/jy_split/test_files_odom.txt'
        if cfg.KITTI_697:
            test_files = '/mnt/lustre/wangjianyuan/data/KITTI_RAW/kitti_raw_dataset/jy_split/test_files.txt'
        else:
            test_files = '/mnt/lustre/wangjianyuan/data/KITTI_RAW/kitti_raw_dataset/jy_split/test_files_benchmark.txt'


        if self.train:
            # gt_depth_path = '/mnt/lustre/wangjianyuan/data/KITTI_RAW/kitti_raw_dataset/jy_split/train_gt_depths.npz'
            # self.gt_depth_list = np.load(gt_depth_path, fix_imports=True, encoding='latin1',allow_pickle=True)["data"]
            self.path_list = kitti_readlines(train_files)

            train_files_ben = kitti_readlines('/mnt/lustre/wangjianyuan/data/KITTI_RAW/kitti_raw_dataset/jy_split/train_files_ben.txt')
            val_files_ben = kitti_readlines('/mnt/lustre/wangjianyuan/data/KITTI_RAW/kitti_raw_dataset/jy_split/val_files_ben.txt')

            temp_list = []
            for num in range(len(self.path_list)):
                cur_file = self.path_list[num]
                if cur_file in train_files_ben or cur_file in val_files_ben:
                    temp_list.append(cur_file)
            self.path_list = temp_list
        else:
            # gt_depth_path = '/mnt/lustre/wangjianyuan/data/KITTI_RAW/kitti_raw_dataset/jy_split/test_gt_depths.npz'
            # gt_depth_path = '/mnt/lustre/wangjianyuan/data/KITTI_RAW/kitti_raw_dataset/jy_split/gt_depths_odom.npz'
            # self.gt_depth_list = np.load(gt_depth_path, fix_imports=True, encoding='latin1',allow_pickle=True)["data"]
            self.path_list  = kitti_readlines(test_files)
            if cfg.EIGEN_FILTER:
                # '256_from_652.npy'
                eigen_filter_mask = np.load('/mnt/lustre/wangjianyuan/f2sfm/256_from_652.npy')
                eigen_filter_idx = np.array(np.nonzero(eigen_filter_mask))[0]
                self.path_list= np.array(self.path_list)[eigen_filter_idx]
                

        # if cfg.SAVE_POSE:
        #     if self.train:
        #         gt_depth_path = '/mnt/lustre/wangjianyuan/data/KITTI_RAW/kitti_raw_dataset/jy_split/test_gt_depths.npz'
        #         self.gt_depth_list = np.load(gt_depth_path, fix_imports=True, encoding='latin1',allow_pickle=True)["data"]
        #         self.path_list  = kitti_readlines(test_files)
        #     else:
        #         gt_depth_path = '/mnt/lustre/wangjianyuan/data/KITTI_RAW/kitti_raw_dataset/jy_split/train_gt_depths.npz'
        #         self.gt_depth_list = np.load(gt_depth_path, fix_imports=True, encoding='latin1',allow_pickle=True)["data"]
        #         self.path_list = kitti_readlines(train_files)


        self.calib_dict = np.load('/mnt/lustre/wangjianyuan/f2sfm/calib_dict.npy',allow_pickle=True).item()
        self.pose_dict = np.load('/mnt/lustre/wangjianyuan/f2sfm/pose_dict.npy',allow_pickle=True).item()

        self.transform = transform
        self.target_transform = target_transform
        self.co_transform = co_transform
        self.photo_aug = ColorJitter.get_params((0.8, 1.2), (0.8, 1.2), (0.8, 1.2), (-0.1, 0.1))

        if cfg.FILTERED_PAIR and not self.train:
            if cfg.KITTI_697:
                # self.img1_path_list = np.load('/mnt/lustre/wangjianyuan/f2sfm/val_img1_list_0.2_697.npy',allow_pickle=True)
                # self.img1_path_list = np.load('/mnt/lustre/wangjianyuan/f2sfm/val_img1_list_0.3_697.npy',allow_pickle=True)
                # self.img1_path_list = np.load('/mnt/lustre/wangjianyuan/f2sfm/val_img1_list_0.4_697.npy',allow_pickle=True)
                # self.img1_path_list = np.load('/mnt/lustre/wangjianyuan/f2sfm/val_img1_list_0.5_697.npy',allow_pickle=True)
                self.img1_path_list = np.load('/mnt/lustre/wangjianyuan/f2sfm/val_img1_list_good_697.npy',allow_pickle=True)
            else:
                self.img1_path_list = np.load('/mnt/lustre/wangjianyuan/f2sfm/val_img1_list_good_652.npy',allow_pickle=True)
                if cfg.EIGEN_FILTER:
                    self.img1_path_list = np.load('/mnt/lustre/wangjianyuan/f2sfm/val_img1_list_good_256.npy',allow_pickle=True)
                    self.img1_path_list = self.img1_path_list[eigen_filter_idx]


    def __getitem__(self, index):

        folder, frame_id_2, _ = self.path_list[index].split()
        frame_id_2 = int(frame_id_2)

        offset = -1
        if cfg.RANDOM_OFFSET and random.random()>0.7:
            offset = -2
        if cfg.RANDOM_FW_BW and random.random()>0.5:
            offset = -offset


        time_name = os.path.basename(os.path.dirname(folder))
        calib = self.calib_dict[time_name]

        # reference frame
        img2_path = os.path.join(self.root, folder, "image_02/data", "{:010d}.png".format(frame_id_2))


        # target frame
        if cfg.FILTERED_PAIR and (not self.train):
            img1_path = self.img1_path_list[index]
        else:
            img1_path = os.path.join(self.root, folder, "image_02/data", "{:010d}.png".format(frame_id_2+offset))
            if not os.path.exists(img1_path):
                img1_path = os.path.join(self.root, folder, "image_02/data", "{:010d}.png".format(frame_id_2-offset))


        frame_id_1 = int(os.path.splitext(os.path.basename(img1_path))[0])

        seq_pose = self.pose_dict[os.path.basename(folder)]
        pose_1 = seq_pose[frame_id_1][:3,:]
        pose_2 = seq_pose[frame_id_2][:3,:]
        pose_fw = compute_deltaRT(pose_2,pose_1).astype(np.float32)
        pose_bw = compute_deltaRT(pose_1,pose_2).astype(np.float32)
        poses = [pose_fw, pose_bw]



        pred_poses = np.load(img2_path.replace('image_02','pred_poses_fb').replace('png','npy'))
        pred_poses = [pred_poses[0],pred_poses[1]]

        inputs = [img1_path,img2_path] 
        inputs = [cv2.imread(img)[:,:,::-1].astype(np.uint8) for img in inputs]



        ###################################################################
        gt_depth_dir = '/mnt/lustre/wangjianyuan/data/KITTI_Depth/depth_all'

        # gt_depth1_path = os.path.join(gt_depth_dir,os.path.basename(folder),'proj_depth/groundtruth/image_02','{:010d}.png'.format(frame_id_1))
        gt_depth2_path = os.path.join(gt_depth_dir,os.path.basename(folder),'proj_depth/groundtruth/image_02','{:010d}.png'.format(frame_id_2))
        

        if not os.path.exists(gt_depth2_path) or cfg.ALL_VELO_RAW:
            calib_dir = os.path.join(self.root, folder.split("/")[0])
            velo_filename = os.path.join(self.root, folder,"velodyne_points/data", "{:010d}.bin".format(frame_id_2))
            gt_depth2 = generate_depth_map(calib_dir, velo_filename, 2, True)
            gt_depth2 = np.expand_dims(gt_depth2,2).astype(np.float32)
            # gt_depth2 = generate_depth_map(calib_dir, velo_filename, 2, False)
        else:
            gt_depth2 = disparity_loader_png(gt_depth2_path)
        gt_depth1 = gt_depth2.copy()
        ###################################################################
        # gt_depth1 = np.expand_dims(self.gt_depth_list[index],2).astype(np.float32)
        # gt_depth2 = np.expand_dims(self.gt_depth_list[index],2).astype(np.float32)
        ###################################################################

        depth_gt = [gt_depth1, gt_depth2]


        if self.train:
            if cfg.FLIP_AUG:
                if random.random() > 0.75:
                    inputs[0] = np.flip(inputs[0],axis=1)
                    inputs[1] = np.flip(inputs[1],axis=1)
                    depth_gt[0] = np.flip(depth_gt[0],axis=1)
                    depth_gt[1] = np.flip(depth_gt[1],axis=1)

            if random.random() > 0.5:
                image_stack = np.concatenate([inputs[0], inputs[1]], axis=0)
                image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
                img1, img2 = np.split(image_stack, 2, axis=0)
                inputs[0] = img1.astype(np.float32);
                inputs[1] = img2.astype(np.float32)


        if self.co_transform is not None:
            inputs, depth_gt, calib = self.co_transform(inputs, depth_gt, calib)
        if self.transform is not None:
            inputs = self.transform(inputs)  
        if self.target_transform is not None:
            depth_gt = self.target_transform(depth_gt)

        if cfg.SAVE_POSE:
            return inputs, calib, poses,pred_poses, depth_gt,img2_path


        if cfg.GENERATE_KITTI_POSE_TO_SAVE:
            return inputs, calib, poses, depth_gt,seq_1,img_id_1
        else:
            return inputs, calib, poses,pred_poses, depth_gt#,img1_path,img2_path



    def __len__(self):
        return len(self.path_list)

    def check_pose(self, index):

        folder, frame_id_2, _ = self.path_list[index].split()
        frame_id_2 = int(frame_id_2)

        scale_list = []
        img1_path_list = []
        offset_list = [-1,1,-2,2,-3,3]

        for m in range(len(offset_list)):
            offset = offset_list[m]
            time_name = os.path.basename(os.path.dirname(folder))
            calib = self.calib_dict[time_name]

            img1_path = os.path.join(self.root, folder, "image_02/data", "{:010d}.png".format(frame_id_2+offset))
            img2_path = os.path.join(self.root, folder, "image_02/data", "{:010d}.png".format(frame_id_2))
            if not os.path.exists(img1_path):
                continue

            frame_id_1 = int(os.path.splitext(os.path.basename(img1_path))[0])

            seq_pose = self.pose_dict[os.path.basename(folder)]
            pose_1 = seq_pose[frame_id_1][:3,:]
            pose_2 = seq_pose[frame_id_2][:3,:]
            pose_fw = compute_deltaRT(pose_2,pose_1).astype(np.float32)
            pose_bw = compute_deltaRT(pose_1,pose_2).astype(np.float32)
            poses = [pose_fw, pose_bw]


            scale = np.linalg.norm(pose_fw[:,-1])   
            scale_list.append(scale)
            img1_path_list.append(img1_path)
            if scale>0.8 and scale<1.3:
                return img1_path, scale_list, True

        scale_list = np.array(scale_list)
        if scale_list[0]<=0.8:
            return_num = scale_list.argmax()
        else:
            return_num = scale_list.argmin()
        return img1_path_list[return_num],scale_list, False


    def eigen_filter(self, index):

        folder, frame_id_2, _ = self.path_list[index].split()
        frame_id_2 = int(frame_id_2)

        scale_list = []
        img1_path_list = []
        offset_list = [-1,1]

        flag = False

        for m in range(len(offset_list)):
            offset = offset_list[m]
            time_name = os.path.basename(os.path.dirname(folder))
            calib = self.calib_dict[time_name]

            img1_path = os.path.join(self.root, folder, "image_02/data", "{:010d}.png".format(frame_id_2+offset))
            img2_path = os.path.join(self.root, folder, "image_02/data", "{:010d}.png".format(frame_id_2))
            if not os.path.exists(img1_path):
                continue

            frame_id_1 = int(os.path.splitext(os.path.basename(img1_path))[0])

            seq_pose = self.pose_dict[os.path.basename(folder)]
            pose_1 = seq_pose[frame_id_1][:3,:]
            pose_2 = seq_pose[frame_id_2][:3,:]
            pose_fw = compute_deltaRT(pose_2,pose_1).astype(np.float32)
            pose_bw = compute_deltaRT(pose_1,pose_2).astype(np.float32)
            poses = [pose_fw, pose_bw]


            scale = np.linalg.norm(pose_fw[:,-1])   
            scale_list.append(scale)
            img1_path_list.append(img1_path)
            if scale>0.8 and scale<1.3:
                flag = True
                break
                
        return  folder,frame_id_2, img1_path, scale, flag
            # else:





def main():

    import flow_transforms

    input_transform = flow_transforms.Compose([   
        flow_transforms.ArrayToTensorCo(),
        flow_transforms.NormalizeCo(mean=[0,0,0], std=[255,255,255]),
        flow_transforms.NormalizeCo(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])   
    ])


    flow_transform = flow_transforms.Compose([ 
        flow_transforms.ArrayToTensorCo(),
        flow_transforms.NormalizeCo(mean=[0,0],std=[1,1])
        
    ])
    depth_transform = flow_transforms.Compose([
        flow_transforms.ArrayToTensorCo()
    ])

    # co_transform_train = flow_transforms.ComposeCo([
    #     flow_transforms.RandomCropCo((240,320)),
    # ])
    co_transform_val = None



    cfg.KITTI_697=True
    cfg.FILTERED_PAIR = False
    val_set = KITTIRAWLoaderGT(root='/mnt/lustre/wangjianyuan/data/KITTI_RAW/kitti_raw_dataset', transform=input_transform,target_transform=depth_transform,co_transform=co_transform_val, train=False)
    # val_set.path_list
    val_img_list = []
    scale_list_all = []
    valid_list  = []

    flag_list = []

    depth_gt_list = []
    # depth_gt_list = {}
    # 375
    # 382-387
    for i in range(len(val_set)):
        # val_set.__getitem__(i) 
        inputs, calib, poses,pred_poses, depth_gt = val_set.__getitem__(i) 
        ref_cv =inputs[0].cpu().numpy().transpose(1,2,0)[:,:,::-1]
        ref_cv = (ref_cv*0.5+0.5)*255
        # cv2.imwrite(inputs[0])
        path_1 = os.path.join('/mnt/lustre/wangjianyuan/f2sfm/image697',str(i)+'.png')
        cv2.imwrite(path_1,ref_cv)
        
        depth_gt_list.append((i,depth_gt[0].cpu().numpy()))
        ################################
        # folder,frame_id_2, img1_path, scale, flag = val_set.eigen_filter(i)
        # val_img_list.append(img1_path)
        # flag_list.append(flag)
        # import pdb;pdb.set_trace()
        ##############################
        # cur_img1,scale_list,valid  = val_set.check_pose(i)
        # val_img_list.append(cur_img1)

        # scale_list_all.append(scale_list)
        # valid_list.append(valid)
        ##############################

    # for (t_id, depth_gt) in depth_gt_list: print(t_id)  #print(depth_gt.shape) if depth_gt.shape[-1] not in [1242,1241,1224,1238] else None

    # for (t_id, depth_gt) in m: print(t_id)
    # np.save('gt_697.npy',depth_gt_list)
    # m=np.load('gt_697.npy',allow_pickle=True)
    import pdb;pdb.set_trace()


    # for m in range(len(scale_list_all)): print(np.array(scale_list_all[m]).max())
    # valid_list_np = np.array(valid_list);np.nonzero(~valid_list_np)
    # np.save('val_img1_list_0.5_697.npy',val_img_list)

    # flag_list_np = np.array(flag_list)

    # np.save('256_from_652.npy',flag_list_np)

    # np.save('val_img1_list_good_256.npy',val_img_list)
    import pdb;pdb.set_trace()

    # np.save('val_img1_list_good_652.npy',val_img_list)

                # self.img1_path_list = np.load('/mnt/lustre/wangjianyuan/f2sfm/val_img1_list_0.2_697.npy',allow_pickle=True)

    # val_set.path_list
    # img1_path_list = np.load('val_img1_list.npy',allow_pickle=True)
    # import pdb;pdb.set_trace()


    m=1


if __name__ == '__main__':
    main()
