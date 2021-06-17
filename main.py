import argparse
import os
import shutil
import time
import datetime

######################

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.autograd as autograd
import torchvision.transforms as transforms

######################

import numpy as np
from loss_functions import *
from models import SFMnet as SFMnet
import flow_transforms
from tensorboardX import SummaryWriter
from utils import tensor2array
import imageio as io
import logging
from lib.config import cfg, cfg_from_file, save_config_to_file
from demon_metrics import compute_motion_errors,l1_inverse,scale_invariant,abs_relative
from flow_training import train_flow
import random
from models.SFMnet import Pose2RT
from flow_viz import flow_to_image

from KITTI_loader import KITTIVOLoaderGT,KITTIRAWLoaderGT
from DEMON_loader import DEMON_GT_LOADER

###################### for mixed precision training

try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass

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

parser = argparse.ArgumentParser(description='Structure from Motion network',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data', metavar='DIR',help='path to train dataset')
parser.add_argument('--cfg', dest='cfg', default=None, type=str)

######################

parser.add_argument('--pretrained', dest='pretrained', default=None, metavar='PATH',
                            help='path to pre-trained SFMnet model')
parser.add_argument('--pretrained-flow', dest='pretrained_flow', default=None, metavar='PATH',
                            help='path to pre-trained epiflow model')
parser.add_argument('--pretrained-depth', dest='pretrained_depth', default=None, metavar='PATH',
                            help='path to pre-trained dspnet model')

parser.add_argument('--nlabel', type=int ,default=64, help='number of label')
parser.add_argument('--save-images', action='store_true',help='save validation images')

######################
parser.add_argument('--solver', default='adam',choices=['adam','sgd'],
                    help='solver algorithms')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epoch-size', default=0, type=int, metavar='N',
                    help='manual epoch size (will match dataset size if set to 0)')
parser.add_argument('-b', '--batch-size', default=4, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum for sgd, alpha parameter for adam')
parser.add_argument('--beta', default=0.999, type=float, metavar='M',
                    help='beta parameter for adam')
parser.add_argument('--weight-decay', '--wd', default=4e-4, type=float,
                    metavar='W', help='weight decay')

######################
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='logger.info frequency')
parser.add_argument('-v', '--validate', dest='validate', action='store_true',
                    help='evaluate model on validation set')

parser.add_argument('--div-flow', default=1,
                    help='value by which flow will be divided. Original value is 20 but 1 with batchNorm gives good results')
parser.add_argument('--milestones', default=[2,5,8], metavar='N', nargs='*', help='epochs at which learning rate is divided by 2')
parser.add_argument('--fix-flownet', dest='fix_flownet', action='store_true', help='do not train flownet')
parser.add_argument('--fix-depthnet', dest='fix_depthnet', action='store_true', help='do not train depthnet')


best_EPE = -1; n_iter = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    global args, best_EPE, save_path,n_iter
    args = parser.parse_args()

    if args.cfg is not None:
        cfg_from_file(args.cfg)
        assert cfg.TAG == os.path.splitext(os.path.basename(args.cfg))[0], 'TAG name should be file name'

    save_path = os.path.join("output",cfg.TAG)
    if not os.path.exists(save_path): os.makedirs(save_path)
    
    # set log files
    log_file = os.path.join(save_path, 'log_train.txt')
    logger = create_logger(log_file)
    logger.info('**********************Start logging**********************')
    logger.info('=> will save everything to {}'.format(save_path))

    # save configs for future reference
    for _,key in enumerate(args.__dict__):
        logger.info(args.__dict__[key])
    save_config_to_file(cfg, logger=logger)

    scaler = GradScaler(enabled=cfg.MIXED_PREC)

    #### set Tensorboard
    train_writer = SummaryWriter(os.path.join(save_path,'train'))
    test_writer = SummaryWriter(os.path.join(save_path,'test'))

    # image loading, normalize
    input_transform = flow_transforms.Compose([   
        flow_transforms.ArrayToTensorCo(),
        flow_transforms.NormalizeCo(mean=[0,0,0], std=[255,255,255]),
        flow_transforms.NormalizeCo(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]) ])

    depth_transform = flow_transforms.Compose([flow_transforms.ArrayToTensorCo()])

    # set image size 
    co_transform_train = flow_transforms.ComposeCo([
        flow_transforms.RandomCropCo((cfg.TRAIN_SIZE[0],cfg.TRAIN_SIZE[1])),])
    co_transform_val = flow_transforms.ComposeCo([
        flow_transforms.CenterCropCo((cfg.VAL_SIZE[0], cfg.VAL_SIZE[1])),])

    logger.info("=> fetching img pairs in '{}'".format(args.data))


    #### data loader
    if cfg.KITTI_RAW_DATASET:
        train_set = KITTIRAWLoaderGT(root=args.data, transform=input_transform,target_transform=depth_transform,co_transform=co_transform_train, train=True)
        val_set = KITTIRAWLoaderGT(root=args.data, transform=input_transform,target_transform=depth_transform,co_transform=co_transform_val, train=False)
    elif cfg.DEMON_DATASET:
        if not args.validate: train_set = DEMON_GT_LOADER(root=args.data, transform=input_transform,target_transform=depth_transform,co_transform=co_transform_train, train=True,ttype='train.txt')
        val_set = DEMON_GT_LOADER(root=args.data, transform=input_transform,target_transform=depth_transform,co_transform=co_transform_val, train=False,ttype='test.txt')
    else:
        train_set = KITTIVOLoaderGT(root=args.data, transform=input_transform,target_transform=depth_transform,co_transform=co_transform_train, train=True)
        val_set = KITTIVOLoaderGT(root=args.data, transform=input_transform,target_transform=depth_transform,co_transform=co_transform_val, train=False)

    if cfg.GENERATE_DEMON_POSE_TO_SAVE or cfg.GENERATE_DEMON_POSE_OR_DEPTH:
        val_set = DEMON_GT_LOADER(root=args.data, transform=input_transform,target_transform=depth_transform,co_transform=co_transform_val, train=False,ttype='train.txt',return_path=True)

    try:
        logger.info('{} samples found, {} train samples and {} test samples '.format(len(val_set)+len(train_set),len(train_set),len(val_set)))
    except:
        logger.info('{} test samples '.format(len(val_set)))
    if not args.validate:
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size,
            num_workers=args.workers, pin_memory=True, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size,
        num_workers=args.workers, pin_memory=True, shuffle=False)



    ##################################################################


    # create model and load
    # nlabel is the number of plane candidates, for plane sweeping
    model = SFMnet(args.nlabel)

    if args.pretrained:
        network_data = torch.load(args.pretrained)
        logger.info("=> using pre-trained model '{}'".format(args.pretrained))
        model.load_state_dict(network_data["state_dict"],strict=False)
    else:
        network_data = None
        logger.info("=> creating new model")

    # optimizer
    assert(args.solver in ['adam', 'sgd'])
    logger.info('=> setting {} solver'.format(args.solver))
    if args.solver == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.lr, betas=(args.momentum, args.beta), weight_decay=args.weight_decay)
    elif args.solver == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError


    model = torch.nn.DataParallel(model).cuda()


    if args.pretrained_flow:
        temp_dict = {}
        pretrained_dict = torch.load(args.pretrained_flow)
        # TBD: remove these dummy codes
        flag = False
        for key in pretrained_dict['state_dict'].keys():   
            if 'flow_estimator' in key:
                flag = True; temp_dict[key.replace('flow_estimator.','')] = pretrained_dict['state_dict'][key]
        if flag: pretrained_dict['state_dict'] = temp_dict

        model_dict = model.module.flow_estimator.state_dict()
        model_dict.update(pretrained_dict)
        model.module.flow_estimator.load_state_dict(model_dict['state_dict'],strict=False)
        logger.info("=> using pre-trained flow network: '{}'".format(args.pretrained_flow))

    if args.pretrained_depth:
        pretrained_dict = torch.load(args.pretrained_depth)
        model_dict = model.module.depth_estimator.state_dict()
        model_dict.update(pretrained_dict)
        model.module.depth_estimator.load_state_dict(model_dict['state_dict'],strict=False)
        logger.info("=> using pre-trained dpsnet: '{}'".format(args.pretrained_depth))

    cudnn.benchmark = True
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.MILESTONES, gamma=0.5)

    ### validate
    if cfg.GENERATE_DEMON_POSE_TO_SAVE:
        from generate_demon_pose import generate_DEMON_pose
        with torch.no_grad():
            best_EPE = generate_DEMON_pose(val_loader, model, epoch,logger)
        return

    if args.validate:
        with torch.no_grad():
            if cfg.SAVE_POSE:
                best_EPE = save_pose(val_loader, model, epoch,logger)
            else:
                best_EPE = validate(val_loader, model, epoch,logger)
        return

    ##################################################################

    for epoch in range(args.start_epoch, args.epochs):
        # fix the weights
        if args.fix_flownet:
            for fparams in model.module.flow_estimator.parameters():  fparams.requires_grad = False
        if args.fix_depthnet:
            for fparams in model.module.depth_estimator.parameters():  fparams.requires_grad = False

        is_best = False


        if cfg.TRAIN_FLOW:
            # train optical flow on Demon datasets
            train_loss,n_iter = train_flow(train_loader, model, optimizer, epoch, 
                                            train_writer,scaler,logger=logger,n_iter=n_iter,args=args)
        else:
            # train depth
            train_loss,scheduler = train_epoch(train_loader, model, optimizer, epoch, 
                                            train_writer,scaler,logger,scheduler=scheduler)

        scheduler.step()

        save_checkpoint({'epoch': epoch + 1,'state_dict': model.module.state_dict()},
                        is_best, filename='checkpoint{}.pth.tar'.format(epoch))

        with torch.no_grad():
            best_EPE = validate(val_loader, model, epoch,logger)



def train_epoch(train_loader, model, optimizer, epoch, train_writer,scaler,logger=None,scheduler=None):
    global n_iter, args
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    epoch_size = len(train_loader) if args.epoch_size == 0 else min(len(train_loader), args.epoch_size)


    # switch to train mode
    model.train()
    end = time.time()

    for i, (input, intrinsics, poses, pred_poses, depth_gt) in enumerate(train_loader):
        optimizer.zero_grad()
        data_time.update(time.time() - end)

        # two frames
        input0 = input[0].to(device); input1 = input[1].to(device)

        # add random noise
        stdv = np.random.uniform(0.0, 3.0/255)
        input0 = (input0 + stdv * torch.randn(*input0.shape).cuda()).clamp(-1, 1)
        input1 = (input1 + stdv * torch.randn(*input1.shape).cuda()).clamp(-1, 1)

        # forward and backward pose
        pose_gt_fw = poses[0].to(device);pose_gt_bw = poses[1].to(device)

        # we could save predicted poses as local files, to reduce training time 
        pred_pose_fw = pred_poses[0].to(device) if pred_poses is not None else None
        pred_pose_bw = pred_poses[1].to(device) if pred_poses is not None else None


        depth_fw_gt = depth_gt[0].to(device); depth_bw_gt = depth_gt[1].to(device)

        
        raw_shape = input1.shape; height_raw = raw_shape[2]; width_raw= raw_shape[3]

        # DICLFlow limits the size of input frames, so we pad the inputs
        # feel free to delete this part if using other flow estimator
        height_new = int(np.ceil(raw_shape[2]/128)*128)
        width_new  = int(np.ceil(raw_shape[3]/128)*128)
        padding = (0, width_new-raw_shape[3], 0, height_new-raw_shape[2])
        input1 = torch.nn.functional.pad(input1, padding, "replicate", 0)
        input0 = torch.nn.functional.pad(input0, padding, "replicate", 0)


        # conduct training in a backward way
        # plz note backward here indicates from input1 (target frame) to input0 (reference frame)
        # rather than back-propagation
        flow_2D_bw, pose_bw, depth_bw, depth_bw_init, rot_and_trans = model(input1, input0, intrinsics, 
                                                                    pose_gt_bw, pred_pose_bw, cfg.GT_POSE, 
                                                                    h_side = raw_shape[2], w_side = raw_shape[3],logger=logger)

        intrinsics_invs = intrinsics.inverse().float().to(device)
        intrinsics = intrinsics.float().to(device)

        # filter out meaningless depth values (out of range, NaN or Inf)
        mask_bw = (depth_bw_gt <= args.nlabel*cfg.MIN_DEPTH) & (depth_bw_gt >= cfg.MIN_DEPTH) & (depth_bw_gt == depth_bw_gt)
        mask_fw = (depth_fw_gt <= args.nlabel*cfg.MIN_DEPTH) & (depth_fw_gt >= cfg.MIN_DEPTH) & (depth_fw_gt == depth_fw_gt)


        if not args.fix_depthnet:
            ##################  Compute Depth Loss ########################
            with autocast(enabled=cfg.MIXED_PREC):
                if cfg.RESCALE_DEPTH:
                    # the translation scale of ground truth poses
                    scale = torch.norm(pose_gt_bw[:,:,-1:].squeeze(-1),dim=-1)
                    scale_mask = (scale> cfg.MIN_TRAIN_SCALE) & (scale< cfg.MAX_TRAIN_SCALE)
                    rescale_ratio = scale/cfg.NORM_TARGET

                    # to keep scale consistent
                    depth_bw = depth_bw*rescale_ratio.view(scale.shape[0],1,1,1)

                    if cfg.RESCALE_DEPTH_REMASK:
                        # recheck the boundary of scales, unnecessary
                        depth_bw_gt_rescale = depth_bw_gt/(rescale_ratio.view(-1,1,1,1))
                        mask_bw = (depth_bw_gt_rescale <= args.nlabel*cfg.MIN_DEPTH) & (depth_bw_gt_rescale >= cfg.MIN_DEPTH) & (depth_bw_gt_rescale == depth_bw_gt_rescale)
                        train_writer.add_scalar('max_gt_depth', depth_bw_gt_rescale.max().item(), n_iter)
                        train_writer.add_scalar('min_gt_depth', depth_bw_gt_rescale[depth_bw_gt_rescale>0].min().item(), n_iter)
                        mask_bw = mask_bw.detach()

                    # pick the valid ones for optimization
                    pred_init_toloss = depth_bw_init[scale_mask][mask_bw[scale_mask]]
                    pred_toloss = depth_bw[scale_mask][mask_bw[scale_mask]]
                    gt_toloss = depth_bw_gt[scale_mask][mask_bw[scale_mask]]
                else:
                    # pick the valid ones for optimization
                    scale = torch.norm(pose_gt_bw[:,:,-1:].squeeze(-1),dim=-1)
                    scale_mask = (scale> cfg.MIN_TRAIN_SCALE)
                    pred_init_toloss = depth_bw_init[scale_mask][mask_bw[scale_mask]]
                    pred_toloss = depth_bw[scale_mask][mask_bw[scale_mask]]
                    gt_toloss = depth_bw_gt[scale_mask][mask_bw[scale_mask]]

                # follow the setting of DPSNet
                loss_depth_init = 0.7*(F.smooth_l1_loss(pred_init_toloss, gt_toloss))
                loss_depth_out = F.smooth_l1_loss(pred_toloss, gt_toloss)
                loss_depth = loss_depth_out + loss_depth_init
                train_writer.add_scalar('depth_init', loss_depth_init.item(), n_iter)
                train_writer.add_scalar('depth_out', loss_depth_out.item(), n_iter)


            loss = loss_depth

        if rot_and_trans is not None:
            ##################  Compute Pose Loss ########################
            ### if use deep pose regression
            gt_rt_raw = Pose2RT(pose_gt_bw)
            gt_rt = torch.cat((gt_rt_raw[:,:3],F.normalize(gt_rt_raw[:, 3:])),dim=1)
            
            loss_fn = torch.nn.MSELoss(reduction='none')
            pose_loss = loss_fn(rot_and_trans,gt_rt).mean(dim=0)
            pose_loss[:3] = pose_loss[:3]*20
            pose_loss = pose_loss.mean()

            train_writer.add_scalar('pose_loss', pose_loss.mean().item(), n_iter)
            loss = loss + pose_loss.mean()

        # check if there are extreme values
        if loss>9999:   import pdb;pdb.set_trace()
            
        losses.update(loss.item(), input0.size(0))
        train_writer.add_scalar('train_loss', loss, n_iter)

        cur_lr = optimizer.param_groups[0]['lr']
        train_writer.add_scalar('cur_lr', cur_lr, n_iter)

        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        
        batch_time.update(time.time() - end)
        end = time.time()

        # recording and visualization
        if i % args.print_freq == 0:
            input0=0.5 + (input0[0])*0.5; input1=0.5 + (input1[0])*0.5
            train_writer.add_image(('left'+str(0)),input0,n_iter)
            train_writer.add_image(('right'+str(0)),input1,n_iter)

            # for visualization
            disp_bw = [cfg.MIN_DEPTH*args.nlabel/(depth_bw_.detach().cpu())]
            disp_bw_gt = [cfg.MIN_DEPTH*args.nlabel/(depth_bw_gt.squeeze(1).detach().cpu())]

            if flow_2D_bw is not None: 
                flo_bw_raw_vis=flow2rgb_raw(flow_2D_bw[0],max_value=128)
                train_writer.add_image(('flo_bw_raw'),flo_bw_raw_vis,n_iter) 

            for j in range(len(disp_bw)): 
                disp_bw_to_show=tensor2array(disp_bw[j][0], max_value=80, colormap='bone')
                disp_bw_gt_to_show=tensor2array(disp_bw_gt[j][0], max_value=80, colormap='bone')
                
                train_writer.add_image(('depth_bw'+str(j)),disp_bw_to_show,n_iter)  
                train_writer.add_image(('depth_bw_gt'+str(j)),disp_bw_gt_to_show,n_iter) 

            logger.info('Epoch: [{0}][{1}/{2}]\t Time {3}\t Data {4}\t Loss {5}'
                  .format(epoch, i, epoch_size, batch_time, data_time, losses))

        n_iter += 1
        if i >= epoch_size:
            break

    return losses.avg, scheduler



def validate(val_loader, model, epoch,logger=None):
    global args
    batch_time = AverageMeter(); depth_EPEs = AverageMeter()

    abs_rel_t = AverageMeter(); sq_rel_t = AverageMeter(); rmse_t = AverageMeter(); rmse_log_t = AverageMeter()
    a1_t = AverageMeter(); a2_t = AverageMeter(); a3_t = AverageMeter(); d1_all_t = AverageMeter()
    l1_inv_t =AverageMeter(); sc_inv_t =AverageMeter()

    # switch to evaluate mode
    import matplotlib

    model.eval()
    end = time.time()

    dis_list = []; rel_dis_list = []; scale_list = []; change_ratio_list = []
    errors_fw_l = []; errors_bw_l = []; depth_list = []; epe_tmp =[]

    for i, (inputs, intrinsics, poses, pred_poses, depth_gt) in enumerate(val_loader):

        input0 = inputs[0].to(device)
        input1 = inputs[1].to(device)
        pose_gt_fw = poses[0].to(device)
        pose_gt_bw = poses[1].to(device)
        depth_fw_gt = depth_gt[0].to(device)
        depth_bw_gt = depth_gt[1].to(device)
        pred_pose_fw = pred_poses[0].to(device) if pred_poses is not None else None
        pred_pose_bw = pred_poses[1].to(device) if pred_poses is not None else None

        intrinsics_invs = intrinsics.inverse().float().to(device)
        intrinsics = intrinsics.float().to(device)
        
        raw_shape = input1.shape
        height_raw = raw_shape[2]; width_raw= raw_shape[3]

        # if the flow estimation module is not 'DICL', you could modify the codes below accordingly
        height_new = int(np.ceil(raw_shape[2]/128)*128)
        width_new  = int(np.ceil(raw_shape[3]/128)*128)
        padding = (0, width_new-raw_shape[3], 0, height_new-raw_shape[2])
        input1 = torch.nn.functional.pad(input1, padding, "replicate", 0)
        input0 = torch.nn.functional.pad(input0, padding, "replicate", 0)

        #######################################
        b, _, h, w = input0.size()

        if cfg.RECORD_POSE:
            # compute forward and backward poses
            pose_fw, flow_2D_fw = model(input0, input1, intrinsics, pose_gt_fw, pred_pose_fw, cfg.GT_POSE,raw_shape[2],raw_shape[3])
            pose_bw, flow_2D_bw = model(input1, input0, intrinsics, pose_gt_bw, pred_pose_bw, cfg.GT_POSE,raw_shape[2],raw_shape[3])
            
            def pose_to_motion(pose):
                # convert pose matrix to motion
                rt = Pose2RT(pose)
                out = torch.cat((rt[:,:3],F.normalize(rt[:, 3:])),dim=1)
                return out[0].cpu().numpy()
            
            for bat in range(len(pose_fw)):
                motion_fw = pose_to_motion(pose_fw[bat])
                motion_bw = pose_to_motion(pose_bw[bat])

                motion_gt_fw = pose_to_motion(pose_gt_fw[bat].unsqueeze(0))
                motion_gt_bw = pose_to_motion(pose_gt_bw[bat].unsqueeze(0))

                error_fw = compute_motion_errors(motion_fw, motion_gt_fw, True)
                error_bw = compute_motion_errors(motion_bw, motion_gt_bw, True)

                errors_fw_l.append(np.array(error_fw))
                errors_bw_l.append(np.array(error_bw))

            errors_fw_mean = np.array(errors_fw_l).mean(axis=0)
            logger.info('Pose Error: [{0}/{1}]\t Time {2}\t error rot {3} t {4} trans {5}'.format(i, len(val_loader), batch_time, errors_fw_mean[0], errors_fw_mean[1], errors_fw_mean[2]))
            continue
        else:
            # conduct inference, both depth and pose
            flow_2D_bw, pose_bw, depth_bw, time_dict = model(input1, input0, intrinsics, pose_gt_bw,pred_pose_bw, cfg.GT_POSE,raw_shape[2],raw_shape[3])


        if cfg.RESCALE_DEPTH:
            #### could skip this step during inference
            batch_num = len(depth_bw_gt)
            scale = torch.norm(pose_gt_bw[:,:,-1:].squeeze(-1),dim=-1)
            rescale_ratio = scale/cfg.NORM_TARGET
            depth_bw = depth_bw*rescale_ratio.view(batch_num,1,1,1)

        depth_bw = depth_bw[:,:,:height_raw,:width_raw]

        if flow_2D_bw is not None:
            flow_2D_bw = flow_2D_bw[:,:,:height_raw,:width_raw]
            flow_2D_plot = flow2rgb_raw(flow_2D_bw[0], max_value=128)
            flow_2D_plot = np.transpose(flow_2D_plot, (1, 2, 0))

        ###############################################################
        ### dummy codes
        ### possibly helpful if someone would like to save visualization
        # if args.save_images:
        #     ref_cv =inputs[0][0].cpu().numpy().transpose(1,2,0)[:,:,::-1]
        #     tar_cv =inputs[1][0].cpu().numpy().transpose(1,2,0)[:,:,::-1]

        #     ref_cv = (ref_cv*0.5+0.5)*255;tar_cv = (tar_cv*0.5+0.5)*255
        #     cv2.imwrite(path_1,ref_cv)
        #     realflow_vis = flow_to_image(flow_2D_bw[0].cpu().detach().numpy().transpose(1,2,0),None)
        #     cv2.imwrite(path_2,tar_cv)
        #     cv2.imwrite(path_pred,realflow_vis)
        ###############################################################

        # the same threshold and masking strategy as used by previous methods
        if cfg.DEMON_DATASET:
            mask_bw = (depth_bw_gt <= 10) & (depth_bw_gt >= 0.5) & (depth_bw_gt == depth_bw_gt)
        else:
            mask_bw = (depth_bw_gt>0) & (depth_bw_gt<80)
            crop_mask = mask_bw.clone()
            crop_mask[:] = 0
            gt_height, gt_width = mask_bw.shape[2:]
            crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
            crop_mask[:,:,crop[0]:crop[1], crop[2]:crop[3]] = 1
            mask_bw = mask_bw & crop_mask

        ### median scale shift, because of the scale ambiguity
        median_scale_list = []
        for b_idx in range(len(depth_bw_gt)): 
            try:
                cur_median_scale = (depth_bw_gt[b_idx][mask_bw[b_idx]].median()/depth_bw[b_idx][mask_bw[b_idx]].median()).detach(); 
            except:
                cur_median_scale = depth_bw_gt[b_idx].median()/depth_bw[b_idx].median()
            median_scale_list.append(cur_median_scale)
        median_scale = torch.FloatTensor(median_scale_list).to(depth_bw.device).type_as(depth_bw)
        depth_bw = depth_bw * (median_scale.view(-1,1,1,1))

        # check bound
        max_range = cfg.MIN_DEPTH * args.nlabel 
        depth_bw[depth_bw<=cfg.MIN_DEPTH] = cfg.MIN_DEPTH
        depth_bw[depth_bw>max_range] = max_range

        # compute errors
        abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3, l1_inv, sc_inv = evaluate_metric(depth_bw_gt[mask_bw].cpu().numpy(),depth_bw[mask_bw].detach().cpu().numpy())
        
        # record errors
        abs_rel_t.update(abs_rel, intrinsics.size(0))
        sq_rel_t.update(sq_rel, intrinsics.size(0))
        rmse_t.update(rmse, intrinsics.size(0))
        rmse_log_t.update(rmse_log, intrinsics.size(0))
        a1_t.update(a1, intrinsics.size(0)); a2_t.update(a2, intrinsics.size(0)); a3_t.update(a3, intrinsics.size(0))
        l1_inv_t.update(l1_inv, intrinsics.size(0)); sc_inv_t.update(sc_inv, intrinsics.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            if args.print_freq==1:
                # print per-frame detail, for debugging
                scale = torch.norm(pose_gt_bw[:,:,-1],dim=-1)
                logger.info('Test: [{0}/{1}]\t Scale {2}\t abs_rel {3} l1_inv {4} sc_inv {5}'.format(i, len(val_loader), scale.mean().item(), abs_rel_t,l1_inv_t,sc_inv_t))

            else:
                logger.info('Test: [{0}/{1}]\t Time {2}\t abs_rel {3}'.format(i, len(val_loader), batch_time,abs_rel_t))


        if args.save_images:
            visual_path = os.path.join(save_path,'val_visual')
            if not os.path.exists(visual_path):
                os.makedirs(visual_path)
            depth_plot = depth_bw[0].detach().cpu().numpy().squeeze()
            matplotlib.image.imsave(visual_path+"/{0:06}.png".format(i),np.uint16(depth_plot*256))
            if flow_2D_bw is not None: matplotlib.image.imsave(visual_path+"/{0:06}_flow2d.png".format(i),flow_2D_plot)


    if cfg.RECORD_POSE:
        logger.info('Forward Error:'); logger.info(np.array(errors_fw_l).mean(axis=0))
        logger.info('Backward Error:'); logger.info(np.array(errors_bw_l).mean(axis=0))
    else:
        logger.info('abs_rel {0} sq_rel {1} rmse {2} rmse_log {3} a1 {4} a2 {5} a3 {6} l1_inv {7} sc_inv {8} d1_all {9}'.format(abs_rel_t.avg,sq_rel_t.avg,rmse_t.avg,rmse_log_t.avg,a1_t.avg,a2_t.avg,a3_t.avg,l1_inv_t.avg,sc_inv_t.avg,d1_all_t.avg))
    return depth_EPEs.avg




def save_pose(val_loader, model, epoch, logger=None):
    '''
    save sequence pose for evaluation
    '''
    global args
    batch_time = AverageMeter(); depth_EPEs = AverageMeter()

    abs_rel_t = AverageMeter(); sq_rel_t = AverageMeter(); rmse_t = AverageMeter(); rmse_log_t = AverageMeter()
    a1_t = AverageMeter(); a2_t = AverageMeter(); a3_t = AverageMeter()
    l1_inv_t =AverageMeter(); sc_inv_t =AverageMeter()

    # switch to evaluate mode
    model.eval()
    import matplotlib
    end = time.time()

    dis_list = []; rel_dis_list = []; scale_list = []; change_ratio_list = []

    errors_fw_l = []; errors_bw_l = []

    for i, (inputs, intrinsics, poses, pred_poses, depth_gt,img2_path) in enumerate(val_loader):
        input0 = inputs[0].to(device)
        input1 = inputs[1].to(device)
        pose_gt_fw = poses[0].to(device)
        pose_gt_bw = poses[1].to(device)
        depth_fw_gt = depth_gt[0].to(device)
        depth_bw_gt = depth_gt[1].to(device)
        pred_pose_fw = pred_poses[0].to(device) if pred_poses is not None else None
        pred_pose_bw = pred_poses[1].to(device) if pred_poses is not None else None

        intrinsics_invs = intrinsics.inverse().float().to(device)
        intrinsics = intrinsics.float().to(device)

        #######################################
        raw_shape = input1.shape
        height_raw = raw_shape[2]; width_raw= raw_shape[3]

        # if the flow estimation module is not 'DICL', you could modify the codes below accordingly
        height_new = int(np.ceil(raw_shape[2]/128)*128)
        width_new  = int(np.ceil(raw_shape[3]/128)*128)
        padding = (0, width_new-raw_shape[3], 0, height_new-raw_shape[2])
        input1 = torch.nn.functional.pad(input1, padding, "replicate", 0)
        input0 = torch.nn.functional.pad(input0, padding, "replicate", 0)

        b, _, h, w = input0.size()

        pose_fw, flow_2D_raw_fw = model(input0, input1, intrinsics, pose_gt_fw, pred_pose_fw, cfg.GT_POSE,raw_shape[2],raw_shape[3])
        pose_bw, flow_2D_raw_bw = model(input1, input0, intrinsics, pose_gt_bw, pred_pose_bw, cfg.GT_POSE,raw_shape[2],raw_shape[3])
        
        # save the predicted poses as a numpy file, each corresponds to a sequence
        for batch_idx in range(len(input0)):
            pred_poses_fb = torch.cat((pose_fw[batch_idx],pose_bw[batch_idx])).cpu().numpy()
            np_save_path = img2_path[batch_idx].replace('.png','.npy').replace('image_02','pred_poses_fb')

            if not os.path.exists(os.path.dirname(np_save_path)):
                os.makedirs(os.path.dirname(np_save_path))
            np.save(np_save_path,pred_poses_fb)
            logger.info('SAVE POSE: [{0}/{1}]\t Time {2}\t '.format(i, len(val_loader), batch_time))
        continue

    if cfg.RECORD_POSE:
        logger.info(cfg.DEMON_DATASET_SPE)
        logger.info('Forward Error:'); logger.info(np.array(errors_fw_l).mean(axis=0))
        logger.info('Backward Error:'); logger.info(np.array(errors_bw_l).mean(axis=0))
    else:
        logger.info('abs_rel {0} sq_rel {1} rmse {2} rmse_log {3} a1 {4} a2 {5} a3 {6} l1_inv {7} sc_inv {8}'.format(abs_rel_t.avg,sq_rel_t.avg,rmse_t.avg,rmse_log_t.avg,a1_t.avg,a2_t.avg,a3_t.avg,l1_inv_t.avg,sc_inv_t.avg))

    return depth_EPEs.avg


############################################## Utility ##############################################

def create_logger(log_file):
    log_format = '%(asctime)s  %(levelname)5s  %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format, filename=log_file)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(log_format))
    logging.getLogger(__name__).addHandler(console)
    return logging.getLogger(__name__)


def convert_disps_to_depths_kitti(gt_disp, intrinsic):
    focus = intrinsic[:,0,0]
    shape = [*gt_disp.size()]
    mask = (gt_disp > 0).float()
    focus = focus.view(shape[0],1,1).repeat(1,shape[1],shape[2]).float()
    gt_depth   = (focus * 0.54) / (gt_disp + (1.0 - mask))
    return gt_depth * mask


def evaluate_metric(gt, pred):  
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25   ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred)**2) / gt)
    
    l1_inv = l1_inverse(gt,pred)

    sc_inv = scale_invariant(gt,pred)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3,l1_inv,sc_inv



class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return '{:.3f} ({:.3f})'.format(self.val, self.avg)

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(save_path,filename))
    if is_best:
        shutil.copyfile(os.path.join(save_path,filename), os.path.join(save_path,'model_best.pth.tar'))

def flow2rgb_raw(flow_map, max_value):
    flow_map_np = flow_map.detach().cpu().numpy()
    _, h, w = flow_map_np.shape
    flow_map_np[:,(flow_map_np[0] == 0) & (flow_map_np[1] == 0)] = float('nan')
    rgb_map = np.ones((3,h,w)).astype(np.float32)
    if max_value is not None:
        normalized_flow_map = flow_map_np / max_value
    else:
        normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max())
    rgb_map[0] += normalized_flow_map[0]
    rgb_map[1] -= 0.5*(normalized_flow_map[0] + normalized_flow_map[1])
    rgb_map[2] += normalized_flow_map[1]
    return rgb_map.clip(0,1)

if __name__ == '__main__':
    main()