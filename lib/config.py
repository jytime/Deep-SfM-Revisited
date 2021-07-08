from easydict import EasyDict as edict
import numpy as np
import torch.nn as nn

__C = edict()
cfg = __C


### Define config flags here
### Some flags are dummy, would be removed later 



### Name of the config
__C.TAG = 'default'


### Training and validation

__C.TRAIN_SIZE = [256,512]
__C.VAL_SIZE = [370,1224]
__C.MIN_DEPTH = 1.0
__C.FLIP_AUG = False

__C.EIGEN_SFM =  False
__C.ZOOM_INPUT = False

__C.SAVE_POSE = False
__C.MILESTONES = [2,5,8]


__C.TRAIN_FLOW = False
__C.STORED_POSE = False


__C.NORM_TARGET = 0.8

__C.PRED_POSE_ONLINE = True

### Deep PSNet, used as our depth estimation module

__C.PSNET_CONTEXT = True
__C.PSNET_DEP_CONTEXT = False

### RANSAC 

__C.POSE_EST = 'RANSAC'

__C.ransac_iter = 5
__C.ransac_threshold = 1e-4
__C.min_matches = 20

### Deep Pose regression, for ablation study

__C.POSE_NET_TYPE =  'plain'
__C.POSE_DOWN_FEAT =  128
__C.POSENET_FLOW = False
__C.POSENET_ENTRO= False
__C.POSE_WITH_BN = True



###

__C.GENERATE_DEMON_POSE_OR_DEPTH =  False

__C.ALL_VELO_RAW = False

__C.NO_MASK = False

__C.NO_SIFT = False

__C.TRUNC_SOFT = False

__C.KITTI_697 = True

__C.RANDOM_FW_BW = False

__C.RANDOM_OFFSET = False

__C.FILTERED_PAIR = True

__C.COST_BY_COLOR = False

__C.COST_BY_COLOR_WITH_FEAT = False

__C.PREDICT_BY_DEPTH = False

__C.NOT_CROP_VAL = False


__C.FILTER_OUT_RGBD = False

__C.KITTI_RAW_DATASET = False



__C.FILTER_DEMON_DATASET = False

__C.FILTER_DEMON_DATASET_FT = False



__C.FLOW_MASK = False

__C.GENERATE_DEMON_POSE_TO_SAVE = False

__C.DEMON_GENERATED_IDX = 0


__C.GENERATE_KITTI_POSE_TO_SAVE = False




__C.DEMON_DATASET = False

__C.DEMON_DATASET_SPE = 'None'

__C.FLOW_SPLIT_TRAIN = False


__C.SEQ_LEN = 5


__C.RESCALE_DEPTH = False

__C.RESCALE_DEPTH_REMASK = False

__C.REL_ABS_LOSS = False

__C.MIN_TRAIN_SCALE = 0.2

__C.MAX_TRAIN_SCALE = 3.0



__C.POSE_SEQ = [9]

__C.PRED_POSE_GT_SCALE = False


__C.RECORD_POSE = False

__C.RECORD_POSE_EVAL = False


__C.PRED_POSE_VAL_ONLINE = False

__C.CHECK_WRONG_POSE = False



__C.CONTEXT_BN = False

__C.FIX_DEPTH = False

__C.SUP_INIT = True

__C.IND_CONTEXT = False


__C.POSE_AWARE_MAX = False


__C.VALIDATE_FW = False

__C.MIXED_PREC = False



__C.NO_SMOOTH = True

__C.FLOW_EST = 'DICL'


__C.DEPTH_EST = 'PSNET'


__C.SCALE_MIN = 0.9

__C.SCALE_MAX = 1.1

__C.SCALE_STEP = 0.025

__C.FLOW_AND_JOINT_LOSS = False

__C.POSE_AWARE_MEAN = False


__C.SKIP = 1


__C.GT_POSE = False

__C.GT_POSE_NORMALIZED = False

__C.FLOW_POSE = True

__C.FLOW_BY_SIFT = False

__C.SIFT_POSE = False

__C.FLOW_CONF = -1.0

__C.SAMPLE_SP = False



####################################################################################

### Configs for DICL Flow
### To Be Removed Soon


__C.MAX_DISP = [[6,6],[6,6],[6,6]]
__C.MIN_DISP = [[-6,-6],[-6,-6],[-6,-6]]
__C.SOFT2D = True
__C.FLOWAGG = True
__C.COST_TOGETHER = False

__C.RANDOM_TRANS = 10

__C.DOWN_FEAT = False

__C.SPARSE_RESIZE = True

__C.KITTI_REMOVE130 = False

__C.SMOOTH_BY_TEMP = False

__C.CORR_BY_COS = False


__C.CLAMP_INPUT = True


__C.MIN_SCALE = 128

__C.UP_TO_RAW_SCALE = False


__C.KITTI_NO_VALID = False

__C.RAW_SINTEL_RATIO = 5


__C.USE_PCA_AUG = False

__C.SHALLOW_DOWN_SMALL = False

__C.BASIC_WITH_LEAKYRELU = False

__C.RAFT_RESIZE_CV2 = True


__C.MATCH_WITHDIS = False

__C.PAD_BY_CONS = False

__C.PAD_CONS = -1


__C.RAW_THING = False


__C.asymmetric_color_aug = False

__C.WEIGHT_DECAY = 0.0
__C.UPCONV = True
__C.DETACH_FUSION = False

__C.USE_CONTEXT6 = True

__C.USE_CONTEXT5 = True

__C.USE_SUBLOSS = False
__C.SUBLOSS_W = 0.001



__C.SHALLOW_SHARE = False
__C.SHALLOW_Down = False

__C.WITH_DIFF = False


__C.REMOVE_WARP_HOLE = True

__C.CONC_KITTI = False

__C.DROP_LAST = True

__C.TRUNCATED = False
__C.TRUNCATED_SIZE = 3

__C.CORRECT_ENTRO = False
__C.CORRECT_ENTRO_SOFT = False

__C.USE_SEQ_LOSS = False

__C.COST6_RATIO = 1.0
__C.COST5_RATIO = 1.0
__C.COST4_RATIO = 1.0
__C.COST3_RATIO = 1.0
__C.COST2_RATIO = 1.0

__C.SMOOTH_COST = False
__C.SMOOTH_LOSS = False
__C.SMOOTH_LOSS_WEIGHT = 0.1

__C.SMOOTH_SHARE = False
__C.SMOOTH_INIT_BY_ID = False
__C.FLOW_REG_BY_MAX = True
__C.SMOOTH_COST_ONLY_FLOW6 = False
__C.SMOOTH_COST_WITH_THREEMLP = False
__C.SCALE_BY_MASK = False
__C.DISTRIBUTED = False
__C.NO_SPATIAL = False
__C.NO_ERASE = False
__C.HALF_THINGS = False
__C.FIX_MATCHING = False
__C.MATCHING_USE_BN   = False
__C.MATCHING_USE_RELU = False
__C.USE_CORR = False
__C.TIMES_RATIO = False
__C.VALID_RANGE = [[8,8],
                    [32,32],
                    [64,64],
                    [128,128]]
__C.USE_VALID_RANGE = True
__C.USE_FUSION = False
__C.FULL_SINTEL = True
__C.DETACH_FLOW = True
__C.COST_COMP_METHOD = 'compute_cost_vcn_together'
__C.LOSS_TYPE = 'L1'
__C.MultiScale_W = [1.,0.5,0.25]
__C.CROP_SIZE = [256,256]
__C.FEATURE_NET = 'SPP'
__C.MATCH_INPUTC = [128,64,64]
__C.SEATCH_RANGE = [8,12,8]
__C.AUG_BY_ROT = False
__C.DILATED_LLL = False

__C.FAC = 1.0
__C.MD = [4,4,4,4,4]
__C.SEP_LEVEL = 1
__C.ADD_FEATURE = False
__C.CTF = False
__C.CTF_CONTEXT = False
__C.CTF_CONTEXT_ONLY_FLOW2 = False
__C.REFINE = 1
__C.REFINE_DETACH = False
__C.SHARE_MATCHING = False
__C.SHARE_MATCHING_MLP = False 
__C.COS_LR = False
__C.COS_TMAX = 20
__C.PSP_FEATURE = False
__C.NO_DECONV = False
__C.USE_RAW_FLOW2 = False
__C.SUP_RAW_FLOW = False
__C.SCALE_CONTEXT6 = 1.0
__C.SCALE_CONTEXT5 = 1.0
__C.SCALE_CONTEXT4 = 1.0
__C.SCALE_CONTEXT3 = 1.0
__C.SCALE_CONTEXT2 = 1.0






##########################################################################################


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)


def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if k not in b:
            raise KeyError('{} is not a valid config key'.format(k))
        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(type(b[k]), type(v), k))
        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print(('Error under config key: {}'.format(k)))
                raise
        else:
            b[k] = v


def cfg_from_list(cfg_list):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = __C
        for subkey in key_list[:-1]:
            assert subkey in d
            d = d[subkey]
        subkey = key_list[-1]
        assert subkey in d
        try:
            value = literal_eval(v)
        except:
            # handle the case when v is a string literal
            value = v
        assert type(value) == type(d[subkey]), \
            'type {} does not match original type {}'.format(type(value), type(d[subkey]))
        d[subkey] = value


def save_config_to_file(cfg, pre='cfg', logger=None):
    for key, val in cfg.items():
        if isinstance(cfg[key], edict):
            if logger is not None:
                logger.info('\n%s.%s = edict()' % (pre, key))
            else:
                print('\n%s.%s = edict()' % (pre, key))
            save_config_to_file(cfg[key], pre=pre + '.' + key, logger=logger)
            continue

        if logger is not None:
            logger.info('%s.%s: %s' % (pre, key, val))
        else:
            print('%s.%s: %s' % (pre, key, val))
