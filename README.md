## Deep Two-View Structure-from-Motion Revisited

This repository provides the code for our CVPR 2021 paper [Deep Two-View Structure-from-Motion Revisited](https://arxiv.org/abs/2104.00556).


We have provided the functions for training, validating, and visualization.

Note: some config flags are designed for ablation study, and we have a plan to re-org the codes later. Please feel free to submit issues if you feel confused about some parts.



## Requirements

```
Python = 3.6.x
Pytorch >= 1.6.0
CUDA >= 10.1
```

and the others could be installed by

```
pip install -r requirements.txt
```

Pytorch from 1.1.0 to 1.6.0 should also work well, but it will disenable mixed precision training, and we have not tested it.

To use the RANSAC five-point algorithm, you also need to 

```
cd RANSAC_FiveP

python setup.py install --user
```

The CUDA extension would be installed as 'essential_matrix'. Tested under Ubuntu and CUDA 10.1.


## Models

Pretrained models are provided [here](https://drive.google.com/drive/folders/1g0uoNrldySyWnkVfQ53etqcNlhzJrAHx?usp=sharing).

## KITTI Depth

To reproduce our results, please first download the KITTI dataset [RAW data](http://www.cvlibs.net/datasets/kitti/raw_data.php) and 14GB [official depth maps](http://www.cvlibs.net/datasets/kitti/eval_depth.php). You should also download the [split files](https://drive.google.com/drive/folders/1g0uoNrldySyWnkVfQ53etqcNlhzJrAHx?usp=sharing) provided by us, and unzip them into the root of the KITTI raw data. Then, modify the gt_depth_dir (KITTI_loader.py, L278) to the address of KITTI official depth maps.

For training, 
```
python main.py -b 32 --lr 0.0005 --nlabel 128 --fix_flownet \
--data PATH/TO/YOUR/KITTI/DATASET --cfg cfgs/kitti.yml \
--pretrained-depth depth_init.pth.tar --pretrained-flow flow_init.pth.tar
```

For evaluation, 
```
python main.py -v -b 1 -p 1 --nlabel 128 \
--data PATH/TO/YOUR/KITTI/DATASET --cfg cfgs/kitti.yml \
--pretrained kitti.pth.tar"
```

The default evaluation split is Eigen. If you would like to use the Eigen SfM split, please set cfg.EIGEN_SFM = True and cfg.KITTI_697 = False.

## KITTI Pose

For fair comparison, we use a KITTI odometry evaluation toolbox as provided [here](https://github.com/Huangying-Zhan/kitti-odom-eval). Please generate poses by sequence, and evaluate the results correspondingly.

## Acknowledgment:

Thanks [Shihao Jiang](https://zacjiang.github.io/) and [Dylan Campbell](https://sites.google.com/view/djcampbell/) for sharing the implementation of the GPU-accelerated RANSAC Five-point algorithm. We really appreciate the valuable feedback from our area chairs and reviewers. We would like to thank [Charles Loop](https://scholar.google.com/citations?user=qqSucBkAAAAJ&hl=en) for helpful discussions and Ke Chen for providing field test images from NVIDIA AV cars.



## BibTex:

```
@article{wang2021deep,
  title={Deep Two-View Structure-from-Motion Revisited},
  author={Wang, Jianyuan and Zhong, Yiran and Dai, Yuchao and Birchfield, Stan and Zhang, Kaihao and Smolyanskiy, Nikolai and Li, Hongdong},
  journal={CVPR},
  year={2021}
}
```