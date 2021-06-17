## Deep Two-View Structure-from-Motion Revisited

This repository provides the code for our CVPR 2021 paper [Deep Two-View Structure-from-Motion Revisited](https://arxiv.org/abs/2104.00556).


We have provided the functions for training, validating, and visualization. _More is coming_.

Note: some config flags are designed for ablation study, and we have a plan to re-org the codes later. Please feel free to submit issues if you feel confused about some parts.


## Requirements


```
Python >= 3.6.0
Pytorch >= 1.6.0
CUDA >= 10.1
```

and the others could be installed by

```
pip install -r requirements.txt
```

Pytorch from 1.1.0 to 1.6.0 should also work well, but it will disenable mixed precision training, and we have not tested it.



```
@article{wang2021deep,
  title={Deep Two-View Structure-from-Motion Revisited},
  author={Wang, Jianyuan and Zhong, Yiran and Dai, Yuchao and Birchfield, Stan and Zhang, Kaihao and Smolyanskiy, Nikolai and Li, Hongdong},
  journal={CVPR},
  year={2021}
}
```




