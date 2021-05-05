# Attentive Information Bottleneck

## Overview
Pytorch code for "Information Bottleneck Approach to Spatial Attention Learning (IJCAI2021)".

What's in this repo so far:
 * Code for CIFAR-10/-100 experiments (VGG backbone) ([this folder](VGG-backbone-32))
 * Code for CIFAR-10/-100 experiments (WRN backbone) ([this folder](WRN-backbone-32))

Coming:
 * Code for CUB experiments (VGG and WRN backbones)



## Reference
[1] [Attention Transfer](https://github.com/szagoruyko/attention-transfer)

[2] [Attention Branch Network](https://github.com/machine-perception-robotics-group/attention_branch_network)

[3] [LearnToPayAttention](https://github.com/SaoYan/LearnToPayAttention)


## Requirements

Create an anaconda environment:

```commandline
$ conda env create -f environment.yaml
```

To run the code:

```commandline
$ source activate torch36
$ <run_python_command> # see the examples in sub folders.
```


## Citation

If you find this repository is useful, please cite the following reference.
```
@inproceedings{lai2021information,
    title = {Information Bottleneck Approach to Spatial Attention Learning},
    author = {Lai, Qiuxia and Li, Yu and Zeng, Ailing and Liu, Minhao and Sun, Hanqiu and Xu, Qiang},
    booktitle = {International Joint Conference on Artificial Intelligence},
    year = {2021}
}
```
