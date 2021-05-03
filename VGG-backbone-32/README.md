# Code for CIFAR-10/-100 experiments (VGG backbone)

PyTorch code for "Information Bottleneck Approach to Spatial Attention Learning (IJCAI2021)".


## Pre-trained Models

Google Drive: Coming soon.

Baidu Pan: Coming soon.


## Experiments

To train with multiple GPUs, use parameters such as ` --gpu_id 0,1,2,3`.


### CIFAR-10

Train `vgg_va` model:

```
python cifar_va.py --env_name vgg_va_cifar10 --lr 0.01
```

Save examplar attention maps for `vgg_va` model:

```
python cifar_va.py --phase save_att --env_name vgg_va_cifar10 --lr 0.01
```

Train `vgg_va_qt` model:

```
python cifar_va.py --phase train_qt --env_name vgg_va_qt_cifar10 --lr 0.001 
python cifar_va.py --phase train_qt --env_name tmp --lr 0.001 --gpu_id 1
```

Save examplar attention maps for `vgg_va_qt` model:

```
python cifar_va.py --phase save_att_qt --env_name vgg_va_qt_cifar10 --lr 0.001
python cifar_va.py --phase save_att_qt --env_name tmp --lr 0.001 --gpu_id 1
```



### CIFAR-100

Train `vgg_va` model:

```
python cifar_va.py --env_name vgg_va_cifar100 --dataset CIFAR100 --lr 0.01
```

Save examplar attention maps for `vgg_va` model:

```
python cifar_va.py --phase save_att --env_name vgg_va_cifar100 --dataset CIFAR100 --lr 0.01
```

Train `vgg_va_qt` model:

```
python cifar_va.py --phase train_qt --env_name vgg_va_qt_cifar100 --dataset CIFAR100 --lr 0.001 --qt_num 50 
```

Save examplar attention maps for `vgg_va_qt` model:

```
python cifar_va.py --phase save_att_qt --env_name vgg_va_qt_cifar100 --dataset CIFAR100 --lr 0.001 --qt_num 50 
```


## Citation

If you find this repository is useful, please cite the following reference.
```
@inproceedings{lai2021information,
    title = {Information Bottleneck Approach to Spatial Attention Learning},
    author = {Lai, Qiuxia and Li, Yu and Zeng, Ailing and Liu, Minhao and Xu, Qiang and Sun, Hanqiu},
    booktitle = {International Joint Conference on Artificial Intelligence},
    year = {2021}
}
```