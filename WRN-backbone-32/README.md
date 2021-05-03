# Code for CIFAR-10/-100 experiments (WRN backbone)

PyTorch code for "Information Bottleneck Approach to Spatial Attention Learning (IJCAI2021)".


## Pre-trained Models

Google Drive: Coming soon.

Baidu Pan: Coming soon.


## Experiments

To train with multiple GPUs, use parameters such as ` --gpu_id 0,1,2,3`.


### CIFAR-10

Train `vgg_va` model:

```
python cifar_va.py -a wrn_va --checkpoint WRN_32/cifar10/WRN-VA-28-10-drop_k512
```

The examplar attention maps can be found in `att_maps/cifar10/WRN-VA-28-10-drop_k512`.
Or use the following command to generate examplar attention maps from trained model:

```
python cifar_va.py --phase save_att_va -a wrn_va --resume WRN_32/cifar10/WRN-VA-28-10-drop_k512
```


Train `vgg_va_qt` model:

```
python cifar_va.py -a wrn_va_qt --checkpoint WRN_32/cifar10/WRN-VA-28-10-drop_qt50_k512 --qt_num 50
```

The examplar attention maps can be found in `att_maps/cifar100/WRN-VA-28-10-drop_qt50_k512`.
Or use the following command to generate examplar attention maps from trained model:

```
python cifar_va.py --phase save_att_va_qt -a wrn_va_qt --resume WRN_32/cifar10/WRN-VA-28-10-drop_qt50_k512 --qt_num 50
```

### CIFAR-100

Train `vgg_va` model:

```
python cifar_va.py -a wrn_va --dataset cifar100 --checkpoint WRN_32/cifar100/WRN-VA-28-10-drop_k512
python cifar_va.py -a wrn_va --dataset cifar100 --checkpoint WRN_32/tmp --gpu_id 2
```

The examplar attention maps can be found in `att_maps/cifar100/WRN-VA-28-10-drop_k512`.
Or use the following command to generate examplar attention maps from trained model:

```
python cifar_va.py --phase save_att_va -a wrn_va --dataset cifar100 --resume WRN_32/cifar100/WRN-VA-28-10-drop_k512
python cifar_va.py --phase save_att_va -a wrn_va --dataset cifar100 --resume WRN_32/tmp --gpu_id 2
```


Train `vgg_va_qt` model:

```
python cifar_va.py -a wrn_va_qt --dataset cifar100 --checkpoint WRN_32/cifar100/WRN-VA-28-10-drop_qt20_k512 --qt_num 20
python cifar_va.py -a wrn_va_qt --dataset cifar100 --checkpoint WRN_32/tmp --qt_num 20 --gpu_id 3
```

The examplar attention maps can be found in `att_maps/cifar100/WRN-VA-28-10-drop_qt20_k512`.
Or use the following command to generate examplar attention maps from trained model:

```
python cifar_va.py --phase save_att_va_qt -a wrn_va_qt --dataset cifar100 --resume WRN_32/cifar100/WRN-VA-28-10-drop_qt20_k512 --qt_num 20
python cifar_va.py --phase save_att_va_qt -a wrn_va_qt --dataset cifar100 --resume WRN_32/tmp --qt_num 20 --gpu_id 3
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
