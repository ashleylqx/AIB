# Code for CUB experiments (VGG and WRN backbone)

PyTorch code for "Information Bottleneck Approach to Spatial Attention Learning (IJCAI2021)".


## Pre-trained Models

Google Drive: Coming soon.

Baidu Pan: https://pan.baidu.com/s/1GAIz4tVz619GeLuVxb3hXQ, code: `zw1s`


## Experiments

To train with multiple GPUs, use `CUDA_VISIBLE_DEVICES=0`. No need to multiply the batch size with #GPUs.
We pre-process the images following previous practice. See the paper for more details.

Please download the CUB dataset from the [official link](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html). The dataset directory is as follows:

```
dataset
|---CUB_200_2011
    |---images
        |---001.Black_footed_Albatross
            |---Black_Footed_Albatross_0001_796111.jpg
            ...
            |---Black_Footed_Albatross_0090_796077.jpg
        |---002.Laysan_Albatross
        ...
        |---200.Common_Yellowthroat

    |---lists
        |---train_list.txt
        |---test_list.txt
        ...

```
where the `lists` is provided here.

### VGG backbone

Train `vgg_aib` model:

```commandline
CUDA_VISIBLE_DEVICES=0 python main.py --mode train --dataset CUB_crop --env_name vgg_va_CUB_crop --K 256 --train_batch 16 --test_batch 4 # ori dataloader
```

Train `vgg_aib` model:

```commandline
CUDA_VISIBLE_DEVICES=1 python main.py --mode test --dataset CUB_crop --env_name vgg_va_CUB_crop --K 256 --train_batch 16 --test_batch 4 # ori dataloader

```

Save examplar attention maps for `vgg_aib` model:

```commandline
CUDA_VISIBLE_DEVICES=0 python main.py --mode saveatt --dataset CUB_crop --env_name vgg_va_CUB_crop --K 256 --train_batch 16 --test_batch 4 # ori dataloader

```

Train `vgg_aib_qt` model:

```commandline
CUDA_VISIBLE_DEVICES=3 python main_qt.py --mode train --dataset CUB_crop --env_name vgg_va_qt_CUB_crop --qt_num 20 --vq_coef 0.4 --comit_coef 0.1 --K 256 --train_batch 16 --test_batch 4 # ori dataloader
```

Test `vgg_aib_qt` model:

```commandline
CUDA_VISIBLE_DEVICES=3 python main_qt.py --mode test --dataset CUB_crop --env_name vgg_va_qt_CUB_crop --qt_num 20 --vq_coef 0.4 --comit_coef 0.1 --K 256 --train_batch 16 --test_batch 4 # ori dataloader

```

Save examplar attention maps for `vgg_aib_qt` model:

```commandline
CUDA_VISIBLE_DEVICES=3 python main_qt.py --mode saveatt --dataset CUB_crop --env_name vgg_va_qt_CUB_crop --qt_num 20 --vq_coef 0.4 --comit_coef 0.1 --K 256 --train_batch 16 --test_batch 4 # ori dataloader

```



### WRN backbone

Train `wrn_aib` model:

```commandline
CUDA_VISIBLE_DEVICES=0 python main.py --mode train --dataset CUB_crop --env_name wrn_va_CUB_crop --backbone wrn_50_2 --K 256 --train_batch 16 --test_batch 4 # ori dataloader
```

Test `wrn_aib` model:

```commandline
CUDA_VISIBLE_DEVICES=2 python main.py --mode test --dataset CUB_crop --env_name wrn_va_CUB_crop --backbone wrn_50_2 --K 256 --train_batch 16 --test_batch 4 # ori dataloader

```

Save examplar attention maps for `vgg_aib` model:

```commandline
CUDA_VISIBLE_DEVICES=0 python main.py --mode saveatt --dataset CUB_crop --env_name wrn_va_CUB_crop --backbone wrn_50_2 --K 256 --train_batch 16 --test_batch 4 # ori dataloader

```

Train `wrn_aib_qt` model:

```commandline
CUDA_VISIBLE_DEVICES=0 python main_qt.py --mode train --dataset CUB_crop --env_name wrn_va_qt_CUB_crop --backbone wrn_50_2 --qt_num 20 --vq_coef 0.4 --comit_coef 0.1 --K 256 --train_batch 16 --test_batch 4 # ori dataloader
```

Test `wrn_aib_qt` model:

```commandline
CUDA_VISIBLE_DEVICES=1 python main_qt.py --mode test --dataset CUB_crop --env_name wrn_va_qt_CUB_crop --backbone wrn_50_2 --qt_num 20 --vq_coef 0.4 --comit_coef 0.1 --K 256 --train_batch 16 --test_batch 4 # ori dataloader
```

Save examplar attention maps for `vgg_aib_qt` model:

```commandline
CUDA_VISIBLE_DEVICES=0 python main_qt.py --mode saveatt --dataset CUB_crop --env_name wrn_va_qt_CUB_crop --backbone wrn_50_2 --qt_num 20 --vq_coef 0.4 --comit_coef 0.1 --K 256 --train_batch 16 --test_batch 4 # ori dataloader
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
