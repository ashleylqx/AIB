from .local_config import base_path
# mnist
X_dim = 784
H_dim = 1024
K_dim = 256
# n_class = 100 # CIFAR100
# n_class = 200  # CUB
n_class = 1000 # ILSVRC
A_dim = K_dim
num_sample = 1

# # configurations for quantum
# qt_num = 2
# vq_coef=0.2
# comit_coef=0.4
# lin_min = 1e-4
lin_min = 0.0
lin_max = 1.0

log_interval = 200
tb_log_interval = 200
coco_test_internal = 1

va_num = 20

MNIST_RESIZE = [28, 28]

gamma = 1e-2 # 1.0 #1e-3
beta = 1e-2 # 1e-3 for MNIST, 1e-2 for natural image

ATT = ['sigmoid', 'softmax', 'sig_sft', 'none', 'relu']
att_mode = ATT[3]

ModelInit = ['kaimingNormal', 'kaimingUniform', 'xavierNormal', 'xavierUniform'] # default

# The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class.
# There are 50000 training images and 10000 test images.
# This dataset is just like the CIFAR-10, except it has 100 classes containing 600 images each. There are 500 training images and 100 testing images per class.
# The 100 classes in the CIFAR-100 are grouped into 20 superclasses.
# Each image comes with a "fine" label (the class to which it belongs) and a "coarse" label (the superclass to which it belongs).
PATH_CIFAR10 = base_path + 'DataSets/CIFAR10'
PATH_CIFAR100 = base_path + 'DataSets/CIFAR100'

CIFAR_RESIZE = [32, 32]
CIFAR_sigma_bias = -0.5 #0.57


# train 5000, unlabeled 100000, test 8000
PATH_STL10 = base_path + 'DataSets/STL10'

STL_RESIZE = [96, 96]
STL_sigma_bias = -0.5 #0.57


# COCO
PATH_COCO = base_path + 'DataSets/MS_COCO/'
coco_classes = ['__background__',  # always index 0
                'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                'traffic_light', 'fire_hydrant', 'stop_sign', 'parking_meter', 'bench', 'bird', 'cat', 'dog', 'horse',
                'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                'suitcase', 'frisbee', 'skis', 'snowboard', 'sports_ball', 'kite', 'baseball_bat', 'baseball_glove',
                'skateboard', 'surfboard', 'tennis_racket', 'bottle', 'wine_glass', 'cup', 'fork', 'knife', 'spoon',
                'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot_dog', 'pizza', 'donut',
                'cake', 'chair', 'couch', 'potted_plant', 'bed', 'dining_table', 'toilet', 'tv', 'laptop', 'mouse',
                'remote', 'keyboard', 'cell_phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
                'clock', 'vase', 'scissors', 'teddy_bear', 'hair_drier', 'toothbrush']
# https://blog.csdn.net/u014106566/article/details/95195333
coco_id_name_map={1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
                   6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
                   11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
                   16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow',
                   22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack',
                   28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee',
                   35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat',
                   40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket',
                   44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon',
                   51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
                   56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
                   61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table',
                   70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard',
                   77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink',
                   82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors',
                   88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}

coco_num_classes = 91

COCO_RESIZE = [224, 224]

# CUB
cub_classes = 200
# PATH_CUB = base_path + 'DataSets/CUB/' # too old; images are different
PATH_CUB = base_path + 'DataSets/CUB_200_2011/' # new dataset

# ILSVRC
ilsvrc_classes = 1000
PATH_ILSVRC = base_path + 'DataSets/ILSVRC/'

WOL_RESIZE = [224, 224]

pre_train = False


# wildcat settings
kmax = 1
kmin = None
alpha = 0.7
num_maps = 4

# more path variables
PATH_EVENT = base_path + 'DataSets/Event_8/'
PATH_ACTION = base_path + 'DataSets/Action_40/'
PATH_SCENE = base_path + 'DataSets/Scene_67/'
PATH_OD = base_path + 'DataSets/ObjectDiscovery/'
PATH_CALTECH101 = base_path + 'DataSets/Caltech_101/'
PATH_CALTECH256 = base_path + 'DataSets/Caltech_256/'

feature_va_folder = 'features_va'
svm_model_va_folder = 'svm_model_va'

PATH_TINYIM = base_path + 'DataSets/tiny-imagenet-200/'

feature_freq_folder = 'freq_results'
feature_aug_folder = 'info_aug_results'
vis_folder = '../va_visualization'

