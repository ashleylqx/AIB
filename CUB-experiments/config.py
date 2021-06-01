from local_config import base_path
# mnist
X_dim = 784
H_dim = 1024
K_dim = 256
# n_class = 10 # CF10, STL10
# n_class = 100 # CF100
n_class = 200 # CUB
# n_class = 1000 # ILSVRC
# n_class = 91  # COCO
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

# CUB
cub_classes = 200
PATH_CUB = base_path + 'dataset/CUB_200_2011/' # new dataset


WOL_RESIZE = [224, 224]

# pre_train = False
pre_train = True

# wildcat settings
kmax = 1
kmin = None
alpha = 0.7
num_maps = 4