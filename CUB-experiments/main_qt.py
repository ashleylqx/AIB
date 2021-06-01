import numpy as np
import json
import torch
import argparse
from utils import str2bool
from solver_qt import SolverVA
from config import *


# use parse_arguments
def main(args):
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    np.set_printoptions(precision=4)
    torch.set_printoptions(precision=4)

    print()
    print('[ARGUMENTS]')
    print(args)
    print()

    net = SolverVA(args)

    if args.mode == 'train':
        if args.anneal:
            net.train_anneal()
            print('Training VA anneal.')
        else:
            net.train()
            print('Training VA.')

    elif args.mode == 'test':
        net.test(save_ckpt=False)
        net.test_ema(save_ckpt=False)

    elif args.mode == 'saveatt':
        net.save_attention()

    else : return 0


def parse_arguments():
    parser = argparse.ArgumentParser(description='TOY VA VIB')
    parser.add_argument('--epoch', default=200, type=int, help='epoch size')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')

    parser.add_argument('--beta', default=beta, type=float, help='beta')
    parser.add_argument('--K', default=256, type=int, help='dimension of encoding Z')
    parser.add_argument('--att_K', default=256, type=int, help='dimension of attention A')
    parser.add_argument('--num_sample', default=num_sample, type=int, help='the number of samples')
    parser.add_argument('--seed', default=1, type=int, help='random seed')
    parser.add_argument('--num_avg', default=12, type=int, help='the number of samples when\
                perform multi-shot prediction')
    parser.add_argument('--train_batch', default=100, type=int, help='train batch size')
    parser.add_argument('--test_batch', default=100, type=int, help='test batch size')
    parser.add_argument('--env_name', default='main', type=str, help='visdom env name')
    parser.add_argument('--dataset', default='MNIST', type=str, help='dataset name')
    parser.add_argument('--dset_dir', default='datasets', type=str, help='dataset directory path')
    parser.add_argument('--summary_dir', default='summary', type=str, help='summary directory path')
    parser.add_argument('--att_dir', default='att_maps', type=str, help='att_mask directory path')
    parser.add_argument('--ckpt_dir', default='VGG_WRN_224', type=str, help='checkpoint directory path')
    parser.add_argument('--load_ckpt', default='best_acc.tar', type=str, help='checkpoint name')
    parser.add_argument('--cuda', default=True, type=str2bool, help='enable cuda')
    parser.add_argument('--mode', default='train', type=str, help='train or test')
    parser.add_argument('--tensorboard', default=False, type=str2bool, help='enable tensorboard')
    parser.add_argument('--return_att', default=False, type=str2bool, help='enable return att_mask')
    parser.add_argument('--min_zax', default=False, type=str2bool, help='enable return att_mask')
    parser.add_argument('--dset_ratio', default=1.0, type=float, help='data ratio for training')
    parser.add_argument('--anneal', default=False, type=str2bool, help='enable anneal training')
    parser.add_argument('--backbone', default='vgg16', type=str, help='model backbone (vgg16, wrn_50_2)')
    parser.add_argument('--qt_num', default=2, type=int, help='number of quantum')
    parser.add_argument('--vq_coef', default=0.2, type=float, help='weight for vq_loss')
    parser.add_argument('--comit_coef', default=0.4, type=float, help='weight for commit_loss')
    parser.add_argument('--rd_init', default=False, type=str2bool, help='randomly init quantum layer or not.')
    parser.add_argument('--decay', default=None, type=float, help='decay quantum layer if not set as None')
    parser.add_argument('--qt_trainable', default=False, type=str2bool, help='quantum is trainable or not.')
    parser.add_argument("--tencrop", type=str2bool, default=False, help='set True for classification')

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_arguments()
    main(args)

