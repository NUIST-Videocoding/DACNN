import os
import torch
import argparse
import random
import numpy as np

from pretrain_model.FRsolver import Solver

torch.cuda.set_device(0)

def main(config):
    folder_path = '../data/IQAbigdatabase/'
    img_num = list(range(0, 10340))

    print('Training and testing on LSIQA dataset for 1 rounds...')
    random.shuffle(img_num)

    np.load('LSQA.npy')

    train_index = img_num[0:int(round(0.8 * len(img_num)))]
    test_index = img_num[int(round(0.8 * len(img_num))):len(img_num)]

    solver = Solver(config, folder_path, train_index, test_index)
    srcc_all, plcc_all = solver.train()





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', dest='lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=0, help='Weight decay')
    parser.add_argument('--lr_ratio', dest='lr_ratio', type=int, default=10,
                        help='Learning rate ratio for hyper network')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', dest='epochs', type=int, default=3000, help='Epochs for training')
    parser.add_argument('--patch_size', dest='patch_size', type=int, default=224,
                        help='Crop size for training & testing image patches')

    config = parser.parse_args()
    main(config)






