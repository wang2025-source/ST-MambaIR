import cv2
import numpy as np
import os
import sys
from multiprocessing import Pool
from os import path as osp
from tqdm import tqdm

from basicsr.utils import scandir
from create_lmdb import create_lmdb_for_custom_dataset

def main():
    # src folder
    gt_folder = '/tmp/gt_src'
    lq_folder = '/tmp/lq_src'

    # train folder
    gt_lmdb_train_path = '/tmp/gt_train.lmdb'
    lq_lmdb_train_path = '/tmp/lq_train.lmdb'

    # validation folder
    gt_lmdb_val_path = '/tmp/gt_val.lmdb'
    lq_lmdb_val_path = '/tmp/lq_val.lmdb'

    create_lmdb_for_custom_dataset(
        gt_folder, lq_folder,
        gt_lmdb_train_path, lq_lmdb_train_path,
        gt_lmdb_val_path, lq_lmdb_val_path)

if __name__ == '__main__':
    main()
    # ... make images to lmdb
