#!/usr/bin/env python3
'''
/************************/
/*        nn.py         */
/*    Version 1.0       */
/*     2023/03/08       */
/************************/
'''

import h5py
import numpy as np
from os import makedirs
import sys

CONFDIR = "./build/config"
H5FILE2 = "nn2.hd5"
H5FILE3 = "nn3.hd5"


def create_folder():
    makedirs(CONFDIR, exist_ok=True)


def create_config2():
    with h5py.File(CONFDIR + '/' + H5FILE2, 'w') as f:
        NN = 'nn2'
        s = np.int8(list(b"network_sgd"))
        f.create_dataset(NN + '/nname', data=s)
        s = np.int8(list(b"./build/data_network2.hd5"))
        f.create_dataset(NN + '/data_file', data=s)
        # first set of parameters
        CSET1 = 'cfg_1'
        DSET1 = NN + '/' + CSET1
        v = np.int64([784, 30, 10])
        f.create_dataset(DSET1 + '/size', data=v)
        i = np.int64([5])
        f.create_dataset(DSET1 + '/nEpochs', data=i)
        i = np.int64([10])
        f.create_dataset(DSET1 + '/miniBatchSize', data=i)
        d = np.double([3.0])
        f.create_dataset(DSET1 + '/eta', data=d)

        CSET2 = 'cfg_2'
        v = np.int64([784, 64, 16, 10])
        DSET2 = NN + '/' + CSET2
        f.create_dataset(DSET2 + '/size', data=v)
        i = np.int64([5])
        f.create_dataset(DSET2 + '/nEpochs', data=i)
        i = np.int64([10])
        f.create_dataset(DSET2 + '/miniBatchSize', data=i)
        d = np.double([3.0])
        f.create_dataset(DSET2 + '/eta', data=d)

        # set the current set
        s = np.int8(list(bytes(CSET1, 'ascii')))
        f.create_dataset(NN + '/current_set', data=s)


def create_config3():
    with h5py.File(CONFDIR + '/' + H5FILE3, 'w') as f:
        NN = 'nn3'
        s = np.int8(list(b"network_ga"))
        f.create_dataset(NN + '/nname', data=s)
        s = np.int8(list(b"./build/data_network3.hd5"))
        f.create_dataset(NN + '/data_file', data=s)
        # first set of parameters
        CSET1 = 'cfg_1'
        DSET1 = NN + '/' + CSET1
        v = np.int64([784, 30, 10])
        f.create_dataset(DSET1 + '/size', data=v)
        i = np.int64([5000])
        f.create_dataset(DSET1 + '/nGenerations', data=i)
        i = np.int64([200])
        f.create_dataset(DSET1 + '/BatchSize', data=i)

        # set the current set
        s = np.int8(list(bytes(CSET1, 'ascii')))
        f.create_dataset(NN + '/current_set', data=s)


def main():
    create_folder()
    create_config2()
    create_config3()


if __name__ == '__main__':
    if sys.version_info[0] < 3:
        raise 'Must be using Python 3'
    main()
