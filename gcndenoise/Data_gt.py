# -*- coding: utf-8 -*-
from __future__ import print_function
import os
# import time
import h5py
from tqdm import tqdm
import numpy as np
# import argparse
# import torch

def get_data(args, dtset, dtfolder, dname):
    fea_num = [0, 1, 2, 3, 4]
    if not args.ab:
        fea_num.remove(3)
    if not args.dist:
        fea_num.remove(4)
    traindata_tem = np.empty([0, 100, len(fea_num)])
    traindata_label = np.empty([0, 100])

    if dname == 'traindata_temper':   
        flist = []
        for ite in range(args.trn):
            data_reference = os.listdir(os.path.join(dtfolder + str(ite+1)))  #数据集以及数据及的实体名称
            if len(data_reference) <= 0:
                continue
            # print(data_reference)
            for file_paths in data_reference:  #获取从data——reference里导入的文件的子文件路径
                # print(file_paths)
                if int(file_paths.split('.')[0].split('_')[-1]) < args.lc:  #不满足条件的直接不移动过来 小于想要训练位置的数据均不移动
                    # print('ok')
                    flist.append(os.path.join(dtfolder + str(ite+1), file_paths))
        # print(flist)
        print(len(flist))
        for _, file in enumerate(tqdm(flist)):
            # print(file)
            fdata = h5py.File(file)[dname]
            traindata_tem = np.concatenate((traindata_tem, fdata[fea_num, :, :].transpose(2, 1, 0)), axis=0)
            traindata_label = np.concatenate((traindata_label, fdata[5, :, :].transpose(1,0)), axis=0)
            del fdata
        # time.sleep(100)
    elif dname == 'testdata_temper':
        flist = []
        data_reference = os.listdir(os.path.join(dtfolder))  #数据集以及数据及的实体名称
        for file_paths in data_reference:  #获取从data——reference里导入的文件的子文件路径
            # print(file_paths)
            if len(flist) < args.tesn:
                flist.append(os.path.join(dtfolder,file_paths))
        print(len(flist))        
        for _, file in enumerate(tqdm(flist)):
            # num = int(flist[i].split('/')[-1].split('.')[0].split('_')[-1])
            # if int(flist[i].split('/')[-1].split('.')[0][-1]) < located:
            fdata = h5py.File(os.path.join(file))[dname]
            for j in range(args.lc):
                traindata_tem = np.concatenate((traindata_tem,
                                            fdata[fea_num, j * 100:(j + 1) * 100, :].transpose(2, 1, 0)), axis=0)
                traindata_label = np.concatenate((traindata_label,
                                            fdata[5, j * 100:(j + 1) * 100, :].transpose(1, 0)), axis=0)
            del fdata
    else:
        raise('check your configurations')
    if args.ab:
        if args.round:
            traindata_tem[:, :, 3] = np.round_(traindata_tem[:, :, 3])  #将状态舍入
            print('将异常状态舍入')
        else:
            print('未将异常状态舍入')
    print(traindata_tem.shape, traindata_label.shape)
    return traindata_tem, traindata_label
# get_data('kpi_train','train','ab',True,3)

# parser = argparse.ArgumentParser("Extracted and Detected abnormal of data")
# parser.add_argument('--epoch', type=int, default=50, help='epoch')
# parser.add_argument('--bs', type=int, default=2048, help='batch size')
# parser.add_argument('--nw', type=int, default=8, help='num workers')
# parser.add_argument('--lr', type=float, default=1e-2, help='learn_ratio')
# parser.add_argument('--lc', type=int, default=2, help='train location')
# parser.add_argument('--trn', type=int, default=2, help='number of train dataset_entity')
# parser.add_argument('--tesn', type=int, default=5, help='number of test dataset')
# parser.add_argument('--tgac', type=float, default=0.9, help='target accuracy')
# parser.add_argument('--ts', type=float, default=1e-2, help='Threshold for measuring the new optimum of lr.scheduler')
# parser.add_argument('--ab', type=int, default=1, help='Include abnormal status value')  # 1/0/true/false 皆可
# parser.add_argument('--round', type=int, default=0, help='Round off exception status value')
# parser.add_argument('--dist', type=int, default=1, help='Include distance features')
# parser.add_argument('--th', type=float, default=0.05, help='threshold')
# arg = parser.parse_args()
# print(arg)

# if torch.cuda.is_available():
#     print(torch.version.cuda)
# else:
#     print('no cuda')
# path = r'D:/model_test/temper/数据/TemperatureDenoiseDataSetWithCompensateScaler/train_interval_'
# print('DataSet1_temper5t1_location_10.mat'.split('.')[0].split('_')[-1])
# traindata_tem, traindata_label = get_data(arg, 'data', path, 'traindata_temper')  #带补偿的数据集 包含异常检测 的 降噪数据集
# # fff = h5py.File('D:/model_test/temper/数据/TemperatureDenoiseDataSetWithCompensateScaler/train_interval_1/DataSet1_temper5t1_location_0.mat','r')['traindata_temper']