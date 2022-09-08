# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import torch
import gcndenoisemodeltrain as gdmt
import Data_gt

def main(args):
    print('准备加载数据！')
    path = r'./datasets/train/train_interval_'
    traindata_tem, traindata_label = Data_gt.get_data(args, 'data', path, 'traindata_temper')  #带补偿的数据集 包含异常检测 的 降噪数据集
    path = r'./dataset/test'
    testdata_tem, testdata_label = Data_gt.get_data(args, 'data', path, 'testdata_temper')
    print('数据加载完成！')

    traindata_tem = torch.from_numpy(traindata_tem).float()
    traindata_label = torch.from_numpy(traindata_label).float()
    testdata_tem = torch.from_numpy(testdata_tem).float()
    testdata_label = torch.from_numpy(testdata_label).float()

    print(traindata_tem.size(), traindata_label.size())
    print(testdata_tem.size(), testdata_label.size())

    print('开始训练模型！')
    max_acc = gdmt.model_denoise(args, traindata_tem, traindata_label, testdata_tem, testdata_label)
    print('模型训练完毕！获得最大训练正确率:{}'.format(max_acc))
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Extracted and Detected abnormal of data")
    parser.add_argument('--epoch', type=int, default=50, help='epoch')
    parser.add_argument('--bs', type=int, default=2048, help='batch size')
    parser.add_argument('--nw', type=int, default=8, help='num workers')
    parser.add_argument('--lr', type=float, default=1e-2, help='learn_ratio')
    parser.add_argument('--lc', type=int, default=2, help='train location')
    parser.add_argument('--trn', type=int, default=2, help='number of train dataset_entity')
    parser.add_argument('--tesn', type=int, default=5, help='number of test dataset')
    parser.add_argument('--tgac', type=float, default=0.9, help='target accuracy')
    parser.add_argument('--ts', type=float, default=1e-2, help='Threshold for measuring the new optimum of lr.scheduler')
    parser.add_argument('--ab', type=int, default=1, help='Include abnormal status value')  # 1/0/true/false 皆可
    parser.add_argument('--round', type=int, default=0, help='Round off exception status value')
    parser.add_argument('--dist', type=int, default=1, help='Include distance features')
    parser.add_argument('--th', type=float, default=0.05, help='threshold')
    arg = parser.parse_args()
    print(arg)

    if torch.cuda.is_available():
        print(torch.version.cuda)
    else:
        print('no cuda')

    main(arg)
