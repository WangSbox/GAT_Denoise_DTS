# -*- coding: utf-8 -*-
import torch
def get_edge_index():
    '''
    中间点往前后两点连接
    '''
    edge_index = []
    edge_index.append([1, 2])
    edge_index.append([1, 3])
    edge_index.append([2, 3])
    edge_index.append([2, 4])
    for i in range(3, 99):
        edge_index.append([i, i - 2])
        edge_index.append([i, i - 1])
        edge_index.append([i, i + 1])
        edge_index.append([i, i + 2])
    edge_index.append([99, 100])
    edge_index = torch.LongTensor(torch.transpose(torch.tensor(edge_index), 0, 1)) - 1  #从0计算
    return edge_index
def get_edge_index1():
    '''
    只连接其后两点
    '''
    edge_index = []
    for i in range(1, 99):
        edge_index.append([i, i + 1])
        edge_index.append([i, i + 2])
    edge_index.append([99, 100])
    edge_index = torch.LongTensor(torch.transpose(torch.tensor(edge_index), 0, 1)) - 1  #从0计算
    return edge_index
def get_edge_index2():
    '''
    只连接其后三点
    '''
    edge_index = []
    for i in range(1, 98):
        edge_index.append([i, i + 1])
        edge_index.append([i, i + 2])
        edge_index.append([i, i + 3])
    edge_index.append([98, 99])
    edge_index.append([98, 100])
    edge_index.append([99, 100])
    edge_index = torch.LongTensor(torch.transpose(torch.tensor(edge_index), 0, 1)) - 1  #从0计算
    return edge_index
# print(get_edge_index2())
