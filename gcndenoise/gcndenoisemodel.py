# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv, SAGEConv, TransformerConv, DNAConv, ResGatedGraphConv
from torch_geometric.nn import GCN, GAT, GraphSAGE, PNA, BatchNorm, GraphNorm


class sage(torch.nn.Module):  #将原来内嵌的层 分开实现
    def __init__(self, channels):
        super().__init__()

        self.sg = GraphSAGE(channels, 64, 2)
        self.sg1 = GraphSAGE(64, 256, 2)
        self.sg2 = GraphSAGE(256, 512, 2)
        self.nl = nn.Linear(512*100, 100)
    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.leaky_relu(self.sg(x, edge_index), 0.1)
        x = F.leaky_relu(self.sg1(x, edge_index), 0.1)
        x = F.leaky_relu(self.sg2(x, edge_index),0.1)
        x = x.view(-1,512*100)
        x = F.relu(self.nl(x))
        return x

class gcn(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.batch1 = BatchNorm(64)
        self.batch2 = BatchNorm(256)

        self.gc = GCN(channels, 64, 2)
        self.gc1 = GCN(64, 256, 2)
        self.gc2 = GCN(256, 512, 2, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.leaky_relu(self.gc(x, edge_index), 0.1)
        x = F.leaky_relu(self.gc1(x, edge_index), 0.1)
        x = self.gc2(x, edge_index)
        return x


class gat(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.batch1 = BatchNorm(64)
        self.batch2 = BatchNorm(256)

        self.gt = GAT(channels, 64, 2, heads=8)
        self.gt1 = GAT(64, 256, 2, heads=8)
        self.gt2 = GAT(256, 512, 2, 1, heads=8)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.leaky_relu(self.gt(x, edge_index), 0.1)
        x = F.leaky_relu(self.gt1(x, edge_index), 0.1)
        x = self.gt2(x, edge_index)
        return x
class gat_1(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.batch1 = BatchNorm(64)
        self.batch2 = BatchNorm(256)

        self.gt = GAT(channels, 64, 2, heads=8, dropout=0.1)
        self.gt1 = GAT(64, 256, 2, heads=8, dropout=0.1)
        self.gt2 = GAT(256, 512, 2, 1, heads=8, dropout=0.1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.leaky_relu(self.gt(x, edge_index), 0.1)
        x = F.leaky_relu(self.gt1(x, edge_index), 0.1)
        x = self.gt2(x, edge_index)
        return x
class merge(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.batch1 = BatchNorm(64)
        self.batch2 = BatchNorm(256)

        self.gt = GAT(channels, 64, 2, heads=8, dropout=0.1)
        self.gt1 = GAT(64, 256, 2, heads=8, dropout=0.1)
        self.gt2 = GAT(256, 512, 2, heads=8, dropout=0.1)
        self.gt3 = GAT(512, 1024, 1, 1, heads=8, dropout=0.1)


    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.leaky_relu(self.gt(x, edge_index) , 0.1)
        x = F.leaky_relu(self.gt1(x, edge_index), 0.1)
        x = F.leaky_relu(self.gt2(x, edge_index), 0.1)
        x = self.gt3(x, edge_index)
        return x