# -*- coding: utf-8 -*-
from torch_geometric.data.dataset import Dataset
from torch_geometric.data import Data
class TempDataset(Dataset):
    def __init__(self, tem_data, label_data, edge):
        self.edge_index = edge
        self.tem = tem_data
        self.label = label_data
    def __len__(self):
        return self.tem.size(0)
    def __getitem__(self, i):
        data = Data(x=self.tem[i], edge_index=self.edge_index, y=self.label[i].unsqueeze(1))
        return  data
        