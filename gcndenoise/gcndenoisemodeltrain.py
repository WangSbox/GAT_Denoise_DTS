# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import random as rand
import time
# import moxing as mox
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm

# from naie.context import Context
import pandas as pd
import numpy as np

from torch_geometric.loader import DataLoader
from gcndataset import TempDataset
from gcndenoisemodel import *
from get_edge import *
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def train_model(model, train_data, optimizer, device, t, trainlosslist):
    all_loss, loss = 0.0, 0.0
    # optimizer.zero_grad()
    # t.set_postfix(sum=len(train_data))
    with torch.set_grad_enabled(True):
        model.train()
        for trainframe, dataset1 in enumerate(train_data):
            # print(trainframe)
            optimizer.zero_grad()
            out = model(dataset1.to(device)).squeeze()
            loss = F.smooth_l1_loss(out, dataset1.y.squeeze()*100)
            loss.backward()
            los = loss.item()
            all_loss += los
            # if trainframe % 40 == 0:
            t.set_postfix(loss=los)
            t.set_description('train iter %i/%i' %( trainframe+1, len(train_data)))
            # if trainframe%8==0:
            optimizer.step()
            # optimizer.zero_grad()
            del dataset1, out
        # optimizer.step()
        # optimizer.zero_grad()
        trainlosslist.append(all_loss)
        torch.cuda.empty_cache()
    return all_loss
def eval_model(args, model, test_data, testol, device, t, testacclist):
    correct = 0
    model.eval()
    # t.set_postfix(testsum=len(test_data))
    with torch.no_grad():
        for testframe, data1 in enumerate(test_data):
            pred = model(data1.to(device))
            x, y = pred.reshape(-1, 100), data1.y.reshape(-1, 100)*100
            correct += np.sum((torch.topk(torch.abs_(x - y), 1).values.detach().cpu().squeeze().numpy()) < args.th*100)
            # if testframe % 40 == 0:
                
            t.set_description('test iter %i/%i' % (testframe+1, len(test_data)))
            del data1, pred, x, y
        testacc = correct / testol
        t.set_postfix(testacc=str(testacc)[:6])
        testacclist.append(testacc)
        torch.cuda.empty_cache()
    return testacc
def save_log(trainlosslist, testacclist):
    test = pd.DataFrame(columns=['test_accuracy'], data=testacclist)
    test.to_csv('./model/test_accuracy.csv', index=0)
    # mox.file.copy('/cache/test_accuracy.csv', os.path.join(Context.get_model_path(), 'test_accuracy.csv'))
    # test = pd.DataFrame(columns='train_accuracy',data=trainacclist)
    # test.to_csv(os.path.join(Context.get_result_path(),'train_accuracy.csv'))

    test = pd.DataFrame(columns=['train_loss'], data=trainlosslist)
    test.to_csv('./model/train_loss.csv', index=0)
    # mox.file.copy('/cache/train_loss.csv', os.path.join(Context.get_model_path(), 'train_loss.csv'))
def model_configuration(args, model):
    # if not os.path.exists('./model'):
    #     os.mkdir('./model')
    torch.cuda.manual_seed(10000), torch.manual_seed(10000), np.random.seed(10000), rand.seed(10000)
    torch.backends.cudnn.enabled, torch.backends.cudnn.benchmark, CUDA_LAUNCH_BLOCKING = True, True, 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    w, max_acc, max_acc1 = torch.randn(10000, 10000), args.tgac, args.tgac
    nn.init.kaiming_normal_(w, mode='fan_in', nonlinearity='relu')

    para = sum([np.prod(list(p.size())) for p in model.parameters()])
    print('Model {} : params: {:4f}M'.format(model._get_name(), para * 4 / 1000 / 1000))
    time.sleep(0.5)
    print('ok')
    return device, max_acc, max_acc1
def model_denoise(args, traindata_tem, traindata_label, testdata_tem, testdata_label):
    '''
    '''
    # model = gat(traindata_tem.size(2))
    model = sage(traindata_tem.size(2))
    device, max_acc, max_acc1 = model_configuration(args, model)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epoch,eta_min=5e-5)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=False, threshold=args.ts, threshold_mode='rel', cooldown=0, min_lr=0, eps=2.5e-5)
    # scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=10,T_mult=2,eta_min=1e-5)

    edge_index = get_edge_index()
    # dataset = Data(x=torch.randn(100,3),edge_index=edge_index,y=torch.rand(100,1))
    train_set = TempDataset(traindata_tem, traindata_label, edge_index)
    test_set = TempDataset(testdata_tem, testdata_label, edge_index)

    train_data = DataLoader(dataset=train_set, batch_size=args.bs, shuffle=True, num_workers=args.nw, pin_memory=False)
    test_data = DataLoader(dataset=test_set, batch_size=args.bs*2, shuffle=False, num_workers=args.nw, pin_memory=False)

    trainlosslist = []
    testol, testacc, testacclist = testdata_tem.size(0), 0.0, []

    with tqdm(range(args.epoch), ncols=120) as t:
        for e in t:
            all_loss = train_model(model, train_data, optimizer, device, t, trainlosslist)
            testacc = eval_model(args, model, test_data, testol, device, t, testacclist)
            # t.set_description_str()
            if testacc - max_acc1 > 0.002:
                max_acc1 = testacc
                torch.save(model.state_dict(), os.path.join('./model', str(testacc)[:7] + ".pth"))
                if len(os.listdir(os.path.join('./model'))) >= 30:
                    os.remove(os.listdir(os.path.join('./model'))[0])
            # scheduler.step(all_loss)
            scheduler.step()
            print('Epoch:{:<3d},Train loss:{:.4f},Test Accuracy:{:.4f},lr=:{:.5f}'.format(
                e, all_loss, testacc, optimizer.state_dict()['param_groups'][0]['lr']))
            save_log(trainlosslist,testacclist)
    model.cpu()
    torch.cuda.empty_cache()
    time.sleep(5)
    del model
    return max_acc if max_acc1 < max_acc else max_acc1