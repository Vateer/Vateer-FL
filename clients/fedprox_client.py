from torchvision import datasets
import torch, os
import numpy as np
from data_handler import *
from clients.base_client import BaseClient
from models import *
import copy
class FedProxOpti(torch.optim.Optimizer):
    def __init__(self, params, lr=0.01, mu=0.0):
        default = dict(lr=lr, mu=mu)
        #params为待优化的参数，default为优化器的超参数
        super().__init__(params, default) 

    @torch.no_grad()
    #global_params是传进来的全局模型，device是设备类型
    def step(self, global_params, device):
        #self.param_groups表示参数组的列表，每个组包含一组参数及其关联的超参数
        for group in self.param_groups:
            for p, g in zip(group['params'], global_params):
                g = g.to(device)
                d_p = p.grad.data + group['mu'] * (p.data - g.data)
                p.data.add_(d_p, alpha=-group['lr'])


class Client(BaseClient):
    def __init__(self, dev, dev_id, batch_size, cfg:dict):
        net = None
        net_name = cfg["dataset"]
        net_name = net_name[0].upper() + net_name[1:]
        net_name=cfg["net"]+"."+net_name+"_"+cfg["net"]
        net = eval(net_name)()
        super().__init__(dev_id = dev_id, model_para = None, model_net = net, batch_size = batch_size, dev = dev)
        self.loader = None
        self.cfg = cfg
        self.dev = cfg["gpu"]
        self.net = self.net.to(self.dev)
        self.epoch = cfg["epoch"]

        self.loader = eval(cfg["dataset"] + "_handler").DataLoader(cfg, dev_id ,batch_size)
        if cfg["loss_func"] == "cross_entropy":
            self.loss_fun = torch.nn.functional.cross_entropy
        self.opt = FedProxOpti(self.net.parameters(), lr=self.cfg["lr"], mu=self.cfg["mu"])
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.opt, 
            gamma=self.cfg["lr_decay_accumulated"]
        )

    def _debug(self):
        # print("debug")
        with open(r"/data/wangweicheng/czj/Vateer-FL/test/train/"+str(self.dev_id)+".npz", 'rb') as f:
            train_data = np.load(f, allow_pickle=True)['data'].tolist()
        X_train = torch.Tensor(train_data['x']).type(torch.float32)
        y_train = torch.Tensor(train_data['y']).type(torch.int64)
        train_data = [(x, y) for x, y in zip(X_train, y_train)]
        self.loader = torch.utils.data.DataLoader(train_data, self.batch_size, drop_last=True, shuffle=True)
    
    # def train(self):
    #     for epoch in range(self.epoch):
    #         for idx in range(len(self.datas)):
    #             data = torch.from_numpy(self.datas[idx]).to(self.dev)
    #             label = torch.tensor(self.label[idx], dtype=torch.long).to(self.dev)
    #             label = torch.nn.functional.one_hot(label, num_classes = 10).float()
    #             label = torch.unsqueeze(label, 0)
    #             pred = self.net(data)
                
    #             loss = self.loss_fun(pred, label)
    #             loss.backward()
    #             self.opt.step()
    #             self.opt.zero_grad()
    #     return self.net.state_dict()


    def train(self):
        self.net.load_state_dict(self.model_para, strict=True)
        self.net.train()
        global_para = copy.deepcopy(list(self.net.parameters()))
        for epoch in range(self.epoch):
            total_loss = 0.0
            num_batches = 0 
            for i, (data, label) in enumerate(self.loader.get_batches()):
                data, label = data.to(self.dev), label.to(self.dev)
                preds = self.net(data)
                loss = self.loss_fun(preds, label)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step(global_para, self.dev)
                total_loss += loss.item()
                num_batches += 1
        avg_loss = total_loss / num_batches
        print('Client {}, Average Loss: {:.8f}'.format(self.dev_id, avg_loss))
        if self.cfg["lr_decay_accumulated"] > 0:
            self.learning_rate_scheduler.step()
        return avg_loss
            
                
        # self.debug(self.loader)
        # return (self.net.state_dict(), self.loader.__len__())
