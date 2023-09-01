from torchvision import datasets
import torch, os, random
import numpy as np
from data_handler import *
from clients.base_client import BaseClient
from models import *

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
        
        if cfg["opti"] == "sgd":
            self.opt = torch.optim.SGD(self.net.parameters(), lr=self.cfg["lr"])
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
    
    def report_distribute(self, server):
        self.local_distribute = [0]*self.loader.get_class_num()
        images, labels = [], []
        for image, label in self.loader.get_batches(bz = 1, process=False):
            label = torch.argmax(label, dim=1)
            self.local_distribute[label] += 1
            images.append(image), labels.append(label[0])
        sorted_indices = np.argsort(labels)
        images = np.concatenate(images, axis=0)[sorted_indices]
        labels = np.array(labels)[sorted_indices]
        idx = 0
        upload_images, upload_label = [], []
        while idx + self.cfg["mean_batch"] < len(images):
            if labels[idx] == labels[idx + self.cfg["mean_batch"]]:
                upload_images.append(np.mean(images[idx: idx + self.cfg["mean_batch"]], axis=0))
                upload_label.append(labels[idx])
            idx += self.cfg["mean_batch"]
        server.receive_distribute(self.local_distribute, upload_images, upload_label)
        
    def balance_distribute(self, server):
        self.download_images, self.download_labels, self.global_distribute = server.balance_distribute(self.local_distribute)
        # idx = np.array(range(len(self.download_images)))
        # np.random.shuffle(idx)
        # self.download_images = self.download_images[idx]
        # self.download_labels = self.download_labels[idx]

        self.idxs = [[]] * self.loader.get_class_num()
        self.add_class_num = 0
        for i, idx in enumerate(self.idxs):
            idx = np.where(self.download_labels == i)[0]
            if idx.__len__() > 0:
                self.add_class_num += 1

        # self.loader_augment = eval(self.cfg["dataset"] + "_handler").DataLoader()
        # self.loader_augment.load_data_manually(self.download_images, self.download_labels)
        if self.cfg["naive"] == 1:
            self.loader.add_data(self.download_images, self.download_labels)


    def train(self):
        self.net.load_state_dict(self.model_para, strict=True)
        self.net.train()
        avg_loss = 0.0
        if self.cfg["naive"] == 1: #放一起训练
            for epoch in range(self.epoch):
                total_loss = 0.0
                num_batches = 0 
                for i, (data, label) in enumerate(self.loader.get_batches()):
                    data, label = data.to(self.dev), label.to(self.dev)
                    preds = self.net(data)
                    loss = self.loss_fun(preds, label)
                    self.opt.zero_grad()
                    loss.backward()
                    self.opt.step()
                    total_loss += loss.item()
                    num_batches += 1
            avg_loss = total_loss / num_batches
            print('Client {}, Average Loss: {:.8f}'.format(self.dev_id, avg_loss))
        elif self.cfg["naive"] == 2: # mixup的方法
            lamb = self.cfg["lamb"]
            for epoch in range(self.epoch):
                total_loss = 0.0
                num_batches = 0 
                for i, (data, label) in enumerate(self.loader.get_batches()):
                    idx = np.random.choice(np.array(range(len(self.download_images))), size = label.shape[0], replace = False)
                    xg, yg = self.download_images[idx], self.download_labels[idx]
                    xg, yg = torch.tensor(xg), torch.tensor(yg)
                    data = (1 - lamb)*data + lamb * xg
                    yg = yg.to(self.dev)
                    data, label = data.to(self.dev), label.to(self.dev)
                    preds = self.net(data)
                    loss1 = (1-lamb)*self.loss_fun(preds, label)
                    loss2 = lamb*self.loss_fun(preds, yg)
                    loss = loss1 + loss2
                    self.opt.zero_grad()
                    loss.backward()
                    self.opt.step()
                    total_loss += loss.item()
                    num_batches += 1
            avg_loss = total_loss / num_batches
            print('Client {}, Average Loss: {:.8f}'.format(self.dev_id, avg_loss))   
        elif self.cfg["naive"] == 3: #放一起训练，loss加权
            # for idx in self.idxs:
            #     np.random.shuffle(idx)
            idx = np.array(range(len(self.download_images)))
            np.random.shuffle(idx)
            self.download_images = self.download_images[idx]
            self.download_labels = self.download_labels[idx]
            lamb = self.cfg["lamb"]
            a2x = max(1, self.download_labels.__len__() // self.loader.__len__())
            a2x *= self.cfg["batch_size"]
            for epoch in range(self.epoch):
                total_loss = 0.0
                num_batches = 0.0

                for i, (data, label) in enumerate(self.loader.get_batches()):
                    
                    data, label = data.to(self.dev), label.to(self.dev)
                    preds = self.net(data)
                    # loss = (1.0 - self.cfg["lamb2"]) * self.loss_fun(preds, label)
                    loss = self.loss_fun(preds, label)
                    self.opt.zero_grad()
                    loss.backward()
                    self.opt.step()
                    total_loss += loss.item()
                    
                    idx = random.sample(range(len(self.download_images)), a2x)
                    
                    data, label = self.download_images[idx], self.download_labels[idx]
                    label = torch.nn.functional.one_hot(torch.tensor(label).to(torch.int64), num_classes=self.loader.get_class_num()).to(dtype=torch.float32)
                    data = torch.from_numpy(data)
                    # data = torch.tensor(data).cuda()
                    data = data.to(self.dev)
                    label = label.to(self.dev)
                    preds = self.net(data)
                    loss = self.cfg["lamb2"] * self.loss_fun(preds, label)
                    self.opt.zero_grad()
                    loss.backward()
                    self.opt.step()
                    total_loss += loss.item()

                    num_batches += 1 + self.cfg["lamb2"]
                    # num_batches += 1 
            avg_loss = total_loss / num_batches
            print('Client {}, Average Loss: {:.8f}'.format(self.dev_id, avg_loss))
        else:
            lamb = self.cfg["lamb"]
            for epoch in range(self.epoch):
                total_loss = 0.0
                num_batches = 0 
                for i, (data, label) in enumerate(self.loader.get_batches()):
                    make_data = (1-lamb) * data
                    make_data.requires_grad_()
                    idg = torch.randint(len(self.get_train_mean), (1, ))
                    xg = self.get_train_mean[idg:idg+1]
                    yg = self.get_label_mean[idg:idg+1]

                    # yg = torch.nn.functional.one_hot(yg.to(torch.int64), num_classes = 10).float()
                    data, label = data.to(self.dev), label.to(self.dev)
                    make_data=make_data.to(self.dev)
                    xg = xg.to(self.dev)
                    yg = yg.to(self.dev)

                    self.opt.zero_grad()
                    
                    preds = self.net(make_data)
                    loss1 = (1 - lamb) * self.loss_fun(preds, label)
                    loss2 = lamb * self.loss_fun(preds, yg.expand_as(preds))

                    #loss1对于make_data求导，保留计算图以便计算高阶导数
                    gradients = torch.autograd.grad(outputs=loss1, inputs=make_data,
                                            create_graph=True, retain_graph=True)[0]
                    
                    loss3 = lamb * torch.inner(gradients.flatten(start_dim=1), xg.flatten(start_dim=1))
                    loss3 = torch.mean(loss3)
                    loss = loss1 + loss2 + loss3
                    loss.backward()
                    self.opt.step()
                    total_loss += loss.item()
                    num_batches += 1
                
            avg_loss = total_loss / num_batches
            print('Client {}, Average Loss: {:.8f}, loss1:{:.8f}, loss2:{:.8f}, loss3:{:.8f}'.format(self.dev_id, avg_loss, loss1, loss2, loss3))
            if self.cfg["lr_decay_accumulated"] > 0:
                self.learning_rate_scheduler.step()
        return avg_loss

                
        # self.debug(self.loader)
        # return (self.net.state_dict(), self.loader.__len__())
