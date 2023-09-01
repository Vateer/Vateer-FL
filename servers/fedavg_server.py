from servers.base_server import BaseServer
from models import *
from data_handler import *
import torch, wandb
class Server(BaseServer):
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        net_name = cfg["dataset"]
        net_name = net_name[0].upper() + net_name[1:]
        net_name=cfg["net"]+"."+net_name+"_"+cfg["net"]
        self.net = eval(net_name)()
        self.loader = eval(cfg["dataset"] + "_handler").TestLoader(cfg, 100)
        self.dev = cfg["gpu"]
        self.net = self.net.to(self.dev)
        self.parameter={}
        for key, var in self.net.state_dict().items():
            # print("key:"+str(key)+",var:"+str(var))
            print("张量的维度:"+str(var.shape))
            print("张量的Size"+str(var.size()))
            self.parameter[key] = var.clone()
        self.size_cnt = 0
        self.rec_list = []
    def send_model(self, clients:list, client_id:list):
        # for client in clients:
        #     client.set_parameter(self.parameter)
        for idx in client_id:
            clients[idx].set_parameter(self.parameter)
    def receive(self, para, size):
        self.size_cnt += size
        self.rec_list.append((para, size))
    
    def aggregate(self):
        self.parameter=None
        for para, size in self.rec_list:
            if self.parameter == None:
                self.parameter = {}
                for key, var in para.items():
                    self.parameter[key] = size/self.size_cnt * var.clone()
            else:
                for key, var in para.items():
                    self.parameter[key] += size/self.size_cnt * var.clone()
        self.size_cnt = 0
        self.rec_list = []
        self.net.load_state_dict(self.parameter, strict=True)

    def evaluate(self):
        self.net.eval()
        sum_accu = 0
        num = 0
        for data, label in self.loader.get_batches():
            data, label = data.to(self.dev), label.to(self.dev)
            # data = data.view(data.shape[0],1,data.shape[1],data.shape[2])
            # print(test_para == global_parameters)
            preds = self.net(data)
            preds = torch.argmax(preds, dim=1)
            sum_accu += (preds == label).float().mean()
            num += 1      
        print('accuracy: {}'.format(sum_accu / num))
        return sum_accu / num
        

        
    
    