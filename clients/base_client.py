import copy, torch
from data_handler.mnist_handler import TestLoader

class BaseClient(object):
    def __init__(self, dev_id = None, cfg=None, model_para = None, model_net = None, batch_size = None, dev = None) -> None:
        self.cfg = cfg
        if model_para:
            self.model_para = model_para
        else:
            self.model_para = {}
        self.net = model_net
        self.dev_id = dev_id
        self.batch_size = batch_size
        self.dev = dev

    def sent_parameter(self, server):
        server.receive(self.net.state_dict(), self.loader.__len__())
        
    def set_parameter(self, model_para):
        for key, val in model_para.items():
            self.model_para[key] = val.clone()
        # for new, old in zip(model_para, self.model_para):
        #     old.data = new.data.clone()


    # def get_parameter(self):

    # def _debug(self, loader):
    #     self.net.eval()
    #     sum_accu = 0.0
    #     num = 0
    #     loader = MnistTestLoader(r"./data/MNIST", 100)
    #     for data, label in loader.get_batches():
    #         data, label = data.to(self.dev), label.to(self.dev)
    #         # print(test_para == global_parameters)
    #         preds = self.net(data)
    #         preds = torch.argmax(preds, dim=1)
    #         # label = torch.argmax(label, dim=1)
    #         sum_accu += (preds == label).float().mean()
    #         num += 1
    #     print("[debug]\n"+'accuracy: {}'.format(sum_accu / num))
        
        
    