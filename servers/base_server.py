class BaseServer(object):
    def __init__(self) -> None:
        pass
    def evaluate(self):
        pass
    def send_model(self, clients:list, client_id:list):
        # for client in clients:
        #     client.set_parameter(self.parameter)
        for idx in client_id:
            clients[idx].set_parameter(self.parameter)
    def receive(self, para, size):
        self.size_cnt += size
        self.rec_list.append((para, size))