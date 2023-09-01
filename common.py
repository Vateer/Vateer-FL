import numpy as np
import torch
import random

#distribute：pathological / practical, 
def create_non_iid_data_splits(images:np.array, labels:np.array, num_clients:int, distribute = "pra", alpha = 0.1, batch_size=10, class_per_client=0, balance = False):
    '''
    Split training data in non-iid way

    Inputs:
    images: training data
    labels: label data
    num_clients: how many clients will split the data
    distribute：non_iid type
    alpha: parameter of Dirichlet, the smaller alpha, the degree of non-iid higher.

    Returns
    data_splits = [[train_data1, label1],[train_data2, label2]...]
    '''
    data_splits = []

    least_sample = batch_size * 2 # ?
    if distribute == "pathological" or distribute == "pat":
        # unique_classes = np.unique(labels)
        # num_sample = labels.shape[0]
        # data_remain_idx = []
        # avg_data_points_per_client = num_samples / num_clients

        # # # 根据标签排序
        # # sorted_indices = np.argsort(labels)
        # # sorted_images = images[sorted_indices]
        # # sorted_labels = labels[sorted_indices]

        # for cls in range(unique_classes.__len__()):
        #     data_remain_idx.append(np.where(labels == cls)[0])
        #     np.random.shuffle(data_remain_idx[-1])

        # # 计算每个客户端应分配的样本数量
        # # num_samples_per_label = np.bincount(sorted_labels) #计算某个值出现了多少次
        # delta = avg_data_points_per_client // 2
        # num_data_points_per_client = np.random.randint(avg_data_points_per_client - delta, avg_data_points_per_client + delta, num_clients) 
        # num_data_points_per_client[-1] = num_samples - np.sum(num_data_points_per_client[:-1]) #这里会有问题

        # for i in range(num_clients):
        #     #客户端随机分配某几个类
        #     client_classes = np.random.choice(unique_classes, size=np.random.randint(1, unique_classes.__len__()//2), replace=False)
        #     #计算客户端每个类需要分配的数据量
        #     class_data_num = num_data_points_per_client[i] // client_classes.__len__()
        #     class_delta = class_data_num // 2
        #     need_cls_data = np.random.randint(class_data_num-class_delta, class_data_num+class_delta, client_classes.__len__())
        #     need_cls_data[-1] = num_data_points_per_client[i] - np.sum(need_cls_data[:-1])
        #     client_image = None
        #     client_label = None
        #     for idx, cl in enumerate(client_classes):
        #         if data_remain_idx[cl].__len__() == 0:
        #             cl = 0 
        #             while data_remain_idx[cl].__len__() == 0:
        #                 cl += 1
        #         minus = int(min(need_cls_data[idx], data_remain_idx[cl].__len__()))
        #         choice = data_remain_idx[cl][0:minus]
        #         if type(client_image) != type(np.array([])):
        #             client_image = images[choice]
        #             client_label = labels[choice]
        #         else:
        #             client_image = np.append(client_image, images[choice], axis=0)
        #             client_label = np.append(client_label, labels[choice], axis=0)
        #         data_remain_idx[cl] = data_remain_idx[cl][minus:]
        #     #shuffle the data
        #     indices = np.arange(len(client_image))
        #     np.random.shuffle(indices)
        #     client_image = client_image[indices]
        #     client_label = client_label[indices]
        #     data_splits.append([client_image, client_label])

        # for cl in unique_classes:
        #     if data_remain_idx[cl].__len__():
        #         while data_remain_idx[cl].__len__():
        #             minus = int(min(data_remain_idx[cl].__len__(), class_data_num//(unique_classes.__len__()//2)))
        #             choice = data_remain_idx[cl][0:minus]
        #             add_client = np.random.randint(1, data_splits.__len__())
        #             data_splits[add_client][0] = np.append(data_splits[add_client][0], images[choice], axis=0) 
        #             data_splits[add_client][1] = np.append(data_splits[add_client][1], labels[choice], axis=0) 
        #             #shuffle
        #             indices = np.arange(len(data_splits[add_client][0]))
        #             np.random.shuffle(indices)
        #             data_splits[add_client][0] = data_splits[add_client][0][indices]
        #             data_splits[add_client][1] = data_splits[add_client][1][indices]
        #             data_remain_idx[cl] = data_remain_idx[cl][minus:]
        data_splits = [[] for _ in range(num_clients)]
        unique_classes = np.unique(labels)
        num_class = unique_classes.__len__()
        idxs = np.array(range(len(labels)))
        idxs_each_class = []
        for i in range(num_class):
            idxs_each_class.append(idxs[labels == i])
        class_num_per_client = [class_per_client] * num_clients
        for i in range(num_class):
            selected = []
            for client_idx in range(num_clients):
                if class_num_per_client[client_idx] > 0:
                    selected.append(client_idx)
                selected = selected[:int(np.ceil(num_clients/num_class * class_per_client))] #每个类分配的客户端数量

            sample_num = len(idxs_each_class[i])
            sample_per_client = sample_num / len(selected) #每个客户端分配的平均数量
            if balance == True:
                num_samples = [int(sample_per_client) for _ in range(len(selected))]
            else:
                # num_samples = np.random.randint(max(sample_per_client/10, least_sample/num_class), sample_per_client, len(selected)-1).tolist()
                num_samples = [int(sample_per_client) for _ in range(len(selected))]
                for idx in range(num_samples.__len__()//2):
                    delta = np.random.randint(-(num_samples[idx] // 1.5), num_samples[idx] // 1.5)
                    num_samples[idx] -= delta
                    num_samples[num_samples.__len__() - idx - 1] += delta

            idx = 0
            for client, num in zip(selected, num_samples):
                if data_splits[client] == []:
                    data_splits[client] = [images[idxs_each_class[i][idx:idx+num]], labels[idxs_each_class[i][idx:idx+num]]]
                else:
                    data_splits[client][0] = np.append(data_splits[client][0], images[idxs_each_class[i][idx:idx+num]], axis=0)
                    data_splits[client][1] = np.append(data_splits[client][1], labels[idxs_each_class[i][idx:idx+num]], axis=0)
                idx += num
                class_num_per_client[client] -= 1

    elif distribute == "practical" or distribute == "pra":
        min_size = 0
        num_samples = labels.shape[0]
        classes = np.unique(labels).__len__()
        unique_classes = np.unique(labels)
        while min_size < least_sample:
            idx_batch = [[] for _ in range(num_clients)]
            for cs in range(len(unique_classes)):
                idx_cs = np.where(labels == cs)[0]
                np.random.shuffle(idx_cs)
                proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
                proportions = np.array([p*(len(idx)<num_samples/num_clients) for p,idx in zip(proportions,idx_batch)])
                proportions = proportions/proportions.sum()
                proportions = (np.cumsum(proportions)*len(idx_cs)).astype(int)[:-1] #获得各个客户端样本的起始坐标
                # np.split(idx_cs,proportions) 将idx_cs按照proportions的下标分割
                idx_batch = [idx + idx2.tolist() for idx,idx2 in zip(idx_batch,np.split(idx_cs,proportions))] 
                min_size = min([len(idx_j) for idx_j in idx_batch])
        for client in range(num_clients):
            data_splits.append([images[idx_batch[client]], labels[idx_batch[client]]])
    else:
        raise ValueError("wrong iid value")

    return data_splits


def create_iid_data_splits(images, labels, num_clients):
    num_samples = labels.shape[0]
    samples_per_client = num_samples / num_clients

    indices = np.random.permutation(num_samples)
    # 使用打乱后的索引重新排序images和labels
    images = images[indices]
    labels = labels[indices]

    # 分配数据给每个客户端
    data_splits = []
    start = 0
    for i in range(num_clients):
        end = int(start + samples_per_client)
        client_images = images[start:end]
        client_labels = labels[start:end]
        data_splits.append([client_images, client_labels])
        start = end

    return data_splits

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True