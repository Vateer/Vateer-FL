import os
import numpy as np
import torch, math

class MnistDataLoader:
    def __init__(self, data_dir, batch_size):
        self.data_dir = data_dir
        self.batch_size = batch_size
        # self.images, self.labels = self.load_data()
        # print(self.labels.shape)
    
    def __len__(self):
        return self.labels.shape[0]

    def get_loader(self):
        path = r'/data/wangweicheng/czj/Vateer-FL/test'
        return 

class MnistTestLoader:
    def __init__(self, data_dir, batch_size):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.images, self.labels = self.load_data()

    def load_data(self):
        images_file = os.path.join(self.data_dir, 'raw/t10k-images-idx3-ubyte')
        labels_file = os.path.join(self.data_dir, 'raw/t10k-labels-idx1-ubyte')
        images = np.fromfile(images_file, dtype=np.uint8)[16:].reshape((-1, 28, 28))
        labels = np.fromfile(labels_file, dtype=np.uint8)[8:]
        images = self.preprocess_data(images)
        return images, labels

    def preprocess_data(self, images):
        images = images.astype(np.float32)
        images /= 255.0
        mean = np.mean(images)
        std = np.std(images)
        images = (images - mean) / std

        return images

    def get_batches(self):
        num_samples = len(self.labels)
        num_batches = num_samples // self.batch_size

        for i in range(num_batches):
            start = i * self.batch_size
            end = start + self.batch_size
            batch_images = self.images[start:end]
            batch_labels = self.labels[start:end]
            batch_images = self.preprocess_data(batch_images)
            batch_images = torch.from_numpy(batch_images)
            batch_labels = torch.from_numpy(batch_labels)
            
            yield batch_images, batch_labels