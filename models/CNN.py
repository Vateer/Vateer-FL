import torch
import torch.nn as nn
import torch.nn.functional as F

class Mnist_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(7*7*64, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, inputs):
        tensor = inputs.view(-1, 1, 28, 28)
        tensor = F.relu(self.conv1(tensor))
        tensor = self.pool1(tensor)
        tensor = F.relu(self.conv2(tensor))
        tensor = self.pool2(tensor)
        tensor = tensor.view(-1, 7*7*64)
        tensor = F.relu(self.fc1(tensor))
        tensor = self.fc2(tensor)
        return tensor

class Cifa10_CNN(nn.Module):
    def __init__(self, in_features=3, num_classes=10, dim=1600):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features,
                        32,
                        kernel_size=5,
                        padding=0,
                        stride=1,
                        bias=True),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32,
                        64,
                        kernel_size=5,
                        padding=0,
                        stride=1,
                        bias=True),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.fc1 = nn.Sequential(
            nn.Linear(dim, 512), 
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.fc(out)
        return out    

# class Mnist_CNN(nn.Module):
#     def __init__(self, in_features=1, num_classes=10, dim=1024):
#         super().__init__()
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(in_features,
#                         32,
#                         kernel_size=5,
#                         padding=0,
#                         stride=1,
#                         bias=True),
#             nn.ReLU(inplace=True), 
#             nn.MaxPool2d(kernel_size=(2, 2))
#         )
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(32,
#                         64,
#                         kernel_size=5,
#                         padding=0,
#                         stride=1,
#                         bias=True),
#             nn.ReLU(inplace=True), 
#             nn.MaxPool2d(kernel_size=(2, 2))
#         )
#         self.fc1 = nn.Sequential(
#             nn.Linear(dim, 512), 
#             nn.ReLU(inplace=True)
#         )
#         self.fc = nn.Linear(512, num_classes)

#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.conv2(out)
#         out = torch.flatten(out, 1)
#         out = self.fc1(out)
#         out = self.fc(out)
#         return out
# class BaseHeadSplit(nn.Module):
#     def __init__(self, base, head):
#         super(BaseHeadSplit, self).__init__()

#         self.base = base
#         self.head = head
        
#     def forward(self, x):
#         out = self.base(x)
#         out = self.head(out)

#         return out