import torch
import torch.nn as nn

class LivenessNet(nn.Module):
    def __init__(self, depth=3, classes=2):
        super(LivenessNet, self).__init__()
        # First CONV => RELU => CONV => RELU => POOL layer set
        self.conv1_1 = nn.Conv2d(depth, 16, kernel_size=3, padding=1)
        self.relu1_1 = nn.ReLU()
        self.bn1_1 = nn.BatchNorm2d(16)
        
        self.conv1_2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.relu1_2 = nn.ReLU()
        self.bn1_2 = nn.BatchNorm2d(16)
        
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout2d(p=0.25)
        
        # Second CONV => RELU => CONV => RELU => POOL layer set
        self.conv2_1 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2_1 = nn.ReLU()
        self.bn2_1 = nn.BatchNorm2d(32)
        
        self.conv2_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.relu2_2 = nn.ReLU()
        self.bn2_2 = nn.BatchNorm2d(32)
        
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout2d(p=0.25)
        
        # First (and only) set of FC => RELU layers
        self.flatten = nn.Flatten()
        # 32x32 image with 2 max pools of stride 2 -> 8x8 spatial dimensions
        self.fc1 = nn.Linear(32 * 8 * 8, 64)
        self.relu_fc = nn.ReLU()
        self.bn_fc = nn.BatchNorm1d(64)
        self.dropout_fc = nn.Dropout(p=0.5)
        
        # Softmax classifier (CrossEntropyLoss in PyTorch handles Softmax)
        self.fc2 = nn.Linear(64, classes)

    def forward(self, x):
        x = self.bn1_1(self.relu1_1(self.conv1_1(x)))
        x = self.dropout1(self.pool1(self.bn1_2(self.relu1_2(self.conv1_2(x)))))
        
        x = self.bn2_1(self.relu2_1(self.conv2_1(x)))
        x = self.dropout2(self.pool2(self.bn2_2(self.relu2_2(self.conv2_2(x)))))
        
        x = self.flatten(x)
        
        x = self.bn_fc(self.relu_fc(self.fc1(x)))
        x = self.dropout_fc(x)
        
        x = self.fc2(x)
        return x
        
    @staticmethod
    def build(width, height, depth, classes):
        return LivenessNet(depth=depth, classes=classes)
