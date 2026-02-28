import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicRadarCNN(nn.Module):
    def __init__(self, input_height, input_width, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Dynamically compute flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_height, input_width)
            dummy_out = self.features(dummy)
            self.flatten_dim = dummy_out.view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Linear(self.flatten_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

class BasicRadarCNN_m1(nn.Module):
    def __init__(self, input_height, input_width, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(7,3), padding=(3,1)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
        
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
        
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Dynamically compute flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_height, input_width)
            dummy_out = self.features(dummy)
            self.flatten_dim = dummy_out.view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Linear(self.flatten_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
    
class BasicRadarCNN_m2(nn.Module):
    # last layer has 96 filters instead of 64 to increase model capacity. This is to test if the model was underfitting.
    def __init__(self, input_height, input_width, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
        
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
        
            nn.Conv2d(32, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Dynamically compute flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_height, input_width)
            dummy_out = self.features(dummy)
            self.flatten_dim = dummy_out.view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Linear(self.flatten_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
    
class BasicRadarCNN_Classifier64(nn.Module):
    ##Changed classifer from 128 to 64 to reduce model size and overfitting
    def __init__(self, input_height, input_width, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Dynamically compute flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_height, input_width)
            dummy_out = self.features(dummy)
            self.flatten_dim = dummy_out.view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Linear(self.flatten_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
    

class BasicRadarCNN_V3(nn.Module):
    # This version uses global average pooling and a single fully connected layer to reduce overfitting and model size
    def __init__(self, input_height, input_width, num_classes, _1st_conv_filters=32, _2nd_conv_filters=64, _3rd_conv_filters=128):
        super(BasicRadarCNN_V3, self).__init__()

        self.input_height = input_height
        self.input_width = input_width

        # ----------------------------
        # Feature Extractor
        # ----------------------------

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, _1st_conv_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(_1st_conv_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(_1st_conv_filters, _2nd_conv_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(_2nd_conv_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(_2nd_conv_filters, _3rd_conv_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(_3rd_conv_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        # ----------------------------
        # Global Pooling
        # ----------------------------

        self.gap = nn.AdaptiveAvgPool2d(1)

        # ----------------------------
        # Classifier
        # ----------------------------

        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(_3rd_conv_filters, num_classes)

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = self.gap(x)          # (B, _3rd_conv_filters, 1, 1)
        x = torch.flatten(x, 1)  # (B, _3rd_conv_filters)

        x = self.dropout(x)
        x = self.fc(x)

        return x