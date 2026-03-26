import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect
import sys

def get_architecture(name: str):
    """
    Returns the nn.Module class with an EXACT name match.
    If no match is found, returns None.
    
    Note: Returns the class itself, NOT an instance.
    """
    current_module = sys.modules[__name__]

    for attr_name, obj in inspect.getmembers(current_module):
        # Check it's a class, subclass of nn.Module, but not nn.Module itself
        if (
            inspect.isclass(obj)
            and issubclass(obj, nn.Module)
            and obj is not nn.Module
        ):
            if attr_name == name:
                return obj

    return None


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
    

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

class RadarCNN_GAP(nn.Module):
    def __init__(self, input_height, input_width, num_classes):
        super().__init__()

        # -------- Feature Extractor --------
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # -------- Global Average Pool --------
        self.gap = nn.AdaptiveAvgPool2d(1)

        # -------- Lightweight Classifier --------
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = self.gap(x)          # (B, 64, 1, 1)
        x = torch.flatten(x, 1)  # (B, 64)

        x = self.fc(x)

        return x
    
import torch
import torch.nn as nn
import torch.quantization as tq

class RadarCNN_GAP_INT8(nn.Module):
    def __init__(self, input_height, input_width, num_classes):
        super().__init__()

        # Required for quantization
        self.quant = tq.QuantStub()
        self.dequant = tq.DeQuantStub()

        # -------- Feature Extractor --------
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.quant(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        x = self.dequant(x)
        return x

    # Required for fusion
    def fuse_model(self):
        tq.fuse_modules(self.conv1, ['0', '1', '2'], inplace=True)
        tq.fuse_modules(self.conv2, ['0', '1', '2'], inplace=True)
        tq.fuse_modules(self.conv3, ['0', '1', '2'], inplace=True)


class cnn_lite_v1(nn.Module):
    def __init__(self, input_height, input_width, num_classes=4):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.gap = nn.AdaptiveAvgPool2d((1 , 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = torch.flatten(x,1)
        x = self.fc(x)
        return x
    
class cnn_lite_v2(nn.Module):
    # Doesnt reduce doppler dimension.
    # DOES NOT have batchnorm, maybe add later
    def __init__(self, input_height, input_width, num_classes=4):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),

            nn.Conv2d(16, 32, 3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
        )

        self.gap = nn.AdaptiveAvgPool2d((1 , 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = torch.flatten(x,1)
        x = self.fc(x)
        return x


class cnn_lite_v3(nn.Module):
    # V1 -> increase classifier from 64 to 96
    def __init__(self, input_height, input_width, num_classes=4):
        # Note: input_height, input_width are only there to make it plug n run in the train pipeline
        #       input_height, input_width are not actually used in this cnn
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 96, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.gap = nn.AdaptiveAvgPool2d((1 , 1))
        self.fc = nn.Linear(96, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = torch.flatten(x,1)
        x = self.fc(x)
        return x

class cnn_lite_d1(nn.Module):
    def __init__(self, input_height, input_width, num_classes=4):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 12, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2,1)),   # preserve Doppler

            nn.Conv2d(12, 24, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2,1)),

            nn.Conv2d(24, 48, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),   # compress at end
        )

        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(48, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class cnn_distill_d2(nn.Module):
    def __init__(self, input_height, input_width, num_classes=4):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 24, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),

            nn.Conv2d(24, 48, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),

            nn.Conv2d(48, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return x


class cnn_distill_d3(nn.Module):
    def __init__(self, input_height, input_width, num_classes=4):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.GELU(),
        )

        self.block2 = nn.Sequential(
            DepthwiseSeparableConv(16, 32),
            nn.GELU(),
            nn.MaxPool2d(2)
        )

        self.block3 = nn.Sequential(
            DepthwiseSeparableConv(32, 48),   # not 64 → controlled capacity
            nn.GELU(),
            nn.MaxPool2d(2)
        )

        self.block4 = nn.Sequential(
            DepthwiseSeparableConv(48, 64),
            nn.GELU(),
            nn.MaxPool2d(2)
        )

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
# 3 convolutional layers
class cnn_distill_d3_3block(nn.Module):
    def __init__(self, input_height, input_width, num_classes=4):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.GELU(),
        )

        self.block2 = nn.Sequential(
            DepthwiseSeparableConv(16, 32),
            nn.GELU(),
            nn.MaxPool2d(2)
        )

        self.block3 = nn.Sequential(
            DepthwiseSeparableConv(32, 48),
            nn.GELU(),
            nn.MaxPool2d(2)
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(48, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# 2 convolutional layers
class cnn_distill_d3_2block(nn.Module):
    def __init__(self, input_height, input_width, num_classes=4):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.GELU(),
        )

        self.block2 = nn.Sequential(
            DepthwiseSeparableConv(16, 32),
            nn.GELU(),
            nn.MaxPool2d(2)
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
class cnn_distill_d4(nn.Module):
    def __init__(self, input_height, input_width, num_classes=4):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 12, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.MaxPool2d(2),

            # Block 2
            nn.Conv2d(12, 24, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.MaxPool2d(2),

            # Block 3
            nn.Conv2d(24, 48, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.MaxPool2d(2),
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(48, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
# D5 — one conv layer
class cnn_distill_d5(nn.Module):
    def __init__(self, input_height, input_width, num_classes=4):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.MaxPool2d(2),
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(16, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# D6 — two conv layers
class cnn_distill_d6(nn.Module):
    def __init__(self, input_height, input_width, num_classes=4):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 12, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.MaxPool2d(2),

            nn.Conv2d(12, 24, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.MaxPool2d(2),
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(24, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x