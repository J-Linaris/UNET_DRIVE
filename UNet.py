import torch
import torch.nn as nn
from torch import nn

#---------------------FUNDAMENTAL CLASSES DEFINITION-------------------------------------
class SingleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.s_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.s_conv(x)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.d_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.d_conv(x)
    
class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        feature_map = self.conv(x)
        p = self.pool(feature_map)

        return feature_map, p
    
class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        """
        x1: feature map produzido pelo encoder
        x2: feature map trazido pela skip connection do bloco correspondente
            no encoder
        """
        x1 = self.up(x1)
        x = torch.cat([x1, x2], 1)
        return self.conv(x)
    
#---------------------UNET VARIATIONS CLASSES DEFINITION-------------------------------------

class UNet_d5_simple(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        # ENCODER INITIALIZATIONS: DOWNCONVS
        self.down_convolution_1 = DownSample(in_channels, 64)
        self.down_convolution_2 = DownSample(64, 128)
        self.down_convolution_3 = DownSample(128, 256)
        self.down_convolution_4 = DownSample(256, 512)

        # BOTTLENECK INITIALIZATION
        self.bottle_neck = DoubleConv(512, 1024)

        # DECODER INITIALIZATIONS: UPCONVS
        self.up_convolution_1 = UpSample(1024, 512)
        self.up_convolution_2 = UpSample(512, 256)
        self.up_convolution_3 = UpSample(256, 128)
        self.up_convolution_4 = UpSample(128, 64)

        # LAST PART INITIALIZATIONS
        self.last_conv = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        
        # ENCODER
        fm1, p1 = self.down_convolution_1(x)
        fm2, p2 = self.down_convolution_2(p1)
        fm3, p3 = self.down_convolution_3(p2)
        fm4, p4 = self.down_convolution_4(p3)

        # BOTTLENECK
        b = self.bottle_neck(p4)

        # DECODER
        up_1 = self.up_convolution_1(b, fm4)
        up_2 = self.up_convolution_2(up_1, fm3)
        up_3 = self.up_convolution_3(up_2, fm2)
        up_4 = self.up_convolution_4(up_3, fm1)
        
        # Applies the last convolution and the sigmoid activation 
        fm_final = self.last_conv(up_4)
        out = self.sig(fm_final)
        
        return out


class UNet_d5_double(UNet_d5_simple):
    """
    # one of depth 5 and other of depth 2 (stops in down_conv2)
    """
    def __init__(self, in_channels, num_classes):
        super().__init__(in_channels, num_classes)

        # Here, we add methods in order to make
        # the auxiliary UpConv path and the
        # convolutional aggregation possible:

        # SPECIFIC INITIALIZATIONS FOR DEPTH 2 PATH

        # Bottle neck for depth 2 path
        self.bottle_neck_d2 = DoubleConv(128,256)
        
        # Decoder for depth 2 path
        self.up_convolution_1_d2 = UpSample(256, 128)
        self.up_convolution_2_d2 = UpSample(128, 64)

        # LAST PART INITIALIZATIONS

        # Conv for aggregating the output of both paths
        self.conj_conv = SingleConv(128, 64) 

    def forward(self, x):

        # DOWNCONVS
        fm1, p1 = self.down_convolution_1(x)
        fm2, p2 = self.down_convolution_2(p1)
        fm3, p3 = self.down_convolution_3(p2)
        fm4, p4 = self.down_convolution_4(p3)

        #--------------------------DEPTH 5---------------------------
        # BOTTLENECK DEPTH 5
        b_d5 = self.bottle_neck(p4)
        
        # DECODER DEPTH 5
        up_1_d5 = self.up_convolution_1(b_d5, fm4)
        up_2_d5 = self.up_convolution_2(up_1_d5, fm3)
        up_3_d5 = self.up_convolution_3(up_2_d5, fm2)
        up_4_d5 = self.up_convolution_4(up_3_d5, fm1)

        #--------------------------DEPTH 2---------------------------
        # BOTTLENECK DEPTH 2
        b_d2 = self.bottle_neck_d2(p2)
        
        # DECODER DEPTH 2
        up_1_d2 = self.up_convolution_1_d2(b_d2, fm2)
        up_2_d2 = self.up_convolution_2_d2(up_1_d2, fm1)

        #-------------------RESULTS CONCATENATION--------------------

        # Concatenates the outputs of the last two DoubleConvs
        up_agr = torch.cat([up_4_d5, up_2_d2], 1)
        
        # Applies a SingleConv over this agregated feature map 7
        # (idea of learning how to unify both decoders)
        fm_agr = self.conj_conv(up_agr)

        # Applies the last convolution and the sigmoid activation 
        fm_final = self.last_conv(fm_agr)
        out = self.sig(fm_final)

        return out


class UNet_d5_double_mean(UNet_d5_double):
    """
    """
    def __init__(self, in_channels, num_classes):
        super().__init__(in_channels, num_classes)

    def forward(self, x):

        # DOWNCONVS
        fm1, p1 = self.down_convolution_1(x)
        fm2, p2 = self.down_convolution_2(p1)
        fm3, p3 = self.down_convolution_3(p2)
        fm4, p4 = self.down_convolution_4(p3)

        #--------------------------DEPTH 5---------------------------
        # BOTTLENECK DEPTH 5
        b_d5 = self.bottle_neck(p4)
        
        # UPCONVS DEPTH 5
        up_1_d5 = self.up_convolution_1(b_d5, fm4)
        up_2_d5 = self.up_convolution_2(up_1_d5, fm3)
        up_3_d5 = self.up_convolution_3(up_2_d5, fm2)
        up_4_d5 = self.up_convolution_4(up_3_d5, fm1)

        # LAST TRANSFORMATIONS DEPTH 5
        fm_final_d5 = self.last_conv(up_4_d5)
        out_deep = self.sig(fm_final_d5)


        #--------------------------DEPTH 2---------------------------
        # BOTTLENECK DEPTH 2
        b_d2 = self.bottle_neck_d2(p2)
        
        # UPCONVS DEPTH 2
        up_1_d2 = self.up_convolution_1_d2(b_d2, fm2)
        up_2_d2 = self.up_convolution_2_d2(up_1_d2, fm1)

        # LAST TRANSFORMATIONS DEPTH 2
        fm_final_d2 = self.last_conv(up_2_d2)
        out_shallow = self.sig(fm_final_d2)

        #-------------------RESULTS CONCATENATION--------------------
        # Concatenates the outputs using the mean
        out = (out_shallow+out_deep)/2

        return out

class UNet_d4_simple(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        
        # ENCODER INITIALIZATIONS: DOWNCONVS
        self.down_convolution_1 = DownSample(in_channels, 64)
        self.down_convolution_2 = DownSample(64, 128)
        self.down_convolution_3 = DownSample(128, 256)

        # BOTTLENECK
        self.bottle_neck = DoubleConv(256, 512)

        # DECODER INITIALIZATIONS: UPCONVS
        self.up_convolution_1 = UpSample(512, 256)
        self.up_convolution_2 = UpSample(256, 128)
        self.up_convolution_3 = UpSample(128, 64)

        # LAST PART INITIALIZATIONS
        self.last_conv = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1)
        self.sig = nn.Sigmoid()

    def forward(self, x):

        # ENCODER
        fm1, p1 = self.down_convolution_1(x)
        fm2, p2 = self.down_convolution_2(p1)
        fm3, p3 = self.down_convolution_3(p2)

        # BOTTLENECK
        b = self.bottle_neck(p3)
        
        # DECODER
        up_1 = self.up_convolution_1(b, fm3)
        up_2 = self.up_convolution_2(up_1, fm2)
        up_3 = self.up_convolution_3(up_2, fm1)
        
        # Applies the last convolution and the sigmoid activation
        fm_final = self.last_conv(up_3)
        out = self.sig(fm_final)
        
        return out

class UNet_d4_double(UNet_d4_simple):
    """
    """
    def __init__(self, in_channels, num_classes):
        super().__init__(in_channels, num_classes)

        # Here, we add methods in order to make
        # the auxiliary UpConv path possible:

        # SPECIFIC INITIALIZATIONS FOR DEPTH 1 PATH

        # Bottle neck for depth 1 path
        self.bottle_neck_d1 = DoubleConv(64,128)
        
        # Decoder for depth 1 path
        self.up_convolution_1_d1 = UpSample(128, 64)

        # LAST PART INITIALIZATIONS

        # Conv for aggregating the output of both paths
        self.conj_conv = SingleConv(128, 64)


    def forward(self, x):

        # ENCODER
        fm1, p1 = self.down_convolution_1(x)
        fm2, p2 = self.down_convolution_2(p1)
        fm3, p3 = self.down_convolution_3(p2)

        #--------------------------DEPTH 4---------------------------
        # BOTTLENECK DEPTH 4
        b_d4 = self.bottle_neck(p3)
        
        # DECODER DEPTH 4
        up_1_d4 = self.up_convolution_1(b_d4, fm3)
        up_2_d4 = self.up_convolution_2(up_1_d4, fm2)
        up_3_d4 = self.up_convolution_3(up_2_d4, fm1)
        
        #--------------------------DEPTH 1---------------------------
        # BOTTLENECK DEPTH 1
        b_d1 = self.bottle_neck_d1(p1)
        
        # DECODER DEPTH 1
        up_1_d1 = self.up_convolution_1_d1(b_d1, fm1)

        #-------------------RESULTS CONCATENATION--------------------

        # Concatenates the outputs of the last two DoubleConvs
        up_agr = torch.cat([up_3_d4, up_1_d1], 1)

        # Applies a SingleConv over this agregated feature map
        #  (idea of learning how to unify both decoders)
        fm_agr = self.conj_conv(up_agr)

        # Applies the last convolution and the sigmoid activation 
        fm_final = self.last_conv(fm_agr)
        out = self.sig(fm_final)

        return out


class UNet_d4_double_mean(UNet_d4_double):
    """
    """
    def __init__(self, in_channels, num_classes):
        super().__init__(in_channels, num_classes)

    def forward(self, x):

        # ENCODER
        fm1, p1 = self.down_convolution_1(x)
        fm2, p2 = self.down_convolution_2(p1)
        fm3, p3 = self.down_convolution_3(p2)

        #--------------------------DEPTH 4---------------------------
        # BOTTLENECK DEPTH 4
        b_d4 = self.bottle_neck(p3)
        
        # DECODER DEPTH 4
        up_1_d4 = self.up_convolution_1(b_d4, fm3)
        up_2_d4 = self.up_convolution_2(up_1_d4, fm2)
        up_3_d4 = self.up_convolution_3(up_2_d4, fm1)
        
        # LAST TRANSFORMATIONS DEPTH 4
        fm_final_d4 = self.last_conv(up_3_d4)
        out_deep = self.sig(fm_final_d4)

        #--------------------------DEPTH 2---------------------------
        # BOTTLENECK DEPTH 2
        b_d1 = self.bottle_neck_d1(p1)
        
       # DECODER DEPTH 2
        up_1_d1 = self.up_convolution_1_d1(b_d1, fm1)
        
        # LAST TRANSFORMATIONS DEPTH 2
        fm_final_d1 = self.last_conv(up_1_d1)
        out_shallow = self.sig(fm_final_d1)

        #-------------------RESULTS CONCATENATION--------------------
        # Concatenates the outputs using the mean
        out = (out_shallow+out_deep)/2

        return out

if __name__ == "__main__":
    x = torch.randn((2, 3, 512, 512))
    f1 = UNet_d4_double_mean(in_channels=3, num_classes=1)
    f2 = UNet_d4_double(in_channels=3, num_classes=1)
    f3 = UNet_d4_simple(in_channels=3, num_classes=1)
    # f_line = UNet_d5_double(in_channels=3, num_classes=1)
    y = f1(x)
    print(y.shape)
