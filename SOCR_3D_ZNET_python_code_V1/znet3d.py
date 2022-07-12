import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import interpolate

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
          )

    def forward(self,x):
        x= self.double_conv(x)
        #print("Dconv: ", x.shape)
        return x
    
class SingleConv(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.single_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
          )
        

    def forward(self,x):
        #print("x is: ",x.shape)
        x=self.single_conv(x)
        #print("Sconv is: ",x.shape)
        return x

class Down(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.MaxPool3d(2, 2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.encoder(x)

    
class Up(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        
        
        self.up = nn.ConvTranspose3d(in_channels , in_channels, kernel_size=2, stride=2)
            
        self.conv = DoubleConv(in_channels, out_channels)
        #self.relu=nn.ReLU(inplace=True)
    def forward(self, x1):
        #print("Before up x1 and x2 ",x1.shape)
        x1 = self.up(x1)
        #print("After up x1 and x2 ",x1.shape)
        return self.conv(x1)

    
class Out(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size = 1)

    def forward(self, x):
        return self.conv(x)

class znet3d(nn.Module):
    def __init__(self, in_channels, n_classes, n_channels):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.n_channels = n_channels

        self.conv = DoubleConv(in_channels, n_channels)
        self.oneConv0 = SingleConv(25, n_channels)
        
        self.enc1 = Down(n_channels, 2 * n_channels)
        self.oneConv1 = SingleConv(72, 2 * n_channels)
        
        self.enc2 = Down(2 * n_channels, 4 * n_channels)
        self.oneConv2 = SingleConv(144, 4 * n_channels)
        
        self.enc3 = Down(4 * n_channels, 8 * n_channels)
        self.oneConv3 = SingleConv(288, 8 * n_channels)
        
        self.enc4 = Down(8 * n_channels, 16 * n_channels)
        self.oneConv4 = SingleConv(576, 16 * n_channels)

        self.dec1 = Up(16 * n_channels, 8 * n_channels)
        self.oneConvDEC1 = SingleConv(576, 8 * n_channels)
        
        self.dec2 = Up(8 * n_channels, 4 * n_channels)
        self.oneConvDEC2 = SingleConv(288, 4 * n_channels)
        
        self.dec3 = Up(4 * n_channels, 2 * n_channels)
        self.oneConvDEC3 = SingleConv(144, 2 * n_channels)
        
        self.dec4 = Up(2 * n_channels, n_channels)
        self.oneConvDEC4 = SingleConv(72,  n_channels)
        
        
        self.out = Out(n_channels, n_classes)
        
        self.relu=nn.ReLU(inplace=True)
        self.max_pool = nn.Sequential(
            nn.MaxPool3d(2, 2),
            nn.ReLU(inplace=True))
        
    def map_x_x_1(self,x2,x1):

        siz=x2.shape[2:]
        if x1.shape[2:] != x2.shape[2:]:
            #print("b x1: ",x1.shape," X2: ",x2.shape)
            x1=interpolate(x1,size=siz,mode="nearest")
            #print("A x1: ",x1.shape," X2: ",x2.shape)
            x = torch.cat([x2, x1], dim=1)
            x=self.max_pool(x)
        else:
            x = torch.cat([x2, x1], dim=1)
            x=self.relu(x)
        #print("mapx: ", x.shape)
        
        return x
    
    def padding_tensors(self,x1,x2):
       
        if x1.shape[2:] != x2.shape[2:]:
            #print("b x1: ",x1.shape," X2: ",x2.shape)
            x1=interpolate(x1,scale_factor=(2,2,2),mode="nearest")
            #print("A x1: ",x1.shape," X2: ",x2.shape)
        x = torch.cat([x1, x2], dim=1)
        x=self.relu(x)
        return x


    def forward(self, x):
        x1 = self.conv(x)# I MAY NEED TO SHuT IT DOWN AND REMOVE THE ONECONV
        #print("x1: ", x1.shape)
        x1 = self.map_x_x_1(x,x1)
        #print("x1: ", x1.shape)
        x1=self.oneConv0(x1)
        #x1 = self.oneConv0(x1)
        #print("x1: ", x1.shape)
        #return x1
        
        x2 = self.enc1(x1)
        #print("x2: ", x2.shape)
        x2 = self.map_x_x_1(x1,x2)
        #print("x2: ", x2.shape)
        x2=self.oneConv1(x2)
        #print("x2: ", x2.shape)
        #return x2
        
        x3 = self.enc2(x2)
        #print("x3: ", x3.shape)
        x3 = self.map_x_x_1(x2,x3)
        #print("x3: ", x3.shape)
        x3 = self.oneConv2(x3)
        #print("x3: ", x3.shape)
        #return x3
        
        x4 = self.enc3(x3)
        #print("x4: ", x4.shape)
        x4 = self.map_x_x_1(x3,x4)
        #print("x4: ", x4.shape)
        x4 = self.oneConv3(x4)
        #print("x4: ", x4.shape)
        #return x4
        
    
    
        x5 = self.enc4(x4)
        #print("x5: ", x5.shape)
        x5 = self.map_x_x_1(x4,x5)
        #print("x5: ", x5.shape)
        x5 = self.oneConv4(x5)
        #print("x5: ", x5.shape)
        #print("x5: ", x5.shape)
        #return x5
        
       
        
        
        u1 = self.dec1(x5)
        #print("u1: ", u1.shape)
        #x5=interpolate(x5,size=(u1.shape[2],u1.shape[3],u1.shape[4]))
        u1=self.padding_tensors(x5, u1)
        #u1 = torch.cat([u1, x5], dim=1)
        u1=self.oneConvDEC1(u1)
        #print("u1 shape: ", u1.shape)
        #return u1
        
        u2 = self.dec2(u1)
        #u1=interpolate(u1,size=(u2.shape[2],u2.shape[3],u2.shape[4]))
        u2=self.padding_tensors(u1, u2)
        #u2 = torch.cat([u2, u1], dim=1)
        u2=self.oneConvDEC2(u2)
        #u2=self.relu(u2)
        #print("u2 shape: ", u2.shape)
        #return u2
        #print("u2 shape: ", u2.shape)
        
        u3= self.dec3(u2)
        #u2=interpolate(u2,size=(u3.shape[2],u3.shape[3],u3.shape[4]))
        u3=self.padding_tensors(u2, u3)
        #u3 = torch.cat([u3, u2], dim=1)
        u3=self.oneConvDEC3(u3)
        #u3=self.relu(u3)
        #print("u3 shape: ", u3.shape)
        #return u3
        
        
        #print("u3 shape: ", u3.shape)
        u4 = self.dec4(u3)
        #u3=interpolate(u3,size=(u4.shape[2],u4.shape[3],u4.shape[4]))
        u4=self.padding_tensors(u3, u4)
        #u4 = torch.cat([u4, u3], dim=1)
        u4=self.oneConvDEC4(u4)
        
        #print("u4 shape: ", u4.shape)
        
        out=self.out(u4)
        return out
    
if __name__=="__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    znet = znet3d(in_channels=1, n_classes=1, n_channels=24).to(device)
    #output = znet(torch.randn(1,4,75,64,64).to(device))
    #print("out: ",output.shape) 
    from torchsummary import summary
    summary(znet, (1,64,64,64))               
