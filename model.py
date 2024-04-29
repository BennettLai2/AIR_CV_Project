import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

# Double Conv is 2 convolutional layers with BN and RELU. Basic building block for the UNET. 
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.a = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False, dtype=torch.float32)
        self.b = nn.BatchNorm2d(out_channels, dtype=torch.float32)
        self.c = nn.ReLU(inplace=True)
        self.d = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False, dtype=torch.float32)
        self.e = nn.BatchNorm2d(out_channels, dtype=torch.float32)
        self.f = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.a(x)
        x = self.b(x)
        x = self.c(x)
        x = self.d(x)
        x = self.e(x)
        x = self.f(x)
        return x

# Unet model. Architecture explained in this paper: https://arxiv.org/pdf/1505.04597v1.pdf
class UNET(nn.Module):
    # Our unet goes down 4 times and up 4 times. in_channels=3 for RGB (BGR), out_channel=1 for mask (Binary). 
    def __init__(
            self, in_channels=3, out_channels=1, features=[64, 128, 256, 512],
    ):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # down sampling
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # connection at bottleneck
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # up sampling
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            # connection from original down sample to current layer 
            skip_connection = skip_connections[idx//2] 

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)
        x = self.final_conv(x)
        return x

def test():
    # Constructing random input and target. 
    x = torch.randn((3, 3, 160, 160))
    targ = torch.randint(0,2,(3, 1, 160, 160))

    # Local testing ground. not relevant to the rest of the code. 
    model = UNET(in_channels=3, out_channels=1)
    preds = model(x)
    # assert preds.shape == x.shape
    loss_fn = nn.BCEWithLogitsLoss()
    print(loss_fn(preds, targ.float()))

if __name__ == "__main__":
    test()