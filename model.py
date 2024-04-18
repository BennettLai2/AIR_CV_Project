import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        # self.conv = nn.Sequential(
        #     nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False, dtype=torch.float32),
        #     nn.BatchNorm2d(out_channels, dtype=torch.float32),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False, dtype=torch.float32),
        #     nn.BatchNorm2d(out_channels, dtype=torch.float32),
        #     nn.ReLU(inplace=True),
        # )

        self.a = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False, dtype=torch.float32)
        self.b = nn.BatchNorm2d(out_channels, dtype=torch.float32)
        self.c = nn.ReLU(inplace=True)
        self.d = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False, dtype=torch.float32)
        self.e = nn.BatchNorm2d(out_channels, dtype=torch.float32)
        self.f = nn.ReLU(inplace=True)

    def forward(self, x):
        y = x
        x = self.a(x)
        # if not x.dtype == torch.float32: 
        #         # Define the file path
        #         file_path1 = "errorlog1.txt"
        #         file_path2 = "errorlog2.txt"

        #         # Open the file in write mode
        #         with open(file_path1, 'w') as file:
        #             # Redirect the print output to the file
        #             print(y, file=file)
        #         # Open the file in write mode
        #         with open(file_path2, 'w') as file:
        #             # Redirect the print output to the file
        #             print(x, file=file)
        #         if torch.isnan(x).any():
        #                 print("nan")
        #         raise ValueError("a")
        x = self.b(x)
        # if not x.dtype == torch.float32: 
        #         raise ValueError("b")
        x = self.c(x)
        # if not x.dtype == torch.float32: 
        #         raise ValueError("c")
        x = self.d(x)
        # if not x.dtype == torch.float32: 
        #         raise ValueError("d")
        x = self.e(x)
        # if not x.dtype == torch.float32: 
        #         raise ValueError("e")
        x = self.f(x)
        # if not x.dtype == torch.float32: 
        #         raise ValueError("f")
        return x

class UNET(nn.Module):
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
        # print(x.dtype)
        skip_connections = []

        for down in self.downs:
            # y = x
            # if not x.dtype == torch.float32: 
            #     x = x.float().to(torch.float32)
            #     print(x.dtype)
            # if torch.isnan(x).any():
            #     print("Tensor contains NaN values:")
            #     print(x)
            #     print(y)
            #     print(down)
            #     raise ValueError("A")
            x = down(x)
            # if not x.dtype == torch.float32: 
            #     raise ValueError("DOWN Came out as 16")
            # if torch.isnan(x).any():
            #     print("Tensor contains NaN values:")
            #     print(x)
            #     print(y)
            #     print(down)
            #     raise ValueError("B")
            skip_connections.append(x)
            x = self.pool(x)
            # if not x.dtype == torch.float32: 
            #     x = x.float().to(torch.float32)
            #     print("ITS FUCKING POOL", x.dtype)
            #     raise ValueError("FUCK ME")
            # if torch.isnan(x).any():
            #     print("Tensor contains NaN values:")
            #     print(x)
            #     print(y)
            #     print(down)
            #     raise ValueError("AFF")

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)
        x = self.final_conv(x)
        return x

def test():
    x = torch.randn((3, 3, 160, 160))
    targ = torch.randint(0,2,(3, 1, 160, 160))

    model = UNET(in_channels=3, out_channels=1)
    preds = model(x)
    # assert preds.shape == x.shape
    loss_fn = nn.BCEWithLogitsLoss()
    print(loss_fn(preds, targ.float()))

if __name__ == "__main__":
    test()