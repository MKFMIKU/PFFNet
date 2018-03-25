import torch
import torch.nn as nn
from model.net import ConvLayer, UpsampleConvLayer, ResidualBlock
import torch.nn.functional as F

class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, sign):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.bias.data = float(sign) * torch.Tensor(rgb_mean) * rgb_range

        # Freeze the MeanShift layer
        for params in self.parameters():
            params.requires_grad = False

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        rgb_mean = (0.5204, 0.5167, 0.5129)
        self.sub_mean = MeanShift(1., rgb_mean, -1)
        self.add_mean = MeanShift(1., rgb_mean, 1)

        self.conv_input = ConvLayer(3, 16, kernel_size=11, stride=1)
        self.conv2x = ConvLayer(16, 32, kernel_size=3, stride=2)
        self.conv4x = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.conv8x = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.conv16x = ConvLayer(128, 256, kernel_size=3, stride=2)


        self.dehaze = nn.Sequential()
        self.dehaze.add_module('res1', ResidualBlock(256))
        self.dehaze.add_module('res2', ResidualBlock(256))
        self.dehaze.add_module('res3', ResidualBlock(256))
        self.dehaze.add_module('res4', ResidualBlock(256))
        self.dehaze.add_module('res5', ResidualBlock(256))
        self.dehaze.add_module('res6', ResidualBlock(256))
        self.dehaze.add_module('res7', ResidualBlock(256))
        self.dehaze.add_module('res8', ResidualBlock(256))
        self.dehaze.add_module('res9', ResidualBlock(256))
        self.dehaze.add_module('res10', ResidualBlock(256))
        self.dehaze.add_module('res11', ResidualBlock(256))
        self.dehaze.add_module('res12', ResidualBlock(256))
        self.dehaze.add_module('res13', ResidualBlock(256))
        self.dehaze.add_module('res14', ResidualBlock(256))
        self.dehaze.add_module('res15', ResidualBlock(256))
        self.dehaze.add_module('res16', ResidualBlock(256))
        self.dehaze.add_module('res17', ResidualBlock(256))
        self.dehaze.add_module('res18', ResidualBlock(256))


        self.convd16x = UpsampleConvLayer(256, 128, kernel_size=3, stride=2)
        self.convd8x = UpsampleConvLayer(128, 64, kernel_size=3, stride=2)
        self.convd4x = UpsampleConvLayer(64, 32, kernel_size=3, stride=2)
        self.convd2x = UpsampleConvLayer(32, 16, kernel_size=3, stride=2)

        self.conv_output = ConvLayer(16, 3, kernel_size=3, stride=1)

        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.relu(self.conv_input(x))
        res2x = self.relu(self.conv2x(x))
        res4x = self.relu(self.conv4x(res2x))

        res8x = self.relu(self.conv8x(res4x))
        res16x = self.relu(self.conv16x(res8x))

        res_dehaze = res16x
        res16x = self.dehaze(res16x)
        res16x = torch.add(res_dehaze, res16x)

        res16x = self.relu(self.convd16x(res16x))
        res16x = F.upsample(res16x, res8x.size()[2:], mode='bilinear')
        res8x = torch.add(res16x, res8x)

        res8x = self.relu(self.convd8x(res8x))
        res8x = F.upsample(res8x, res4x.size()[2:], mode='bilinear')
        res4x = torch.add(res8x, res4x)

        res4x = self.relu(self.convd4x(res4x))
        res4x = F.upsample(res4x, res2x.size()[2:], mode='bilinear')
        res2x = torch.add(res4x, res2x)

        res2x = self.relu(self.convd2x(res2x))
        res2x = F.upsample(res2x, x.size()[2:], mode='bilinear')
        x = torch.add(res2x, x)

        x = self.conv_output(x)

        return x