from unet import *

batch_size = 3
height_size, width_size = 572, 572
in_channels = 1
out_channels = 2

unet = UNet(in_channels, out_channels)

x = torch.randn(batch_size, in_channels, height_size, width_size)

y = unet(x)

print(y.shape)
