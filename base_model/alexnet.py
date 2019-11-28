import torch.nn as nn
from builder import ConvBuilder
from constants import SIMPLE_ALEXNET_DEPS


class AlexBN(nn.Module):

    def __init__(self, builder:ConvBuilder, deps=SIMPLE_ALEXNET_DEPS):
        super(AlexBN, self).__init__()
        # self.bd = builder
        stem = builder.Sequential()
        stem.add_module('conv1', builder.Conv2dBNReLU(in_channels=3, out_channels=deps[0], kernel_size=11, stride=4, padding=2))
        stem.add_module('maxpool1', builder.Maxpool2d(kernel_size=3, stride=2))
        stem.add_module('conv2', builder.Conv2dBNReLU(in_channels=deps[0], out_channels=deps[1], kernel_size=5, padding=2))
        stem.add_module('maxpool2', builder.Maxpool2d(kernel_size=3, stride=2))
        stem.add_module('conv3',
                        builder.Conv2dBNReLU(in_channels=deps[1], out_channels=deps[2], kernel_size=3, padding=1))
        stem.add_module('conv4',
                        builder.Conv2dBNReLU(in_channels=deps[2], out_channels=deps[3], kernel_size=3, padding=1))
        stem.add_module('conv5',
                        builder.Conv2dBNReLU(in_channels=deps[3], out_channels=deps[4], kernel_size=3, padding=1))
        stem.add_module('maxpool3', builder.Maxpool2d(kernel_size=3, stride=2))
        self.stem = stem
        self.flatten = builder.Flatten()
        self.linear1 = builder.Linear(in_features=deps[4] * 6 * 6, out_features=4096)
        self.relu1 = builder.ReLU()
        self.drop1 = builder.Dropout(0.5)
        self.linear2 = builder.Linear(in_features=4096, out_features=4096)
        self.relu2 = builder.ReLU()
        self.drop2 = builder.Dropout(0.5)
        self.linear3 = builder.Linear(in_features=4096, out_features=1000)

# '''
# v0/cg/affine0/biases:0 (4096,) 0.0025711544  positive 3151, negative 945, zeros 0
# v0/cg/affine0/weights:0 (9216, 4096) -0.000108150794  positive 17932163, negative 19816573, zeros 0
# v0/cg/affine1/biases:0 (4096,) 0.010545971  positive 4086, negative 10, zeros 0
# v0/cg/affine1/weights:0 (4096, 4096) -0.0002852586  positive 7711397, negative 9065819, zeros 0
# v0/cg/affine2/biases:0 (1001,) -1.2894492e-07  positive 500, negative 501, zeros 0
# v0/cg/affine2/weights:0 (4096, 1001) 9.5962655e-08  positive 1810358, negative 2289738, zeros 0
# v0/cg/conv0/biases:0 (64,) -0.08356731  positive 32, negative 32, zeros 0
# v0/cg/conv0/conv2d/kernel:0 (11, 11, 3, 64) -1.3330064e-05  positive 11684, negative 11548, zeros 0
# v0/cg/conv1/biases:0 (192,) 0.044359412  positive 133, negative 59, zeros 0
# v0/cg/conv1/conv2d/kernel:0 (5, 5, 64, 192) -0.0011889511  positive 150704, negative 156496, zeros 0
# v0/cg/conv2/biases:0 (384,) 0.013650507  positive 234, negative 150, zeros 0
# v0/cg/conv2/conv2d/kernel:0 (3, 3, 192, 384) -0.00074954977  positive 319544, negative 344008, zeros 0
# v0/cg/conv3/biases:0 (384,) 0.012080639  positive 245, negative 139, zeros 0
# v0/cg/conv3/conv2d/kernel:0 (3, 3, 384, 384) -0.00066925987  positive 612052, negative 715052, zeros 0
# v0/cg/conv4/biases:0 (256,) 0.025955569  positive 195, negative 61, zeros 0
# v0/cg/conv4/conv2d/kernel:0 (3, 3, 384, 256) -0.0013808252  positive 379061, negative 505675, zeros 0
# number of kernel params:  61831872
# '''

    def forward(self, x):
        out = self.stem(x)
        # print(out.size())
        out = self.flatten(out)
        out = self.linear1(out)
        out = self.relu1(out)
        out = self.drop1(out)
        out = self.linear2(out)
        out = self.relu2(out)
        out = self.drop2(out)
        out = self.linear3(out)
        return out


def create_alexBN(cfg, builder):
    return AlexBN(builder=builder)
