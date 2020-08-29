
import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph import Conv2D, BatchNorm, Sequential


class Mish(fluid.dygraph.Layer):
    """Mish激活函数"""
    def __init__(self):
        super(Mish, self).__init__()
    
    def forward(self, x):
        x = x * fluid.layers.tanh(fluid.layers.log(1 + fluid.layers.exp(x)))
        return x


class ConvBnBlock(fluid.dygraph.Layer):
    def __init__(self, in_channels, out_channels, filter_size, stride=1, padding=0):
        super(ConvBnBlock, self).__init__()
        self.conv = Conv2D(in_channels, out_channels, filter_size, stride=stride, padding=padding, bias_attr=False)
        self.bn = BatchNorm(out_channels)
        self.act = Mish()
     
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class ResidualBlockInCSP(fluid.dygraph.Layer):
    """构建CSP中的Residual小模块"""
    def __init__(self, channels1, channels2):
        super(ResidualBlockInCSP, self).__init__()
        self.conv_bn_block1 = ConvBnBlock(channels1, channels2, 1, stride=1, padding=0)
        self.conv_bn_block2 = ConvBnBlock(channels2, channels1, 3, stride=1, padding=1)
    def forward(self, x):
        skip = x 
        x = self.conv_bn_block1(x)
        x = self.conv_bn_block2(x)
        x = x + skip 
        return x
    
    
class CSPDarkNet53(fluid.dygraph.Layer):
    def __init__(self):
        super(CSPDarkNet53, self).__init__()

        self.conv_bn_block1_1 = ConvBnBlock(in_channels=3, out_channels=32, filter_size=3, stride=1, padding=1)
        self.conv_bn_block1_2 = ConvBnBlock(in_channels=32, out_channels=64, filter_size=3, stride=2, padding=1)
        # 第一个CSP layer,其中的一个支路含有1个ResidualBlock
        self.layer1_part1,  self.layer1_part2= self.make_layers(64, 1) 
        
        self.conv_bn_block2_1 = ConvBnBlock(in_channels=128, out_channels=64, filter_size=1, stride=1, padding=0)
        self.conv_bn_block2_2 = ConvBnBlock(in_channels=64, out_channels=128, filter_size=3, stride=2, padding=1)
        # 第二个CSP layer,其中的一个支路含有2个ResidualBlock
        self.layer2_part1, self.layer2_part2 = self.make_layers(128, 2)

        self.conv_bn_block3_1 = ConvBnBlock(in_channels=128, out_channels=128, filter_size=1, stride=1, padding=0)
        self.conv_bn_block3_2 = ConvBnBlock(in_channels=128, out_channels=256, filter_size=3, stride=2, padding=1)
        # 第三个CSP layer,其中的一个支路含有8个ResidualBlock
        self.layer3_part1, self.layer3_part2 = self.make_layers(256, 8)

        self.conv_bn_block4_1 = ConvBnBlock(in_channels=256, out_channels=256, filter_size=1, stride=1, padding=0)
        self.conv_bn_block4_2 = ConvBnBlock(in_channels=256, out_channels=512, filter_size=3, stride=2, padding=1)
        # 第四个CSP layer,其中的一个支路含有8个ResidualBlock
        self.layer4_part1, self.layer4_part2 = self.make_layers(512, 8)

        self.conv_bn_block5_1 = ConvBnBlock(in_channels=512, out_channels=512, filter_size=1, stride=1, padding=0)
        self.conv_bn_block5_2 = ConvBnBlock(in_channels=512, out_channels=1024, filter_size=3, stride=2, padding=1)
        # 第五个CSP layer,其中的一个支路含有4个ResidualBlock
        self.layer5_part1, self.layer5_part2 = self.make_layers(1024, 4)
    
    def make_layers(self, channels, num_layers):
        """用于构建CSPDarkNet53中的每一个CSPBlock"""
        if num_layers == 1:
            part1 = ConvBnBlock(channels, channels, 1, 1, 0)
            part2 = []
            part2.append(ConvBnBlock(channels, channels, 1, 1, 0))
            for i in range(num_layers):
                part2.append(ResidualBlockInCSP(channels,  int(channels / 2)))
            part2.append(ConvBnBlock(channels, channels, 1, 1, 0))
        else:
            part1 = ConvBnBlock(channels, int(channels / 2), 1, 1, 0)
            part2 = []
            part2.append(ConvBnBlock(channels, int(channels / 2), 1, 1, 0))
            for i in range(num_layers):
                part2.append(ResidualBlockInCSP(int(channels / 2),  int(channels / 2)))
            part2.append(ConvBnBlock(int(channels / 2), int(channels / 2), 1, 1, 0))
        return part1, Sequential(*part2)
        
    def forward(self, x):
        
        # stage 1
        x = self.conv_bn_block1_1(x)
        x = self.conv_bn_block1_2(x)  # downsample
        x_part1 = self.layer1_part1(x)
        x_part2 = self.layer1_part2(x)
        x = fluid.layers.concat([x_part1, x_part2], axis=1)  # concat

        # stage 2
        x = self.conv_bn_block2_1(x)
        x = self.conv_bn_block2_2(x)  # downsample
        x_part1 = self.layer2_part1(x)
        x_part2 = self.layer2_part2(x)
        x = fluid.layers.concat([x_part1, x_part2], axis=1)  # concat

        pyramid = []

        # stage 3
        x = self.conv_bn_block3_1(x)
        x = self.conv_bn_block3_2(x)  # downsample
        x_part1 = self.layer3_part1(x)
        x_part2 = self.layer3_part2(x)
        x = fluid.layers.concat([x_part1, x_part2], axis=1)  # concat
        x_at_stage3 = x
        pyramid.append(x_at_stage3)

        # stage 4
        x = self.conv_bn_block4_1(x)
        x = self.conv_bn_block4_2(x)  # downsample
        x_part1 = self.layer4_part1(x)
        x_part2 = self.layer4_part2(x)
        x = fluid.layers.concat([x_part1, x_part2], axis=1)  # concat
        x_at_stage4 = x
        pyramid.append(x_at_stage4)

        # stage 5
        x = self.conv_bn_block5_1(x)
        x = self.conv_bn_block5_2(x)  # downsample
        x_part1 = self.layer5_part1(x)
        x_part2 = self.layer5_part2(x)
        x = fluid.layers.concat([x_part1, x_part2], axis=1)  # concat
        x_at_stage5 = x
        pyramid.append(x_at_stage5)
        return pyramid








