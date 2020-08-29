
import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph import Conv2D, BatchNorm, Sequential


class ConvBnBlock(fluid.dygraph.Layer):
    def __init__(self, in_channels, out_channels, filter_size, stride=1, padding=0):
        super(ConvBnBlock, self).__init__()
        self.conv = Conv2D(in_channels, out_channels, filter_size, stride=stride, padding=padding, bias_attr=False)
        self.bn = BatchNorm(out_channels, 'leaky_relu')
     
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class BasicBlock(fluid.dygraph.Layer):
    def __init__(self, channels):
        super(BasicBlock, self).__init__()
        self.conv_bn_block1 = ConvBnBlock(channels, int(channels / 2), 1, stride=1, padding=0)
        self.conv_bn_block2 = ConvBnBlock(int(channels / 2), channels, 3, stride=1, padding=1)
    def forward(self, x):
        skip = x 
        x = self.conv_bn_block1(x)
        x = self.conv_bn_block2(x)
        x = x + skip 
        return x
    
    
        
class DarkNet53(fluid.dygraph.Layer):
    def __init__(self):
        super(DarkNet53, self).__init__()
        self.conv_bn_block1 = ConvBnBlock(in_channels=3, out_channels=32, filter_size=3, stride=1, padding=1)
        self.conv_bn_block2 = ConvBnBlock(in_channels=32, out_channels=64, filter_size=3, stride=2, padding=1)
        
        self.layer1 = self.make_layers(64, 1)  
        self.conv_bn_block3 = ConvBnBlock(in_channels=64, out_channels=128, filter_size=3, stride=2, padding=1)
        self.layer2 = self.make_layers(128, 2)
        self.conv_bn_block4 = ConvBnBlock(in_channels=128, out_channels=256, filter_size=3, stride=2, padding=1)
        self.layer3 = self.make_layers(256, 8)
        self.conv_bn_block5 = ConvBnBlock(in_channels=256, out_channels=512, filter_size=3, stride=2, padding=1)
        self.layer4 = self.make_layers(512, 8)
        self.conv_bn_block6 = ConvBnBlock(in_channels=512, out_channels=1024, filter_size=3, stride=2, padding=1)
        self.layer5 = self.make_layers(1024, 4)
    
    def make_layers(self, channels, num_layers):
        layers = []
        for i in range(num_layers):
            layers.append(BasicBlock(channels))
        return Sequential(*layers)
        
    
        
    def forward(self, x):
        
        # stem block
        x = self.conv_bn_block1(x)
        x = self.conv_bn_block2(x)
        # stage 1
        
        x = self.layer1(x)

        # stage 2
        x = self.conv_bn_block3(x)
        x = self.layer2(x)

        pyramid = []

        # stage 3
        x = self.conv_bn_block4(x)
    
        x = self.layer3(x)
        x_at_stage3 = x
        pyramid.append(x_at_stage3)

        # stage 4
        x = self.conv_bn_block5(x)

        x = self.layer4(x)
        x_at_stage4 = x
        pyramid.append(x_at_stage4)

        # stage 5
        x = self.conv_bn_block6(x)

        x = self.layer5(x)
        x_at_stage5 = x
        pyramid.append(x_at_stage5)
        
        return pyramid








