import random
from models.layers import *

cfg = {
    'resnet17': [
        [64, 64, 3, 2],
        [64, 128, 4, 2],
        [],
        []
    ],
    'resnet19': [
        [128, 128, 3, 1],
        [128, 256, 3, 2],
        [256, 512, 2, 2],
        []
    ]
}

class conv(nn.Module):
    def __init__(self,in_plane, out_plane, kernel_size, stride, padding, bias=True):
        super(conv, self).__init__()
        self.fwd = nn.Sequential(nn.Conv2d(in_plane,out_plane,kernel_size=kernel_size,stride=stride,padding=padding, bias=bias),
        nn.BatchNorm2d(out_plane))

    def forward(self,x):
        x = self.fwd(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, neuron_module, stride=1, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.stride = stride
        self.conv1 = conv(in_ch, out_ch, 3, stride, 1, bias=False)
        self.neuron1 = neuron_module()
        self.conv2 = conv(out_ch, out_ch, 3, 1, 1, bias=False)
        self.neuron2 = neuron_module()
        self.right = shortcut

    def forward(self, input):
        out = self.conv1(input)
        out = self.neuron1(out)
        out = self.conv2(out)
        residual = input if self.right is None else self.right(input)
        out += residual
        out = self.neuron2(out)
        return out


class ResNet(nn.Module):
    def __init__(self, resnet_name, T, dt, num_classes, 
                 neuron_module, norm=None, 
                 init_c=3, in_h=32, in_w=32, input_type='img', 
                 stable=False):
        super(ResNet, self).__init__()
        self.resnet_name = resnet_name
        self.neuron_module = neuron_module
        self.T = T
        self.dt = dt
        self.input_type = input_type
        self.stable = stable
        if init_c == 3:
            if norm is not None and isinstance(norm, tuple):
                self.norm = TensorNormalization(*norm)
            else:
                self.norm = TensorNormalization((0, 0, 0), (1, 1, 1))
        elif init_c == 1: # mnist
            self.norm = TensorNormalization((0), (1))
        _cfg = cfg[self.resnet_name]
        self.pre_conv = conv(init_c, _cfg[0][0], 3, stride=1, padding=1, bias=False)
        self.neuron1 = neuron_module()
        p = 1
        out_c = 0
        if len(_cfg[0]) > 0:
            self.layer1 = self.make_layer(_cfg[0][0], _cfg[0][0], _cfg[0][2], stride=_cfg[0][3])
            p *= _cfg[0][3]
            out_c = _cfg[0][1]
        else:
            self.layer1 = nn.Sequential()
        if len(_cfg[1]) > 0:
            self.layer2 = self.make_layer(_cfg[1][0], _cfg[1][1], _cfg[1][2], stride=_cfg[1][3])
            p *= _cfg[1][3]
            out_c = _cfg[1][1]
        else:
            self.layer2 = nn.Sequential()
        if len(_cfg[2]) > 0:
            self.layer3 = self.make_layer(_cfg[2][0], _cfg[2][1], _cfg[2][2], stride=_cfg[2][3])
            p *= _cfg[2][3]
            out_c = _cfg[2][1]
        else:
            self.layer3 = nn.Sequential()
        if len(_cfg[3]) > 0:
            self.layer4 = self.make_layer(_cfg[3][0], _cfg[3][1], _cfg[3][2], stride=_cfg[3][3])
            p *= _cfg[3][3]
            out_c = _cfg[3][1]
        else:
            self.layer4 = nn.Sequential()
        self.pool = nn.AvgPool2d(2,2)
        W = in_h // (p * 2)
        self.fc1 = nn.Sequential(nn.Linear(out_c*W*W, 256), nn.BatchNorm1d(256))
        self.neuron2 = neuron_module()
        self.fc2 = nn.Linear(256, num_classes)
        self.merge = MergeTemporalDim(int(T/dt))
        self.expand = ExpandTemporalDim(int(T/dt))

    def make_layer(self, in_ch, out_ch, block_num, stride=1):
        shortcut = conv(in_ch, out_ch, 1, stride, 0, bias=False)
        layers = []
        layers.append(ResidualBlock(in_ch, out_ch, self.neuron_module, stride, shortcut))
        for i in range(1, block_num):
            layers.append(ResidualBlock(out_ch, out_ch, self.neuron_module))
        return nn.Sequential(*layers)

    def forward(self, input):
        if self.input_type == 'img':
            input = self.norm(input)
            input = add_dimention(input, int(self.T/self.dt))
        input = self.merge(input)
        
        x = self.pre_conv(input)
        x = self.neuron1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.neuron2(x)
        out = self.fc2(x)
        out = self.expand(out)
        return out
