import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers import *

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, neuron_module, dropRate=0.3):
        super(BasicBlock, self).__init__()
        self.neuron_module = neuron_module
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.act1 = neuron_module()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.act2 = neuron_module()
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False) or None
        self.convex = ConvexCombination(2)
    def forward(self, x):
        if not self.equalInOut:
            x = self.act1(self.bn1(x))
        else:
            out = self.act1(self.bn1(x))
        out = self.act2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return self.convex(x if self.equalInOut else self.convShortcut(x), out)

class WideResNet(nn.Module):
    def __init__(self, name, T, dt, num_classes, 
                 neuron_module, norm=None, 
                 init_c=3, in_h=32, in_w=32, input_type='img', dropRate=0.0, stable=False):
        super(WideResNet, self).__init__()
        assert(isinstance(neuron_module(), nn.Module))
        self.neuron_module = neuron_module
        self.input_type = input_type
        if "16" in name:
            depth = 16
            widen_factor = 4
        elif "20" in name:
            depth = 28
            widen_factor = 10
        else:
            raise AssertionError("Invalid wide-resnet name: " + name)

        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]

        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6

        if init_c == 3:
            if norm is not None and isinstance(norm, tuple):
                self.norm = TensorNormalization(*norm)
            else:
                self.norm = TensorNormalization((0, 0, 0), (1, 1, 1))
        elif init_c == 1: # mnist
            self.norm = TensorNormalization((0), (1))

        block = BasicBlock
        self.T = T
        self.dt = dt
        self.merge = MergeTemporalDim(int(T/dt))
        self.expand = ExpandTemporalDim(int(T/dt))

        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)

        self.block1 = self._make_layer(block, nChannels[0], nChannels[1], n, 1, dropRate)
        self.block2 = self._make_layer(block, nChannels[1], nChannels[2], n, 2, dropRate)
        self.block3 = self._make_layer(block, nChannels[2], nChannels[3], n, 2, dropRate)

        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.act = self.neuron_module()
        
        if in_h >= 32:
            self.avg_stride = 8
        else:
            self.avg_stride = 4
        
        self.fc = nn.Linear(nChannels[3] * (in_h // (4 * self.avg_stride))  * (in_w // (4 * self.avg_stride)), num_classes)
        self.nChannels = nChannels[3]
        self.flatten = nn.Flatten()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
    
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, self.neuron_module, dropRate))
        return nn.Sequential(*layers)

    # pass T to determine whether it is an ANN or SNN
    # def set_simulation_time(self, T, mode='bptt'):
    #     self.T = T
    #     for module in self.modules():
    #         if isinstance(module, (LIFSpike, ExpandTemporalDim)):
    #             module.T = T
    #             if isinstance(module, LIFSpike):
    #                 module.mode = mode
    #     return
    
    def forward(self, input):
        if self.input_type == 'img':
            input = self.norm(input)
            input = add_dimention(input, int(self.T/self.dt))
        input = self.merge(input)
        out = self.conv1(input)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.act(self.bn1(out))
        out = F.avg_pool2d(out, self.avg_stride)
        out = self.flatten(out)
        out = self.expand(out)
        return self.fc(out)