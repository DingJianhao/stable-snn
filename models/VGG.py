# from sklearn.semi_supervised import SelfTrainingClassifier
from models.layers import *
from models.stable_classifier import StableClassifier

cfg = {
    'vgg5' : [[64, 'A'], 
              [128, 128, 'A'],
              [],
              [],
              []],
    'vgg11': [
        [64, 'A'],
        [128, 256, 'A'],
        [512, 512, 512, 'A'],
        [512, 512],
        []
    ],
    'vgg13': [
        [64, 64, 'A'],
        [128, 128, 'A'],
        [256, 256, 'A'],
        [512, 512, 'A'],
        [512, 512, 'A']
    ],
    'vgg16': [
        [64, 64, 'A'],
        [128, 128, 'A'],
        [256, 256, 256, 'A'],
        [512, 512, 512, 'A'],
        [512, 512, 512, 'A']
    ],
    'vgg19': [
        [64, 64, 'A'],
        [128, 128, 'A'],
        [256, 256, 256, 256, 'A'],
        [512, 512, 512, 512, 'A'],
        [512, 512, 512, 512, 'A']
    ]
}

class VGG(nn.Module):
    def __init__(self, vgg_name, 
                 T, dt, num_classes, 
                 neuron_module, 
                 norm=None, init_c=3, in_h=32, in_w=32, input_type='img'):
        super(VGG, self).__init__()
        assert(isinstance(neuron_module(), nn.Module))
        self.neuron_module = neuron_module
        self.input_type = input_type
        if init_c == 3:
            if norm is not None and isinstance(norm, tuple):
                self.norm = TensorNormalization(*norm)
            else:
                self.norm = TensorNormalization((0, 0, 0), (1, 1, 1))
        elif init_c == 1: # mnist
            self.norm = TensorNormalization((0), (1))
        self.T = T
        self.dt = dt
        self.init_channels = init_c
        
        self.layer1 = self._make_layers(cfg[vgg_name][0])
        self.layer2 = self._make_layers(cfg[vgg_name][1])
        self.layer3 = self._make_layers(cfg[vgg_name][2])
        self.layer4 = self._make_layers(cfg[vgg_name][3])
        self.layer5 = self._make_layers(cfg[vgg_name][4])
        
        if vgg_name == 'vgg5':
            self.W = (in_h // 2 // 2) * (in_w // 2 // 2)
        elif vgg_name == 'vgg11':
            self.W = (in_h // 2 // 2 // 2) * (in_w // 2 // 2 // 2)
        else:
            self.W = (in_h // 2 // 2 // 2 // 2 // 2) * (in_w // 2 // 2 // 2 // 2 // 2)
        
        self.classifier = self._make_classifier(num_classes)
        
        self.merge = MergeTemporalDim(T, dt)
        self.expand = ExpandTemporalDim(T, dt)
        self.adddim = AddDim(T, dt)
        self.flatten = nn.Flatten()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, val=1)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _make_layers(self, cfg):
        layers = []
        for x in cfg:
            if x == 'A':
                layers.append(nn.AvgPool2d(2))
            else:
                layers.append(nn.Conv2d(self.init_channels, x, kernel_size=3, padding=1))
                layers.append(nn.BatchNorm2d(x))
                layers.append(self.neuron_module())
                self.init_channels = x
        return nn.Sequential(*layers)

    def _make_classifier(self, num_class):
        layer = [nn.Linear(self.init_channels*self.W, 4096), 
                 self.neuron_module(), 
                 nn.Linear(4096, 4096), 
                 self.neuron_module(), 
                 nn.Linear(4096, num_class)]    
        return nn.Sequential(*layer)

    def forward(self, input):
        if self.input_type == 'img':
            input = self.norm(input)
            input = self.adddim(input)
        input = self.merge(input)
        out = self.layer1(input)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.flatten(out)
        out = self.classifier(out)
        out = self.expand(out)
        return out
        
    
