import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

class TensorNormalization(nn.Module):
    def __init__(self,mean, std):
        super(TensorNormalization, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.mean = mean
        self.std = std
    def forward(self,X):
        return normalizex(X,self.mean,self.std)
    def extra_repr(self) -> str:
        return 'mean=%s, std=%s'%(self.mean, self.std)

def normalizex(tensor, mean, std):
    mean = mean[None, :, None, None]
    std = std[None, :, None, None]
    if mean.device != tensor.device:
        mean = mean.to(tensor.device)
        std = std.to(tensor.device)
    return tensor.sub(mean).div(std)

class AddDim(nn.Module):
    def __init__(self, T, dt):
        super().__init__()
        self.T = T
        self.dt = dt

    def forward(self, x: torch.Tensor):
        T = int(self.T/self.dt)
        x.unsqueeze_(1)
        return x.repeat(T, 1, 1, 1, 1)

class MergeTemporalDim(nn.Module):
    def __init__(self, T, dt=1):
        super().__init__()
        self.T = T
        self.dt = dt

    def forward(self, x_seq: torch.Tensor):
        return x_seq.flatten(0, 1).contiguous()

class ExpandTemporalDim(nn.Module):
    def __init__(self, T, dt=1):
        super().__init__()
        self.T = T
        self.dt = dt

    def forward(self, x_seq: torch.Tensor):
        T = int(self.T/self.dt)
        y_shape = [T, int(x_seq.shape[0]/T)]
        y_shape.extend(x_seq.shape[1:])
        return x_seq.view(y_shape)

class ZIF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gama):
        out = (input >= 0).float()
        L = torch.tensor([gama])
        ctx.save_for_backward(input, out, L)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input, out, others) = ctx.saved_tensors
        gama = others[0].item()
        grad_input = grad_output
        tmp = (1 / gama) * (1 / gama) * ((gama - input.abs()).clamp(min=0))
        grad_input = grad_input * tmp
        return grad_input, None

def lif_forward(model, x):
    x = model.expand(x)
    mem = 0
    spikes = []
    for t in range(model.T):
        mem = mem * model.tau + x[t, ...]
        spike = model.act(mem - model.thresh, model.gama)
        mem = (1 - spike) * mem
        spikes.append(spike)
    x = torch.stack(spikes, dim=0)
    x = model.merge(x)
    return x

def dlif_forward(model, x):
    x = model.expand(x)
    mem = 0
    spikes = []
    for t in range(model.T):
        mem = mem * model.tau + x[t, ...] * model.p[t]
        spike = model.act(mem - model.thresh, model.gama)
        mem = (1 - spike) * mem
        spikes.append(spike)
    x = torch.stack(spikes, dim=0)
    x = model.merge(x)
    return x

class DLIFSpike(nn.Module):
    def __init__(self, T, thresh=1.0, tau=0.99, gama=1.0):
        super(DLIFSpike, self).__init__()
        self.act = ZIF.apply
        self.thresh = thresh
        self.tau = tau
        self.gama = gama
        self.expand = ExpandTemporalDim(T)
        self.merge = MergeTemporalDim(T)
        self.T = T
        self._forward = dlif_forward
        self.p = torch.nn.Parameter(torch.ones(T,))

    def forward(self, x):
        return self._forward(self, x)

class LIFSpike(nn.Module):
    def __init__(self, T, thresh=1.0, tau=0.99, gama=1.0):
        super(LIFSpike, self).__init__()
        self.act = ZIF.apply
        self.thresh = thresh
        self.tau = tau
        self.gama = gama
        self.expand = ExpandTemporalDim(T)
        self.merge = MergeTemporalDim(T)
        self.T = T
        self._forward = lif_forward

    def forward(self, x):
        return self._forward(self, x)

def add_dimention(x, T):
    x.unsqueeze_(1)
    x = x.repeat(T, 1, 1, 1, 1)
    return x

def build_neuron_function(type='LIFSpike', **kwargs):
    k = set(['T', 'thresh', 'tau', 'gama']) & set(kwargs.keys())
    if type.lower() == 'lifspike':
        class_func = LIFSpike
    else:
        class_func = DLIFSpike
    return partial(class_func,**{_k:kwargs[_k] for _k in k})

class ConvexCombination(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n
        self.comb = nn.Parameter(torch.ones(n) / n)

    def forward(self, *args):
        assert(len(args) == self.n)
        out = 0.
        for i in range(self.n):
            out += args[i] * self.comb[i]
        return out

