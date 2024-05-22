import argparse
import os
import torch
import warnings
import torch.nn as nn
import torch.nn.parallel
import torch.optim
from data_loaders import cifar10, cifar100, mnist
from functions import *
import attack
from models import *
from utils import forward_function #, train, val
import models.layers as layers
import time
from tqdm import tqdm
from collections import defaultdict
import torch.nn.functional as F
import copy
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler as GradScaler
from stable_modules import LyapunovFunction
from regularize import snn_rat_reg1, snn_rat_reg2

# torch.backends.cudnn.enabled = False

scaler = GradScaler()


parser = argparse.ArgumentParser()
# just use default setting
parser.add_argument('-j','--workers',default=8, type=int,metavar='N',help='number of data loading workers')
parser.add_argument('-bs','--batch_size',default=64, type=int,metavar='N',help='mini-batch size')
parser.add_argument('-sd', '--seed',default=42,type=int,help='seed for initializing training. ')
parser.add_argument('--optim', default='sgd',type=str,help='optimizer')
parser.add_argument('-suffix','--suffix',default='', type=str,help='suffix')

# model configuration
parser.add_argument('-data', '--dataset',default='cifar10',type=str,help='dataset')
parser.add_argument('-arch','--model',default='vgg5',type=str,help='model')
parser.add_argument('-T','--time',default=8, type=int,metavar='N',help='snn train time')

# neuron configuration
parser.add_argument('-ntype','--neuron_type',default='DLIFSpike',type=str)
parser.add_argument('-vth','--vth',default=1., type=float)
parser.add_argument('-tau','--tau',default=0.99, type=float)
parser.add_argument('-gama','--gama',default=1.0, type=float)

# training configuration
parser.add_argument('--epochs',default=100,type=int,metavar='N',help='number of total epochs to run')
parser.add_argument('-lr','--lr',default=0.1,type=float,metavar='LR', help='initial learning rate')
parser.add_argument('-lr_p','--lr_p',default=0.1,type=float,metavar='LR', help='initial learning rate')
parser.add_argument('-dev','--device',default='0',type=str,help='device')

# adv training configuration
parser.add_argument('-training_mode','--training_mode', default='Raw', type=str, help='[Raw, Gaussian, AT_FGSM, RAT_FGSM]')
parser.add_argument('-fix_p','--fix_p', default='no', type=str, help='[yes no]')
parser.add_argument('-beta','--beta',default=5e-4, type=float,help='regulation beta')

parser.add_argument('-kappa','--kappa',default=0.5, type=float,help='ratio of clean training')
parser.add_argument('-GN','--GNnoise', default=8, type=float, metavar='N', help='GN noise budget') # 
parser.add_argument('-alpha','--alpha', default=1, type=float, metavar='N', help='lynapunov perturb data alpha') # 
parser.add_argument('-eps','--eps',default=4, type=float, metavar='N',help='attack eps') #
# parser.add_argument('-steps','--steps',default=1, type=int, metavar='N',help='attack steps')

args = parser.parse_args()
# fp16 = (args.fp16 != 0)
# os.environ["CUDA_VISIBLE_DEVICES"] = args.device
device = torch.device("cuda:%s" % args.device) #torch.device("cuda" if torch.cuda.is_available() else "cpu")

internal_state = {}
fx = {}
neuron_input = {}

if_dlif = False

def ff(model, image, T):
    output = model(image).mean(0)
    return output

def simple_dynamic(x, tau=0.99):
    T = x.shape[0]
    l = []
    f = []
    mem = torch.zeros_like(x[0, ...]).to(x)
    for t in range(T):
        fx = (tau - 1) * mem + x[t, ...] 
        f.append(fx)
        mem = mem * tau + x[t, ...]
        l.append(mem)
    return torch.stack(l, dim=0), torch.stack(f, dim=0)

def dlif_forward_for_V(model, x):
    global internal_state, fx, neuron_input, if_dlif
    x = model.expand(x)
    mem = 0.
    spikes = []
    mems = []
    f = []
    xx = []
    for t in range(model.T):
        if if_dlif:
            dmem = mem * (model.tau - 1) + x[t, ...] * model.p[t]
        else:
            dmem = mem * (model.tau - 1) + x[t, ...]
        mem = mem + dmem
        spike = model.act(mem - model.thresh, model.gama)
        sub = spike * mem
        mem = mem - sub
        spikes.append(spike)
        mems.append(mem)
        f.append(dmem - sub)
        xx.append(x[t, ...])
    x = torch.stack(spikes, dim=0)
    x = model.merge(x)
    internal_state['last'] = torch.stack(mems, dim=0)
    fx['last'] = torch.stack(f, dim=0)
    neuron_input['last'] = torch.stack(xx, dim=0)
    return x

def dlif_forward(model, x):
    global if_dlif
    x = model.expand(x)
    mem = 0
    spikes = []
    for t in range(model.T):
        if if_dlif:
            mem = mem * model.tau + x[t, ...] * model.p[t]
        else:
            mem = mem * model.tau + x[t, ...]
        
        mem = mem * model.tau + x[t, ...] * model.p[t]
        spike = model.act(mem - model.thresh, model.gama)
        mem = (1 - spike) * mem
        spikes.append(spike)
    x = torch.stack(spikes, dim=0)
    x = model.merge(x)
    return x

def main():
    global args, id_to_name, dynamic_layers, id_to_module, if_dlif
    if args.dataset.lower() == 'cifar10':
        num_labels = 10
        input_type = 'img'
        in_h = 32
        in_w = 32
        init_c = 3
        train_dataset, val_dataset, znorm = cifar10()
    elif args.dataset.lower() == 'cifar100':
        num_labels = 100
        input_type = 'img'
        in_h = 32
        in_w = 32
        init_c = 3
        train_dataset, val_dataset, znorm = cifar100()
    elif args.dataset.lower() == 'mnist':
        num_labels = 10
        input_type = 'img'
        in_h = 28
        in_w = 28
        init_c = 1
        train_dataset, val_dataset, znorm = mnist()
        znorm = None

    log_dir = '%s-checkpoints'% (args.dataset)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    seed_all(args.seed)
    args.dt = 1
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                              shuffle=False, num_workers=args.workers, pin_memory=True)

    NeuronFunction = layers.build_neuron_function(type=args.neuron_type, T=args.time, dt=args.dt,
                 thresh=args.vth, tau=args.tau, gama=args.gama)
    
    if 'dli' in args.neuron_type.lower():
        if_dlif = True
    
    if 'vgg' in args.model.lower():
        model = VGG(vgg_name=args.model, T=args.time, dt=args.dt, num_classes=num_labels, norm=znorm,
              neuron_module=NeuronFunction, 
              init_c=init_c, in_h=in_h, in_w=in_w, input_type=input_type)
    elif 'wrn' in args.model.lower():
        model = WideResNet(name=args.model, T=args.time, dt=args.dt, num_classes=num_labels, norm=znorm,
              neuron_module=NeuronFunction, 
              init_c=init_c, in_h=in_h, in_w=in_w, input_type=input_type)
    elif 'res' in args.model.lower():
        model = ResNet(resnet_name=args.model, T=args.time, dt=args.dt, num_classes=num_labels, norm=znorm,
              neuron_module=NeuronFunction, 
              init_c=init_c, in_h=in_h, in_w=in_w, input_type=input_type)
    else:
        raise AssertionError("model not supported")
    
    model.to(device)

    last_lif_module = None
    lif_modules = nn.ModuleList()
    for n, m in model.named_modules():
        if isinstance(m, nn.Module) and hasattr(m, '_forward'):
            last_lif_module = m
        if isinstance(m, nn.Linear):
            last_weight_layer = m
        if isinstance(m, layers.DLIFSpike) or isinstance(m, layers.LIFSpike):
            lif_modules.append(m)
            
    

    criterion = nn.CrossEntropyLoss().to(device)

    if 'rat' in args.training_mode.lower():
        if args.optim.lower() == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)
        if args.optim.lower() == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0)
    else:
        if args.optim.lower() == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.beta)
        elif args.optim.lower() == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.beta)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    lyapunov_function = LyapunovFunction('square')

    best_acc = 0
    identifier = args.model
    identifier += args.neuron_type[:3] #+ 'xxx'# + # + time.strftime("%m%d%H%M%S")

    if args.training_mode.lower() == 'raw':
        identifier += '_RAW'
    elif args.training_mode.lower() == 'gaussian':
        identifier += '_Gaus[%.4f,%.2f,%.2f]' % (args.alpha, args.GNnoise,args.kappa)
    elif args.training_mode.lower() == 'at_fgsm':
        identifier += '_ATFGSM[%.4f,%.2f,%.2f]' % (args.alpha, args.eps,args.kappa)
    elif args.training_mode.lower() == 'rat_fgsm':
        identifier += '_RATFGSM[%.4f,%.2f,%.2f]' % (args.alpha, args.eps,args.kappa)
    
    # if args.fix_p.lower() == 'yes':
    #     identifier += '_fixp'
    identifier += 'v2' + args.suffix

    logger = get_logger(os.path.join(log_dir, '%s.log'%(identifier)))
    logger.info('start training!')
    for arg in vars(args):
        logger.info('{}: {}'.format(arg, getattr(args, arg)))

    last_lif_module._forward = dlif_forward_for_V
    for epoch in range(args.epochs):
        running_loss = 0
        running_loss_V = 0

        M = len(train_loader)
        total = 0
        correct = 0
        
        for i, (images, labels) in enumerate(tqdm(train_loader)):
            labels = labels.to(device)
            images = images.to(device)

            if 'fgsm' in args.training_mode.lower():
                model.eval()
                with autocast():
                    clean_outputs = model(images)
                    clean_neuron_input = neuron_input['last'].detach().clone()
                    clean_outputs = clean_outputs.clone()
                images_adv = images.clone().detach() + 0.001 * torch.randn(images.shape).detach().to(device)
                with autocast():
                    images_adv.requires_grad_()
                    with torch.enable_grad():
                        outputs = model(images_adv)
                        loss_adv = criterion(outputs.mean(0), labels)
                        grad = torch.autograd.grad(loss_adv, [images_adv])[0]
                        images_adv = images_adv + args.eps*grad.sign() / 255
                        images_adv = torch.clamp(images_adv, min=0, max=1).detach()
            elif args.training_mode.lower() == 'gaussian':
                images_adv = images.clone().detach() + args.GNnoise / 255 * torch.randn(images.shape).detach().to(device)
            
            model.train()
            # print(last_lif_module.p)
            if args.training_mode.lower() == 'raw':
                optimizer.zero_grad()
                with autocast():
                    clean_outputs = model(images)
                    loss = criterion(clean_outputs.mean(0), labels)
                running_loss += loss.item()
                scaler.scale(loss).backward()
                _, predicted = clean_outputs.mean(0).cpu().max(1)
                scaler.step(optimizer)
                
            if args.training_mode.lower() == 'gaussian' or 'fgsm' in args.training_mode.lower():
                # clean to label
                model.train()
                optimizer.zero_grad()
                with autocast():
                    clean_outputs = model(images)
                    clean_neuron_input = neuron_input['last'].detach().clone()
                    adv_outputs = model(images_adv)
                    loss = criterion(clean_outputs.mean(0), labels) * args.kappa + (1 - args.kappa) * criterion(adv_outputs.mean(0), labels)
                    
                    Y, fx = simple_dynamic(clean_neuron_input - neuron_input['last'], tau=args.tau)
                    V, dVdY = lyapunov_function(Y)
                    loss_V = V * args.alpha
                    
                running_loss += loss.item()
                running_loss_V += V.item()
                
                actual_loss = loss + loss_V
                scaler.scale(actual_loss).backward()
                
                scaler.step(optimizer)
                
                _, predicted = clean_outputs.mean(0).cpu().max(1)
            
            scaler.update()

            if 'rat' in args.training_mode.lower():
                snn_rat_reg1(model, args.beta)
                snn_rat_reg2(model)

            correct += float(predicted.eq(labels.cpu()).sum().item())
            total += float(labels.size(0))
            

        logger.info('Epoch:[{:04d}/{:04d}] Train acc={:.3f}'.format(epoch , args.epochs, 100 * correct / total))
        logger.info('                  CE_loss={:.5f}   sim loss={:.5f}'.format(running_loss, running_loss_V))
        print(identifier)
        scheduler.step()
        # scheduler_lif.step()

        # validation
        correct = 0
        total = 0
        model.eval()
        for batch_idx, (inputs, targets) in enumerate(tqdm(test_loader)):
            inputs = inputs.to(device)
            with torch.no_grad():
                outputs = model(inputs).mean(0)
            _, predicted = outputs.cpu().max(1)
            total += float(targets.size(0))
            correct += float(predicted.eq(targets).sum().item())
        val_acc = 100 * correct / total
        logger.info('Epoch:[{:04d}/{:04d}] Test acc={:.3f}\n'.format(epoch , args.epochs, val_acc))

        if best_acc < val_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(log_dir, '%s.pth'%(identifier)))

    logger.info('Best Test acc={:.3f}'.format(best_acc))

if __name__ == "__main__":
    main()
