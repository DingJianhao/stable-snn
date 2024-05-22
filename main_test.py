import argparse
import os
import sys
from models.VGG import *
import data_loaders
from functions import *
# from utils import val
from models import *
import attack
import copy
import torch
import json
from data_loaders import cifar10, cifar100, mnist
from tqdm import tqdm
import models.layers as layers


parser = argparse.ArgumentParser()
# just use default setting
parser.add_argument('-j','--workers',default=4, type=int,metavar='N',help='number of data loading workers')
parser.add_argument('-b','--batch_size',default=64, type=int,metavar='N',help='mini-batch size')
parser.add_argument('-sd', '--seed',default=42,type=int,help='seed for initializing training.')
parser.add_argument('-suffix','--suffix',default='', type=str,help='suffix')

# model configuration
parser.add_argument('-data', '--dataset', default='cifar10',type=str,help='dataset')
parser.add_argument('-arch','--model', default='vgg11', type=str,help='model')
parser.add_argument('-T','--time', default=8, type=int, metavar='N',help='snn simulation time')
parser.add_argument('-id', '--identifier', type=str, help='model statedict identifier')
parser.add_argument('-config', '--config', default='', type=str,help='test configuration file')

# training configuration
parser.add_argument('-dev','--device',default='0',type=str,help='device')

parser.add_argument('-ntype','--neuron_type',default='DLIFSpike',type=str)
parser.add_argument('-vth','--vth',default=1., type=float)
parser.add_argument('-tau','--tau',default=0.99, type=float)
parser.add_argument('-gama','--gama',default=1.0, type=float)

# adv atk configuration
parser.add_argument('-atk','--attack',default='fgsm', type=str,help='attack')
parser.add_argument('-eps','--eps',default=2, type=float, metavar='N',help='attack eps')
parser.add_argument('-atk_m','--attack_mode',default='', type=str,help='attack mode')

# only pgd
parser.add_argument('-alpha','--alpha',default=2.55/1,type=float,metavar='N',help='pgd attack alpha')
parser.add_argument('-steps','--steps',default=7,type=int,metavar='N',help='pgd attack steps')
parser.add_argument('-atk_ls', '--attack_loss', default='ce', type=str, metavar='N', help='attack loss')
parser.add_argument('-bb','--bbmodel',default='',type=str,help='black box model') # vgg11_clean_l2[0.000500]bb
args = parser.parse_args()

args.dt = 1
# os.environ["CUDA_VISIBLE_DEVICES"] = args.device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:%s" % args.device) #torch.device("cuda" if torch.cuda.is_available() else "cpu")

def forward_function(model, image, T):
    output = model(image).mean(0)
    return output

def val(model, test_loader, device, atk):
    if atk is not None and not hasattr(atk, 'set_training_mode'):
        atk.set_training_mode = atk.set_model_training_mode
    correct = 0
    total = 0
    model.eval()
    for batch_idx, (inputs, targets) in enumerate(tqdm(test_loader)):
        inputs = inputs.to(device)
        if atk is not None:
            atk.set_training_mode(model_training=False, batchnorm_training=False, dropout_training=False)
            inputs = atk(inputs, targets.to(device))
        with torch.no_grad():
            outputs = model(inputs).mean(0)# / model.dt
        _, predicted = outputs.cpu().max(1)
        total += float(targets.size(0))
        correct += float(predicted.eq(targets).sum().item())

    final_acc = 100 * correct / total
    return final_acc

def main():
    global args
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

    log_dir = '%s-results'% (args.dataset)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    

    model_dir = '%s-checkpoints'% (args.dataset)
    if not os.path.exists(os.path.join(model_dir, args.identifier + '.pth')):
        model_dir = '%s-staticcheckpoints'% (args.dataset)
    if not os.path.exists(os.path.join(model_dir, args.identifier + '.pth')):
        print('error')
        exit(0)
    
    log_name = '%s.log'%(args.identifier+args.suffix)
    if len(args.config)!=0:
        log_name = '[%s]' % args.config + log_name
    logger = get_logger(os.path.join(log_dir, log_name))
    logger.info('start testing!')

    seed_all(args.seed)
    test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                              shuffle=False, num_workers=args.workers, pin_memory=True)

    NeuronFunction = layers.build_neuron_function(type=args.neuron_type, T=args.time, dt=args.dt,
                 thresh=args.vth, tau=args.tau, gama=args.gama)

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

    # model.set_simulation_time(args.time)
    model.to(device)

    # have bb model
    if len(args.bbmodel) > 0:
        bbmodel = copy.deepcopy(model)
        bbstate_dict = torch.load(os.path.join(model_dir, args.bbmodel+'.pth'), map_location=torch.device('cpu'))
        bbmodel.load_state_dict(bbstate_dict, strict=False)
    else:
        bbmodel = None

    if len(args.config) > 0:
        with open(args.config+'.json', 'r') as f:
            config = json.load(f)
    else:
        config = [{}]
    for atk_config in config:
#         logger.info(json.dumps(atk_config))
        for arg in atk_config.keys():
            setattr(args, arg, atk_config[arg])
        if 'bb' in atk_config.keys() and atk_config['bb']:
            atkmodel = bbmodel
        else:
            atkmodel = model

        ff = forward_function

        if args.attack.lower() == 'fgsm':
            atk = attack.FGSM(model, forward_function=ff, eps=args.eps / 255, T=args.time)
            atk.targeted = False
            atk._targeted = False
        elif args.attack.lower() == 'pgd':
            atk = attack.PGD(atkmodel, forward_function=ff, eps=args.eps / 255, alpha=args.alpha / 255, steps=args.steps, T=args.time)
            atk.targeted = False
            atk._targeted = False
        elif args.attack.lower() == 'gn':
            atk = attack.GN(atkmodel, forward_function=ff, eps=args.eps / 255, T=args.time)
        elif args.attack.lower() == 'apgd':
            atk = attack.APGD(atkmodel, forward_function=ff, eps=args.eps / 255, steps=args.steps, T=args.time, loss=args.attack_loss)
        else:
            atk = None
        
        state_dict = torch.load(os.path.join(model_dir, args.identifier + '.pth'), map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        model.to(device)

        acc = val(model, test_loader, device, atk)
        logger.info(json.dumps(atk_config)+' Test acc={:.3f}'.format(acc))

if __name__ == "__main__":
    main()