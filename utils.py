# from cv2 import mean
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
from models.layers import *
import random
import time
# from spikingjelly.activation_based import functional

def forward_function(model, image, T):
    # functional.reset_net(model)
    output = model(image).mean(0)
    return output

def train(model, device, train_loader, criterion, optimizer, atk=None):
    running_loss = 0
    model.train()
    M = len(train_loader)
    total = 0
    correct = 0
    for i, (images, labels) in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()
        labels = labels.to(device)
        images = images.to(device)

        # if atk is not None:
        #     atk.set_training_mode(model_training=False, batchnorm_training=False, dropout_training=False)
        #     images = atk(images, labels)
        
        functional.reset_net(model)
        outputs = model(images).mean(0)# / model.dt
        loss = criterion(outputs, labels)
        running_loss += loss.item()
        loss.mean().backward()
        optimizer.step()

        total += float(labels.size(0))
        _, predicted = outputs.cpu().max(1)
        correct += float(predicted.eq(labels.cpu()).sum().item())

    return running_loss, 100 * correct / total

def val(model, test_loader, device):
    correct = 0
    total = 0
    model.eval()
    for batch_idx, (inputs, targets) in enumerate(tqdm(test_loader)):
        inputs = inputs.to(device)
        # if atk is not None:
        #     atk.set_training_mode(model_training=False, batchnorm_training=False, dropout_training=False)
        #     inputs = atk(inputs, targets.to(device))
        #     model.set_simulation_time(T)
        functional.reset_net(model)
        with torch.no_grad():
            outputs = model(inputs).mean(0)# / model.dt
        _, predicted = outputs.cpu().max(1)
        total += float(targets.size(0))
        correct += float(predicted.eq(targets).sum().item())

    final_acc = 100 * correct / total
    return final_acc


# def convex_constraint(model):
#     with torch.no_grad():
#         for module in model.modules():
#             if isinstance(module, ConvexCombination):
#                 comb = module.comb.data
#                 alpha = torch.sort(comb, descending=True)[0]
#                 k = 1
#                 for j in range(1,module.n+1):
#                     if (1 + j * alpha[j-1]) > torch.sum(alpha[:j]):
#                         k = j
#                     else:
#                         break
#                 gamma = (torch.sum(alpha[:k]) - 1)/k
#                 module.comb.data -= gamma
#                 torch.relu_(module.comb.data)