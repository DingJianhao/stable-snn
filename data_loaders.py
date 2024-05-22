import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10, CIFAR100, MNIST
import numpy as np
import torch

# code from https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py 
# Improved Regularization of Convolutional Neural Networks with Cutout.
class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask
        return img

def cifar10(cutout=True, download=True):
    aug = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),transforms.ToTensor()]
    if cutout:
        aug.append(Cutout(n_holes=1, length=16))
    transform_train = transforms.Compose(aug)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_dataset = CIFAR10(root='~/dataset/cifar10',
                            train=True, download=download, transform=transform_train)
    val_dataset = CIFAR10(root='~/dataset/cifar10',
                            train=False, download=download, transform=transform_test)
    norm = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    return train_dataset, val_dataset, norm


def cifar100(cutout=True, download=True):
    aug = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),transforms.ToTensor()]
    if cutout:
        aug.append(Cutout(n_holes=1, length=16))
    transform_train = transforms.Compose(aug)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_dataset = CIFAR100(root='~/dataset/cifar100',
                                train=True, download=download, transform=transform_train)
    val_dataset = CIFAR100(root='~/dataset/cifar100',
                            train=False, download=download, transform=transform_test)
    norm = ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    return train_dataset, val_dataset, norm

def mnist(download=True):
    aug = [transforms.RandomCrop(28, padding=4), transforms.RandomHorizontalFlip(),transforms.ToTensor()]
    transform_train = transforms.Compose(aug)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_dataset = MNIST(root='~/dataset/mnist',
                                train=True, download=download, transform=transform_train)
    val_dataset = MNIST(root='~/dataset/mnist',
                            train=False, download=download, transform=transform_test)
    norm = ((0), (1))
    return train_dataset, val_dataset, norm
