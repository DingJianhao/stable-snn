import torch
import torch.nn as nn
from torchattacks.attack import Attack


class RFGSM(Attack):
    r"""
    altered from torchattack
    """
    def __init__(self, model, forward_function=None, eps=8/255,
                 alpha=8/255, steps=1, random_start=True, T=None, **kwargs):
        super().__init__("RFGSM", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self._supported_mode = ['default', 'targeted']
        self.forward_function = forward_function
        self.T = T

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self._targeted:
            target_labels = self._get_target_label(images, labels)

        loss = nn.CrossEntropyLoss()
        images = images + self.alpha*torch.randn_like(images)# torch.empty_like(images).uniform_(-self.alpha, self.alpha)

        images.requires_grad = True
        if self.forward_function is not None:
            outputs = self.forward_function(self.model, images, self.T)
        else:
            outputs = self.model(images)

        # Calculate loss
        if self.targeted:
            cost = -loss(outputs, target_labels)
        else:
            cost = loss(outputs, labels)

        # Update adversarial images
        grad = torch.autograd.grad(cost, images,
                                   retain_graph=False, create_graph=False)[0]

        adv_images = images + self.eps*grad.sign()
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        return adv_images