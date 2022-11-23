import torch
import torch.optim as optim


def SimpleMultiStepLR(optimizer, total_epoch, gamma=0.1, **kw):
    milestones = [int(total_epoch * 0.5)]
    return optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma, **kw)


def PostMultiStepLR(optimizer, total_epoch, gamma=0.2, **kw):
    milestones = [int(total_epoch * 0.5)]
    # milestones = [int(total_epoch * 0.2), int(total_epoch * 0.5)]
    return optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma, **kw)
    # return optim.lr_scheduler.ExponentialLR(optimizer, gamma, **kw)


def PriorMultiStepLR(optimizer, total_epoch, gamma=0.2, **kw):
    milestones = [int(total_epoch * 0.2), int(total_epoch * 0.5)]
    return optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma, **kw)

def PriorExponentialLR(optimizer, total_epoch, gamma=0.9, **kw):
    return optim.lr_scheduler.ExponentialLR(optimizer, gamma, **kw)


