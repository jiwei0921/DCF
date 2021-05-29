import numpy as np

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay


def iou(outputs: np.array, labels: np.array):

    intersection = (outputs & labels).sum((1, 2))
    union = ((outputs) | (labels)).sum((1, 2))

    iou = (intersection + 1e-6) / (union + 1e-6)

    return iou.mean()