


import torch
import torchvision
import torch.nn as nn
from .mean_teacher_losses import KLLoss


losses_dict={'CrossEntropyLoss':nn.CrossEntropyLoss,
             'L1Loss':nn.L1Loss,
             'MeanTeacherLoss':KLLoss,
             }


def builder_loss(name='CrossEntropyLoss',**kwargs)->nn.Module:

    if name in losses_dict.keys():
        return losses_dict[name](**kwargs)
    else:
        raise NotImplementedError('name not in availables values.'.format(name))