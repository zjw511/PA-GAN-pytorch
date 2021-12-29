# Copyright (C) 2019 Elvis Yu-Jing Lin <elvisyjlin@gmail.com>
# 
# This work is licensed under the MIT License. To view a copy of this license,
# visit https://opensource.org/licenses/MIT.

"""Helper functions"""

import os
from glob import glob
import torch

def find_model(path, epoch='latest'):
    if epoch == 'latest':
        files = glob(os.path.join(path, '*.pth'))
        file = sorted(files, key=lambda x: int(x.rsplit('.', 2)[1]))[-1]
    else:
        file = os.path.join(path, 'weights.{:d}.pth'.format(int(epoch)))
    assert os.path.exists(file), 'File not found: ' + file
    print('Find model of {} epoch: {}'.format(epoch, file))
    return file

def title_concat(a, b = []):
    b_tile = b.view(b.size(0),-1,1,1).repeat(1,1,a.size(2),a.size(3))
    z = torch.cat([a, b_tile], dim=1)
    return z

import torch.nn.functional as F
def overlap_loss_fn(ms_multi, att_names):
    # ======================================
    # =        customized relation         =
    # ======================================

    full_overlap_pairs = [
        # ('Black_Hair', 'Blond_Hair'),
        # ('Black_Hair', 'Brown_Hair'),

        # ('Blond_Hair', 'Brown_Hair')
    ]

    non_overlap_pairs = [
        # ('Bald', 'Bushy_Eyebrows'),
        # ('Bald', 'Eyeglasses'),
        ('Bald', 'Mouth_Slightly_Open'),
        ('Bald', 'Mustache'),
        ('Bald', 'No_Beard'),

        ('Bangs', 'Mouth_Slightly_Open'),
        ('Bangs', 'Mustache'),
        ('Bangs', 'No_Beard'),

        ('Black_Hair', 'Mouth_Slightly_Open'),
        ('Black_Hair', 'Mustache'),
        ('Black_Hair', 'No_Beard'),

        ('Blond_Hair', 'Mouth_Slightly_Open'),
        ('Blond_Hair', 'Mustache'),
        ('Blond_Hair', 'No_Beard'),

        ('Brown_Hair', 'Mouth_Slightly_Open'),
        ('Brown_Hair', 'Mustache'),
        ('Brown_Hair', 'No_Beard'),

        # ('Bushy_Eyebrows', 'Mouth_Slightly_Open'),
        ('Bushy_Eyebrows', 'Mustache'),
        ('Bushy_Eyebrows', 'No_Beard'),

        # ('Eyeglasses', 'Mouth_Slightly_Open'),
        ('Eyeglasses', 'Mustache'),
        ('Eyeglasses', 'No_Beard'),
    ]

    # ======================================
    # =                 losses             =
    # ======================================

    full_overlap_pair_loss = torch.zeros(1,device=ms_multi[-1].device) 
    for p in full_overlap_pairs:
        id1 = att_names.index(p[0])
        id2 = att_names.index(p[1])
        for m in ms_multi[-1:]:
            m1 = m[:, id1]
            m2 = m[:, id2]
            full_overlap_pair_loss += F.l1_loss(m1, m2)

    non_overlap_pair_loss = torch.zeros(1,device=ms_multi[-1].device) 
    for p in non_overlap_pairs:
        id1 = att_names.index(p[0])
        id2 = att_names.index(p[1])
        for m in ms_multi[-1:]:
            m1 = m[:, id1]
            m2 = m[:, id2]
            non_overlap_pair_loss += torch.mean(torch.sigmoid(m1) * torch.sigmoid(m2))

    return full_overlap_pair_loss, non_overlap_pair_loss
