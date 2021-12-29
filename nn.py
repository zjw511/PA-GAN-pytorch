# Copyright (C) 2018 Elvis Yu-Jing Lin <elvisyjlin@gmail.com>
# 
# This work is licensed under the MIT License. To view a copy of this license,
# visit https://opensource.org/licenses/MIT.

"""Network components."""

import torch
import torch.nn as nn
from switchable_norm import SwitchNorm1d, SwitchNorm2d
import torch.nn.functional as F

def add_normalization_1d(layers, fn, n_out):
    if fn == 'none':
        pass
    elif fn == 'batchnorm':
        layers.append(nn.BatchNorm1d(n_out))
    elif fn == 'instancenorm':
        layers.append(Unsqueeze(-1))
        layers.append(nn.InstanceNorm1d(n_out, affine=True))
        layers.append(Squeeze(-1))
    elif fn == 'switchnorm':
        layers.append(SwitchNorm1d(n_out))
    else:
        raise Exception('Unsupported normalization: ' + str(fn))
    return layers

def add_normalization_2d(layers, fn, n_out):
    if fn == 'none':
        pass
    elif fn == 'batchnorm':
        layers.append(nn.BatchNorm2d(n_out))
    elif fn == 'instancenorm':
        layers.append(nn.InstanceNorm2d(n_out, affine=True))
    elif fn == 'switchnorm':
        layers.append(SwitchNorm2d(n_out))
    else:
        raise Exception('Unsupported normalization: ' + str(fn))
    return layers

def add_activation(layers, fn):
    if fn == 'none':
        pass
    elif fn == 'relu':
        layers.append(nn.ReLU())
    elif fn == 'lrelu':
        layers.append(nn.LeakyReLU())
    elif fn == 'sigmoid':
        layers.append(nn.Sigmoid())
    elif fn == 'tanh':
        layers.append(nn.Tanh())
    else:
        raise Exception('Unsupported activation function: ' + str(fn))
    return layers

class Squeeze(nn.Module):
    def __init__(self, dim):
        super(Squeeze, self).__init__()
        self.dim = dim
    
    def forward(self, x):
        return x.squeeze(self.dim)

class Unsqueeze(nn.Module):
    def __init__(self, dim):
        super(Unsqueeze, self).__init__()
        self.dim = dim
    
    def forward(self, x):
        return x.unsqueeze(self.dim)


class LinearBlock(nn.Module):
    def __init__(self, n_in, n_out, norm_fn='none', acti_fn='none'):
        super(LinearBlock, self).__init__()
        layers = [nn.Linear(n_in, n_out, bias=(norm_fn=='none'))]
        layers = add_normalization_1d(layers, norm_fn, n_out)
        layers = add_activation(layers, acti_fn)
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)

class Conv2dBlock(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, stride=1, padding=0, 
                 norm_fn=None, acti_fn=None):
        super(Conv2dBlock, self).__init__()
        layers = [nn.Conv2d(n_in, n_out, kernel_size, stride=stride, padding=padding, bias=(norm_fn=='none'))]
        layers = add_normalization_2d(layers, norm_fn, n_out)
        layers = add_activation(layers, acti_fn)
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)

class ConvTranspose2dBlock(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, stride=1, padding=0, 
                 norm_fn=False, acti_fn=None):
        super(ConvTranspose2dBlock, self).__init__()
        layers = [nn.ConvTranspose2d(n_in, n_out, kernel_size, stride=stride, padding=padding, bias=(norm_fn=='none'))]
        layers = add_normalization_2d(layers, norm_fn, n_out)
        layers = add_activation(layers, acti_fn)
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)
import utils

class AttentionEditor(nn.Module):
    def __init__(self,indim,outdim,n_att=13,norm_fn=False, acti_fn=None) -> None:
        super(AttentionEditor,self).__init__()
        self.Ge = nn.Sequential(ConvTranspose2dBlock(
                    indim + n_att, outdim, (3, 3), stride=1, padding=1, norm_fn=norm_fn, acti_fn=acti_fn
                ),ConvTranspose2dBlock(
                    outdim, outdim, (3, 3), stride=2, padding=0, norm_fn=norm_fn, acti_fn=acti_fn
                ))
        # self.Ge = nn.ModuleList(Ge)
        Gm = [Conv2dBlock(
                    outdim+ outdim + 2* n_att , outdim, (1, 1), stride=1, padding=0, norm_fn=norm_fn, acti_fn=acti_fn
                ),
                Conv2dBlock(
                    outdim+ outdim +2* n_att , outdim, (3, 3), stride=1, padding=1, norm_fn=norm_fn, acti_fn=acti_fn
                ),
                nn.Sequential(Conv2dBlock(
                    outdim+ outdim +2* n_att , outdim, (3, 3), stride=1, padding=1, norm_fn=norm_fn, acti_fn=acti_fn
                ),Conv2dBlock(
                    outdim, outdim, (3, 3), stride=1, padding=1, norm_fn=norm_fn, acti_fn=acti_fn
                )),
                nn.Sequential(Conv2dBlock(
                    outdim+ outdim +2* n_att , outdim, (3, 3), stride=1, padding=1, norm_fn=norm_fn, acti_fn=acti_fn
                ),Conv2dBlock(
                    outdim, outdim, (3, 3), stride=1, padding=1, norm_fn=norm_fn, acti_fn=acti_fn
                ),Conv2dBlock(
                    outdim, outdim, (3, 3), stride=1, padding=1, norm_fn=norm_fn, acti_fn=acti_fn
                ))]
        self.Gm = nn.ModuleList(Gm)
        self.Gm2 = nn.Sequential(Conv2dBlock(
                    outdim*4, outdim*2, (4,4), stride=2, padding=2, norm_fn=norm_fn, acti_fn=acti_fn
                ),nn.ConvTranspose2d(outdim*2, n_att, (4,4), stride=2, padding=2, bias=(norm_fn=='none'))
                )
        self.n_att = n_att
    def forward(self,fa,fb,b,m_multi_pre=None):
        e_ipt = utils.title_concat(fb,b)
        e = self.Ge(e_ipt)[:,:,:-1,:-1]
        if m_multi_pre is not None:

            # shape = [None ,m_multi_pre.shape[1] * 2, m_multi_pre.shape[2] * 2, m_multi_pre.shape[3]]
            m_multi_pre = F.upsample(m_multi_pre,scale_factor=2,mode='bicubic')
        dm_multi_ipt = utils.title_concat(torch.cat([fa,e,m_multi_pre],dim=1),b) # ? i not know what to do
        ms = []
        for layer in self.Gm:
            m_ = layer(dm_multi_ipt)
            ms.append(m_)
        ms = torch.cat(ms,dim=1)
        dm_multi = self.Gm2(ms)
        if m_multi_pre is not None:
            m_multi = m_multi_pre + dm_multi
        else:
            m_multi = dm_multi
        # b = tf.reshape(tf.abs(tf.sign(b)), [-1, 1, 1, n_att])
        # m = tf.clip_by_value(tf.reduce_sum(b * tf.nn.sigmoid(m_multi), axis=-1, keep_dims=True), 0.0, 1.0)
        b = torch.abs(torch.sign(b)).view(-1,self.n_att,1,1)
        m = torch.clip(torch.sum(b*torch.sigmoid(m_multi),dim=1,keepdim=True),0.0,1.0)
        fb = m * e + (1 - m) * fa

        return fb, e, m, m_multi
class BasicAttentionEditor(nn.Module):
    def __init__(self,indim,outdim,n_att=13,norm_fn=False, acti_fn=None) -> None:
        super(BasicAttentionEditor,self).__init__()
        self.Ge = nn.Sequential(ConvTranspose2dBlock(
                    indim+n_att, outdim, (3, 3), stride=1, padding=1, norm_fn=norm_fn, acti_fn=acti_fn
                ),nn.ConvTranspose2d(
                    outdim, 3, (3, 3), stride=2, padding=0
                ),
                nn.Tanh())
        # self.Ge = nn.ModuleList(Ge)
        Gm = [Conv2dBlock(
                    outdim+ 3 +2* n_att , outdim, (1, 1), stride=1, padding=0, norm_fn=norm_fn, acti_fn=acti_fn
                ),
                Conv2dBlock(
                    outdim+ 3 +2* n_att, outdim, (3, 3), stride=1, padding=1, norm_fn=norm_fn, acti_fn=acti_fn
                ),
                nn.Sequential(Conv2dBlock(
                    outdim+ 3 +2* n_att, outdim, (3, 3), stride=1, padding=1, norm_fn=norm_fn, acti_fn=acti_fn
                ),Conv2dBlock(
                    outdim, outdim, (3, 3), stride=1, padding=1, norm_fn=norm_fn, acti_fn=acti_fn
                )),
                nn.Sequential(Conv2dBlock(
                    outdim+ 3 +2* n_att, outdim, (3, 3), stride=1, padding=1, norm_fn=norm_fn, acti_fn=acti_fn
                ),Conv2dBlock(
                    outdim, outdim, (3, 3), stride=1, padding=1, norm_fn=norm_fn, acti_fn=acti_fn
                ),Conv2dBlock(
                    outdim, outdim, (3, 3), stride=1, padding=1, norm_fn=norm_fn, acti_fn=acti_fn
                ))]
        self.Gm = nn.ModuleList(Gm)
        self.Gm2 = nn.Sequential(Conv2dBlock(
                    outdim*4, outdim*2, (4,4), stride=2, padding=2, norm_fn=norm_fn, acti_fn=acti_fn
                ),nn.ConvTranspose2d(outdim*2, n_att, (4,4), stride=2, padding=2, bias=(norm_fn=='none'))
                )
        self.n_att = n_att
    def forward(self,fa,fb,b,m_multi_pre=None):
        e_ipt = utils.title_concat(fb,b)
        e = self.Ge(e_ipt)[:,:,:-1,:-1]
        if m_multi_pre is not None:

            # shape = [None ,m_multi_pre.shape[1] * 2, m_multi_pre.shape[2] * 2, m_multi_pre.shape[3]]
            m_multi_pre = F.upsample(m_multi_pre,scale_factor=2,mode='bicubic')
        dm_multi_ipt = utils.title_concat(torch.cat([fa,e,m_multi_pre],dim=1),b) # ? i not know what to do
        ms = []
        for layer in self.Gm:
            m_ = layer(dm_multi_ipt)
            ms.append(m_)
        ms = torch.cat(ms,dim=1)
        dm_multi = self.Gm2(ms)
        if m_multi_pre is not None:
            m_multi = m_multi_pre + dm_multi
        else:
            m_multi = dm_multi
        # b = tf.reshape(tf.abs(tf.sign(b)), [-1, 1, 1, n_att])
        # m = tf.clip_by_value(tf.reduce_sum(b * tf.nn.sigmoid(m_multi), axis=-1, keep_dims=True), 0.0, 1.0)
        b = torch.abs(torch.sign(b)).view(-1,self.n_att,1,1)
        m = torch.clip(torch.sum(b*torch.sigmoid(m_multi),dim=1,keepdim=True),0.0,1.0)
        fb = m * e + (1 - m) * fa

        return fb, e, m, m_multi

class FAttentionEditor(nn.Module):
    def __init__(self,indim,outdim,n_att=13,norm_fn=False, acti_fn=None) -> None:
        super(FAttentionEditor,self).__init__()
        self.Ge = nn.Sequential(ConvTranspose2dBlock(
                    indim + n_att, outdim, (3, 3), stride=1, padding=1, norm_fn=norm_fn, acti_fn=acti_fn
                ),ConvTranspose2dBlock(
                    outdim, outdim, (3, 3), stride=2, padding=0, norm_fn=norm_fn, acti_fn=acti_fn
                ))
        # self.Ge = nn.ModuleList(Ge)
        Gm = [Conv2dBlock(
                   2*outdim + n_att, outdim, (1, 1), stride=1, padding=0, norm_fn=norm_fn, acti_fn=acti_fn
                ),
                Conv2dBlock(
                    2*outdim+ n_att , outdim, (3, 3), stride=1, padding=1, norm_fn=norm_fn, acti_fn=acti_fn
                ),
                nn.Sequential(Conv2dBlock(
                    2*outdim+ n_att, outdim, (3, 3), stride=1, padding=1, norm_fn=norm_fn, acti_fn=acti_fn
                ),Conv2dBlock(
                    outdim, outdim, (3, 3), stride=1, padding=1, norm_fn=norm_fn, acti_fn=acti_fn
                )),
                nn.Sequential(Conv2dBlock(
                    2*outdim+ n_att, outdim, (3, 3), stride=1, padding=1, norm_fn=norm_fn, acti_fn=acti_fn
                ),Conv2dBlock(
                    outdim, outdim, (3, 3), stride=1, padding=1, norm_fn=norm_fn, acti_fn=acti_fn
                ),Conv2dBlock(
                    outdim, outdim, (3, 3), stride=1, padding=1, norm_fn=norm_fn, acti_fn=acti_fn
                ))]
        self.Gm = nn.ModuleList(Gm)
        self.Gm2 = nn.Sequential(Conv2dBlock(
                    outdim*4, outdim*2, (4,4), stride=2, padding=2, norm_fn=norm_fn, acti_fn=acti_fn
                ),nn.ConvTranspose2d(outdim*2, n_att, (4,4), stride=2, padding=2, bias=(norm_fn=='none'))
                )
        self.n_att = n_att
    def forward(self,fa,fb,b,m_multi_pre=None):
        e_ipt = utils.title_concat(fb,b)
        e = self.Ge(e_ipt)[:,:,:-1,:-1]
        dm_multi_ipt = utils.title_concat(torch.cat([fa,e],dim=1),b) # ? i not know what to do
        ms = []
        for layer in self.Gm:
            # print(layer)
            # print(dm_multi_ipt.shape)
            m_ = layer(dm_multi_ipt)
            ms.append(m_)
        ms = torch.cat(ms,dim=1)
        dm_multi = self.Gm2(ms)
        m_multi = dm_multi
        # b = tf.reshape(tf.abs(tf.sign(b)), [-1, 1, 1, n_att])
        # m = tf.clip_by_value(tf.reduce_sum(b * tf.nn.sigmoid(m_multi), axis=-1, keep_dims=True), 0.0, 1.0)
        b = torch.abs(torch.sign(b)).view(-1,self.n_att,1,1)
        # print('b',b.shape)
        # print('m_multi',m_multi.shape)
        m = torch.clip(torch.sum(b*torch.sigmoid(m_multi),dim=1,keepdim=True),0.0,1.0)
        # print('m',m.shape)
        fb = m * e + (1 - m) * fa

        return fb, e, m, m_multi