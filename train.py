# Copyright (C) 2018 Elvis Yu-Jing Lin <elvisyjlin@gmail.com>
# 
# This work is licensed under the MIT License. To view a copy of this license,
# visit https://opensource.org/licenses/MIT.

"""Main entry point for training paGAN network."""

import argparse
import datetime
import json
import os
from os.path import join
from torch.nn.functional import mse_loss

import torch.utils.data as data
from torchvision import transforms

import torch
import torchvision.utils as vutils
from pagan import PAGAN
from data import check_attribute_conflict
from helpers import Progressbar, add_scalar_dict
from tensorboardX import SummaryWriter
import wandb

attrs_default = [
    'Bald', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Bushy_Eyebrows',
    'Eyeglasses', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'No_Beard', 'Pale_Skin', 'Young'
]

def parse(args=None):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--attrs', dest='attrs', default=attrs_default, nargs='+', help='attributes to learn')
    parser.add_argument('--data', dest='data', type=str, choices=['CelebA', 'CelebA-HQ'], default='CelebA')
    parser.add_argument('--data_path', dest='data_path', type=str, default='/home/lilipan/桌面/AAAI22/data/celeba/img_align_celeba')
    parser.add_argument('--attr_path', dest='attr_path', type=str, default='/home/lilipan/桌面/AAAI22/data/celeba/list_attr_celeba.txt')
    parser.add_argument('--image_list_path', dest='image_list_path', type=str, default='data/image_list.txt')
    parser.add_argument('--load_size', dest='load_size', type=int, default=143)
    parser.add_argument('--img_size', dest='img_size', type=int, default=128)
    # parser.add_argument('--shortcut_layers', dest='shortcut_layers', type=int, default=1)
    # parser.add_argument('--inject_layers', dest='inject_layers', type=int, default=0)
    parser.add_argument('--enc_dim', dest='enc_dim', type=int, default=64)
    parser.add_argument('--dec_dim', dest='dec_dim', type=int, default=64)
    parser.add_argument('--dis_dim', dest='dis_dim', type=int, default=64)
    parser.add_argument('--dis_fc_dim', dest='dis_fc_dim', type=int, default=1024)
    parser.add_argument('--enc_layers', dest='enc_layers', type=int, default=5)
    parser.add_argument('--dec_layers', dest='dec_layers', type=int, default=5)
    parser.add_argument('--dis_layers', dest='dis_layers', type=int, default=5)
    parser.add_argument('--enc_norm', dest='enc_norm', type=str, default='batchnorm')
    parser.add_argument('--dec_norm', dest='dec_norm', type=str, default='batchnorm')
    parser.add_argument('--dis_norm', dest='dis_norm', type=str, default='layernorm')
    parser.add_argument('--dis_fc_norm', dest='dis_fc_norm', type=str, default='none')
    parser.add_argument('--enc_acti', dest='enc_acti', type=str, default='relu')
    parser.add_argument('--dec_acti', dest='dec_acti', type=str, default='relu')
    parser.add_argument('--dis_acti', dest='dis_acti', type=str, default='lrelu')
    parser.add_argument('--dis_fc_acti', dest='dis_fc_acti', type=str, default='lrelu')
    parser.add_argument('--lambda_1', dest='lambda_1', type=float, default=100.0)
    parser.add_argument('--lambda_2', dest='lambda_2', type=float, default=10.0)
    parser.add_argument('--lambda_3', dest='lambda_3', type=float, default=1.0)
    parser.add_argument('--lambda_gp', dest='lambda_gp', type=float, default=10.0)
    parser.add_argument('--g_spasity_loss_weight', dest='g_spasity_loss_weight', type=float, default=0.05)
    parser.add_argument('--g_attribute_loss_weight', dest='g_attribute_loss_weight', type=float, default=20.0)
    parser.add_argument('--g_full_overlap_mask_pair_loss_weight', dest='g_full_overlap_mask_pair_loss_weight', type=float, default=1.0)
    parser.add_argument('--g_non_overlap_mask_pair_loss_weight', dest='g_non_overlap_mask_pair_loss_weight', type=float, default=1.0)


    parser.add_argument('--mode', dest='mode', default='wgan', choices=['wgan', 'lsgan', 'dcgan'])
    parser.add_argument('--epochs', dest='epochs', type=int, default=200, help='# of epochs')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=32)
    parser.add_argument('--num_workers', dest='num_workers', type=int, default=4)
    parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--beta1', dest='beta1', type=float, default=0.5)
    parser.add_argument('--beta2', dest='beta2', type=float, default=0.999)
    parser.add_argument('--n_d', dest='n_d', type=int, default=5, help='# of d updates per g update')
    
    parser.add_argument('--b_distribution', dest='b_distribution', default='none', choices=['none', 'uniform', 'truncated_normal'])
    parser.add_argument('--thres_int', dest='thres_int', type=float, default=1.0)
    parser.add_argument('--test_int', dest='test_int', type=float, default=1.5)
    parser.add_argument('--n_samples', dest='n_samples', type=int, default=16, help='# of sample images')
    
    parser.add_argument('--save_interval', dest='save_interval', type=int, default=1000)
    parser.add_argument('--sample_interval', dest='sample_interval', type=int, default=600)
    parser.add_argument('--gpu', dest='gpu', action='store_true')
    parser.add_argument('--multi_gpu', dest='multi_gpu', action='store_true')
    parser.add_argument('--experiment_name', dest='experiment_name', default=datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y"))
    
    return parser.parse_args(args)

args = parse()
print(args)

args.lr_base = args.lr
args.n_attrs = len(args.attrs)
args.betas = (args.beta1, args.beta2)

os.makedirs(join('output', args.experiment_name), exist_ok=True)
os.makedirs(join('output', args.experiment_name, 'checkpoint'), exist_ok=True)
os.makedirs(join('output', args.experiment_name, 'sample_training'), exist_ok=True)
with open(join('output', args.experiment_name, 'setting.txt'), 'w') as f:
    f.write(json.dumps(vars(args), indent=4, separators=(',', ':')))

if args.data == 'CelebA':
    from data import CelebA
    train_dataset = CelebA(args.data_path, args.attr_path, args.load_size ,args.img_size, 'train', args.attrs)
    valid_dataset = CelebA(args.data_path, args.attr_path,  args.load_size ,args.img_size, 'valid', args.attrs)
if args.data == 'CelebA-HQ':
    from data import CelebA_HQ
    train_dataset = CelebA_HQ(args.data_path, args.attr_path, args.image_list_path, args.img_size, 'train', args.attrs)
    valid_dataset = CelebA_HQ(args.data_path, args.attr_path, args.image_list_path, args.img_size, 'valid', args.attrs)
train_dataloader = data.DataLoader(
    train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
    shuffle=True, drop_last=True
)
valid_dataloader = data.DataLoader(
    valid_dataset, batch_size=args.n_samples, num_workers=args.num_workers,
    shuffle=False, drop_last=False
)
print('Training images:', len(train_dataset), '/', 'Validating images:', len(valid_dataset))

pagan = PAGAN(args)
progressbar = Progressbar()
writer = SummaryWriter(join('output', args.experiment_name, 'summary'))

fixed_img_a, fixed_att_a = next(iter(valid_dataloader))
fixed_img_a = fixed_img_a.cuda() if args.gpu else fixed_img_a
fixed_att_a = fixed_att_a.cuda() if args.gpu else fixed_att_a
fixed_att_a = fixed_att_a.type(torch.float)
sample_att_b_list = [fixed_att_a]
for i in range(args.n_attrs):
    tmp = fixed_att_a.clone()
    tmp[:, i] = 1 - tmp[:, i]
    tmp = check_attribute_conflict(tmp, args.attrs[i], args.attrs)
    sample_att_b_list.append(tmp)
# wandb.init(project='PAGAN')
it = 0
it_per_epoch = len(train_dataset) // args.batch_size
for epoch in range(args.epochs):
    # train with base lr in the first 100 epochs
    # and half the lr in the last 100 epochs
    lr = args.lr_base / (10 ** (epoch // 100))
    pagan.set_lr(lr)
    writer.add_scalar('LR/learning_rate', lr, it+1)
    for img_a, att_a in progressbar(train_dataloader):
        pagan.train()
        
        img_a = img_a.cuda() if args.gpu else img_a
        att_a = att_a.cuda() if args.gpu else att_a
        idx = torch.randperm(len(att_a))
        att_b = att_a[idx].contiguous()
        
        att_a = att_a.type(torch.float)
        att_b = att_b.type(torch.float)
        
        att_a_ = (att_a * 2 - 1) * args.thres_int
        if args.b_distribution == 'none':
            att_b_ = (att_b * 2 - 1) * args.thres_int
        if args.b_distribution == 'uniform':
            att_b_ = (att_b * 2 - 1) * \
                     torch.rand_like(att_b) * \
                     (2 * args.thres_int)
        if args.b_distribution == 'truncated_normal':
            att_b_ = (att_b * 2 - 1) * \
                     (torch.fmod(torch.randn_like(att_b), 2) + 2) / 4.0 * \
                     (2 * args.thres_int)
        
        if (it+1) % (args.n_d+1) != 0:
            errD = pagan.trainD(img_a, att_a, att_a_, att_b, att_b_)
            add_scalar_dict(writer, errD, it+1, 'D')
            # wandb.log(errD,step=it)
        else:
            errG = pagan.trainG(img_a, att_a, att_a_, att_b, att_b_)
            add_scalar_dict(writer, errG, it+1, 'G')
            progressbar.say(epoch=epoch, iter=it+1, d_loss=errD['d_loss'], g_loss=errG['g_loss'])
            # wandb.log(errG,step=it)
        if (it+1) % args.save_interval == 0:
            # To save storage space, I only checkpoint the weights of G.
            # If you'd like to keep weights of G, D, optim_G, optim_D,
            # please use save() instead of saveG().
            pagan.saveG(os.path.join(
                'output', args.experiment_name, 'checkpoint', 'weights.{:d}.pth'.format(epoch)
            ))
            # pagan.save(os.path.join(
            #     'output', args.experiment_name, 'checkpoint', 'weights.{:d}.pth'.format(epoch)
            # ))
        if (it+1) % args.sample_interval == 0:
            pagan.eval()
            with torch.no_grad():
                samples = [fixed_img_a]
                # print(fixed_img_a.sha/pe)
                e_opt_list = [torch.full_like(fixed_img_a, -1.0)]
                ms_opt_list = []
                for i, att_b in enumerate(sample_att_b_list):
                    att_b_ = (att_b * 2 - 1) 
                    if i > 0 :
                        att_b_[...,i-1] = att_b_[...,i-1] * args.test_int
                    # samples.append(pagan.G(fixed_img_a, att_b_-))
                    att_a = fixed_att_a*2-1
                    sample,sample_e,sample_mask,_ = pagan.G(fixed_img_a, att_b_-att_a)
                    samples.append(sample)
                    e_opt_list.append(sample_e)
                    ms_opt_list.append(sample_mask)
                    # print('ms',sample_mask[0].shape)
                e_opt_list = torch.cat(e_opt_list,dim=3)
                
                e_opt_list = e_opt_list.view(1,-1,e_opt_list.shape[1],e_opt_list.shape[2], e_opt_list.shape[3])
                # print(e_opt_list.shape)
                samples = torch.cat(samples, dim=3)
                # print('sample',samples.shape)
                samples_copy = samples.clone()
                samples_copy = samples_copy.view(1,-1,samples.shape[1],samples.shape[2],samples.shape[3])
                resize = transforms.Resize([128,128])
                for k,ms_opt in enumerate(ms_opt_list):
                    for i, m_opt in enumerate(ms_opt):
                        m_opt_resize = []
                        for m_j_opt in m_opt: 
                            # print(m_j_opt.shape)
                            m_opt_resize.append(resize(m_j_opt*2-1).view(1,1,128,128).expand(1,3,128,128))
                        # print(i,m_opt_resize[0].shape,len(m_opt_resize))
                        ms_opt[i] = torch.cat(m_opt_resize,dim=0)
                    ms_opt_list[k] = torch.cat(ms_opt,dim=0)
                ms_opt_list = [torch.full_like(ms_opt_list[0], -1.0)] + ms_opt_list
                ms_opt_list = torch.cat(ms_opt_list,dim=3)
                # print(ms_opt_list.shape)
                ms_opt_list = ms_opt_list.view(4,-1,ms_opt_list.shape[1],ms_opt_list.shape[2],ms_opt_list.shape[3])
                # print('samples_copy',samples_copy.shape)
                # print('e_opt_list',e_opt_list.shape)
                # print('ms_opt_list',ms_opt_list.shape)
                sample_m = torch.cat([samples_copy,e_opt_list,ms_opt_list],dim=0)
                sample_m = sample_m.view(-1,sample_m.shape[2],sample_m.shape[3],sample_m.shape[4])
                sample_ms = sample_m.clone()
                # sample_m = torch.zeros(())
                for i in range(16):
                    for k in range(6):
                        sample_m[i*6+k,:] = sample_ms[k*16+i,:]
                    # sample_m[(i-1)*6+1,:] = sample_ms[2,i:]
                # sample_m = sample_m.view(-1,6,sample_m.shape[2],sample_m.shape[3],sample_m.shape[4])
                # sample_m = sample_m.reshape(-1,sample_m.shape[2],sample_m.shape[3],sample_m.shape[4])

                
                writer.add_image('sample', vutils.make_grid(torch.cat([samples,sample_m],dim=0), nrow=1, normalize=True, range=(-1., 1.)), it+1)
                vutils.save_image(torch.cat([samples,sample_m],dim=0), os.path.join(
                        'output', args.experiment_name, 'sample_training',
                        'Epoch_({:d})_({:d}of{:d}).jpg'.format(epoch, it%it_per_epoch+1, it_per_epoch)
                    ), nrow=1, normalize=True, range=(-1., 1.))
        it += 1
# wandb.finish()