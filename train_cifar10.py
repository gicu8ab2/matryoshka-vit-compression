#! /usr/bin/env python

from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import pandas as pd
import csv
import time

from models import *
from utils import progress_bar
from randomaug import RandAugment
from models.vit import ViT_LearnPos, ViTFixedPos, ViTFixedPosDualPatchNorm, ViTFixedPosVqVae

from MRL import Matryoshka_CE_Loss, MRL_Entire_Transformer, MRL_Entire_Transformer_NestedProj

# parsers
parser = argparse.ArgumentParser(description='Matryoshka-ViT CIFAR10/100 Training')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate') # resnets.. 1e-3, Vit..1e-4
parser.add_argument('--opt', default="adam")
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--noaug', action='store_false', help='disable use randomaug')
parser.add_argument('--noamp', action='store_true', help='disable mixed precision training. for older pytorch versions')
parser.add_argument('--notb', action='store_true', help='disable tensorboard')
parser.add_argument('--mixup', action='store_true', help='add mixup augumentations')
parser.add_argument('--net', default='vitfixedposdualpatchnorm')
parser.add_argument('--mrl_flag', action='store_true', help='use mrl')
parser.add_argument('--mrl_dims', default='8,16,32,64,128,256,512', type=str, help='comma-separated list of embedding dimensions for MRL')
parser.add_argument('--nested_proj_mrl', action='store_true', help='use nested projection MRL variant')
parser.add_argument('--dp', action='store_true', help='use data parallel')
parser.add_argument('--bs', default='512')
parser.add_argument('--size', default="32")
parser.add_argument('--n_epochs', type=int, default='100')
parser.add_argument('--patch', default='4', type=int, help="patch for ViT")
parser.add_argument('--embedding_dim', default="512", type=int)
parser.add_argument('--convkernel', default='8', type=int, help="parameter for convmixer")
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset to use (cifar10 or cifar100)')

args = parser.parse_args()

# take in args
usetb = not args.notb
if usetb:
    from torch.utils.tensorboard import SummaryWriter
    import os
    watermark = "{}_lr{}_{}" .format(args.net, args.lr, args.dataset)
    log_dir = os.path.join('runs', watermark)
    writer = SummaryWriter(log_dir=log_dir)

bs = int(args.bs)
imsize = int(args.size)

use_amp = not args.noamp
aug = args.noaug

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
if args.net=="vit_timm":
    size = 384
else:
    size = imsize

# Set up normalization based on the dataset
if args.dataset == 'cifar10':
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    num_classes = 10
    dataset_class = torchvision.datasets.CIFAR10
elif args.dataset == 'cifar100':
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)
    num_classes = 100
    dataset_class = torchvision.datasets.CIFAR100
else:
    raise ValueError("Dataset must be either 'cifar10' or 'cifar100'")

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.Resize(size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

transform_test = transforms.Compose([
    transforms.Resize(size),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

# Add RandAugment with N, M(hyperparameter)
if aug:  
    N = 2; M = 14;
    transform_train.transforms.insert(0, RandAugment(N, M))

# Prepare dataset
trainset = dataset_class(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=8)

testset = dataset_class(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)

# Set up class names based on the dataset
if args.dataset == 'cifar10':
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
else:
    # CIFAR100 has 100 classes, so we don't list them all here
    classes = None

# Model factory..
print('==> Building model..')
if args.net=='vitfixedposdualpatchnorm':
    if args.mrl_flag:
        # Parse MRL dimensions
        nesting_list = [int(d) for d in args.mrl_dims.split(',')]
        print(f'Using MRL with dimensions: {nesting_list}')
        if args.nested_proj_mrl:
            print('Using nested-projection MRL variant')
            net = MRL_Entire_Transformer_NestedProj(
                nesting_list=nesting_list,
                image_size=size,
                patch_size=args.patch,
                num_classes=num_classes,
                depth=6,
                heads=8,
                pool='cls',
                channels=3,
                dropout=0.1,
                emb_dropout=0.1,
            )
        else:
            # Define ViT kwargs (without 'dim' - it will be set by nesting_list)
            vit_kwargs = {
                'image_size': size,
                'patch_size': args.patch,
                'num_classes': num_classes,
                'depth': 6,
                'heads': 8,
                'dropout': 0.1,
                'emb_dropout': 0.1
            }
            # Create standard MRL model
            net = MRL_Entire_Transformer(
                nesting_list=nesting_list,
                vit_class=ViTFixedPosDualPatchNorm,
                vit_kwargs=vit_kwargs
            )
    else:
        net = ViTFixedPosDualPatchNorm(
        image_size = size,
        patch_size = args.patch,
        num_classes = num_classes,
        dim = int(args.embedding_dim),
        depth = 6,
        heads = 8,
        dropout = 0.1,
        emb_dropout = 0.1
    )
elif args.net=="vitfixedposvqvae":
    vq_config = {
        "embedding_dim": args.embedding_dim,
        "use_patches": True,
        "patch_size": args.patch,
        "batch_size": 512,
        "num_training_updates": 25000,
    }
    net = ViTFixedPosVqVae(
        image_size = size,
        patch_size = args.patch,
        num_classes = num_classes,
        dim = int(args.embedding_dim),
        depth = 6,
        heads = 8,
        dropout = 0.1,
        emb_dropout = 0.1,
        vqvae_config_kwargs=vq_config,
    )
elif args.net=="vitfixedpos":
    net = ViTFixedPos(
    image_size = size,
    patch_size = args.patch,
    num_classes = num_classes,
    dim = int(args.embedding_dim),
    depth = 6,
    heads = 8,
    dropout = 0.1,
    emb_dropout = 0.1
)  
elif args.net=="vitlearnedpos":
    net = ViT_LearnPos(
    image_size = size,
    patch_size = args.patch,
    num_classes = num_classes,
    dim = int(args.embedding_dim),
    depth = 6,
    heads = 8,
    dropout = 0.1,
    emb_dropout = 0.1
)  
else:
    raise ValueError(f"'{args.net}' is not a valid model")

# For Multi-GPU
if 'cuda' in device:
    print(device)
    if args.dp:
        print("using data parallel")
        net = torch.nn.DataParallel(net) # make parallel
        cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint_path = './checkpoint/{}-{}-{}-ckpt.t7'.format(args.net, args.dataset, args.patch)
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

# Loss criterion
if args.mrl_flag:
    nesting_list = [int(d) for d in args.mrl_dims.split(',')]
    relative_importance = [1.0] * len(nesting_list)
    criterion = Matryoshka_CE_Loss(relative_importance=relative_importance)
else:
    criterion = nn.CrossEntropyLoss()

if args.opt == "adam":
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
elif args.opt == "sgd":
    optimizer = optim.SGD(net.parameters(), lr=args.lr)  

    
# use cosine scheduling
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs)

##### Training
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        # Train with amp
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = net(inputs)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        train_loss += loss.item()
        # For MRL, use the last (largest) model's predictions for accuracy
        if args.mrl_flag:
            _, predicted = outputs[-1].max(1)
        else:
            _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return train_loss/(batch_idx+1)

##### Validation
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            # For MRL, use the last (largest) model's predictions for accuracy
            if args.mrl_flag:
                _, predicted = outputs[-1].max(1)
            else:
                _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            "net": net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "acc": acc,
            "epoch": epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/{}-{}-{}-ckpt.t7'.format(args.net, args.dataset, args.patch))
        best_acc = acc
    
    os.makedirs("log", exist_ok=True)
    content = time.ctime() + ' ' + f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, val loss: {test_loss:.5f}, acc: {(acc):.5f}'
    print(content)
    log_file = f'log/log_{args.net}_{args.dataset}_patch{args.patch}.txt'
    with open(log_file, 'a') as appender:
        appender.write(content + "\n")
    return test_loss, acc

list_loss = []
list_acc = []

# TensorBoard setup is already done above
    
net.cuda()
for epoch in range(start_epoch, args.n_epochs):
    start = time.time()
    trainloss = train(epoch)
    val_loss, acc = test(epoch)
    
    scheduler.step(epoch-1) # step cosine scheduling
    
    list_loss.append(val_loss)
    list_acc.append(acc)
    
    # Log training..
    if usetb:
        writer.add_scalar('Loss/Train', trainloss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Accuracy/Validation', acc, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]["lr"], epoch)
        writer.add_scalar('Epoch_Time', time.time()-start, epoch)

    # Write out csv..
    csv_file = f'log/log_{args.net}_{args.dataset}_patch{args.patch}.csv'
    with open(csv_file, 'w') as f:
        csv_writer = csv.writer(f, lineterminator='\n')
        csv_writer.writerow(list_loss) 
        csv_writer.writerow(list_acc) 
    print(list_loss)

# Save individual nested models if using MRL
if args.mrl_flag:
    print('\n==> Saving individual nested ViT models...')
    os.makedirs('checkpoint/nested_models', exist_ok=True)
    nesting_list = [int(d) for d in args.mrl_dims.split(',')]
    
    for i, dim in enumerate(nesting_list):
        save_path = f'checkpoint/nested_models/{args.net}_{args.dataset}_patch{args.patch}_dim{dim}.pth'
        net.save_vit_model(i, save_path)
    
    print(f'All {len(nesting_list)} nested models saved to checkpoint/nested_models/')

# Close TensorBoard writer
if usetb:
    writer.close()