#!/usr/bin/env python
from __future__ import print_function
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
import torchvision.transforms as transforms
import os
import argparse
import time
from models.vit import ViT_LearnPos, ViTFixedPos, ViTFixedPosDualPatchNorm, ViTFixedPosNestedDualPatchNorm
from utils import progress_bar

parser = argparse.ArgumentParser(description='Evaluate Individual ViT Model from MRL Training')
parser.add_argument('--model_path', type=str, required=True, help='path to saved ViT model checkpoint')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset to use (cifar10 or cifar100)')
parser.add_argument('--bs', default='100', type=int, help='batch size for evaluation')
parser.add_argument('--size', default='32', type=int, help='image size')
parser.add_argument('--notb', action='store_true', help='disable tensorboard')
parser.add_argument('--num_workers', default=8, type=int, help='number of data loading workers')

args = parser.parse_args()

usetb = not args.notb
if usetb:
    from torch.utils.tensorboard import SummaryWriter
    model_name = os.path.basename(args.model_path).replace('.pth', '')
    log_dir = os.path.join('runs', f'eval_{model_name}')
    writer = SummaryWriter(log_dir=log_dir)
    print(f'TensorBoard logging to: {log_dir}')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print('==> Preparing data..')
size = args.size

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

transform_test = transforms.Compose([
    transforms.Resize(size),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

testset = dataset_class(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.bs, shuffle=False, num_workers=args.num_workers)

print(f'==> Loading model from {args.model_path}')
if not os.path.exists(args.model_path):
    raise FileNotFoundError(f"Model checkpoint not found at {args.model_path}")

checkpoint = torch.load(args.model_path)

vit_class_name = checkpoint['vit_class']
if vit_class_name == 'ViTFixedPosDualPatchNorm':
    vit_class = ViTFixedPosDualPatchNorm
elif vit_class_name == 'ViTFixedPosNestedDualPatchNorm':
    vit_class = ViTFixedPosNestedDualPatchNorm
elif vit_class_name == 'ViTFixedPos':
    vit_class = ViTFixedPos
elif vit_class_name == 'ViT_LearnPos':
    vit_class = ViT_LearnPos
else:
    raise ValueError(f"Unknown ViT class: {vit_class_name}")

vit_kwargs = checkpoint['vit_kwargs'].copy()
vit_kwargs['dim'] = checkpoint['dim']

if vit_class is ViTFixedPosNestedDualPatchNorm:
    state_dict = checkpoint['model_state_dict']
    if 'nested_proj.weight' in state_dict:
        max_dim = state_dict['nested_proj.weight'].shape[0]
        vit_kwargs['max_dim'] = max_dim

net = vit_class(**vit_kwargs)
net.load_state_dict(checkpoint['model_state_dict'])
net = net.to(device)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

print(f"Model loaded: {vit_class_name} with dim={checkpoint['dim']}")
print(f"Model parameters: {sum(p.numel() for p in net.parameters()):,}")

criterion = nn.CrossEntropyLoss()

def evaluate():
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    all_targets = []
    all_predictions = []
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            all_targets.append(targets.cpu())
            all_predictions.append(predicted.cpu())
            for i in range(targets.size(0)):
                label = targets[i].item()
                class_correct[label] += (predicted[i] == targets[i]).item()
                class_total[label] += 1
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    avg_loss = test_loss / len(testloader)
    accuracy = 100. * correct / total
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    if all_targets and all_predictions:
        targets_cat = torch.cat(all_targets)
        preds_cat = torch.cat(all_predictions)
        for t, p in zip(targets_cat, preds_cat):
            confusion_matrix[t.item(), p.item()] += 1
    return avg_loss, accuracy, class_correct, class_total, confusion_matrix


print('\n==> Starting evaluation...')
start_time = time.time()
avg_loss, accuracy, class_correct, class_total, confusion_matrix = evaluate()
eval_time = time.time() - start_time

print(f'\n==> Evaluation Results:')
print(f'Average Loss: {avg_loss:.4f}')
print(f'Overall Accuracy: {accuracy:.2f}% ({int(accuracy * len(testset) / 100)}/{len(testset)})')
print(f'Evaluation Time: {eval_time:.2f}s')

print('\n==> Per-Class Accuracy:')
for i in range(num_classes):
    if class_total[i] > 0:
        class_acc = 100. * class_correct[i] / class_total[i]
        print(f'Class {i}: {class_acc:.2f}% ({class_correct[i]}/{class_total[i]})')

if usetb:
    writer.add_scalar('Evaluation/Loss', avg_loss, 0)
    writer.add_scalar('Evaluation/Accuracy', accuracy, 0)
    writer.add_scalar('Evaluation/Time', eval_time, 0)
    
    for i in range(num_classes):
        if class_total[i] > 0:
            class_acc = 100. * class_correct[i] / class_total[i]
            writer.add_scalar(f'PerClass/Class_{i}', class_acc, 0)
    
    model_info = f"""
    Model: {checkpoint['vit_class']}
    Embedding Dimension: {checkpoint['dim']}
    Dataset: {args.dataset}
    Test Samples: {len(testset)}
    Accuracy: {accuracy:.2f}%
    Loss: {avg_loss:.4f}
    Parameters: {sum(p.numel() for p in net.parameters()):,}
    """
    writer.add_text('Model_Info', model_info, 0)
    writer.close()
    print(f'\nTensorBoard logs saved to: {log_dir}')

os.makedirs('eval_results', exist_ok=True)
result_file = os.path.join('eval_results', f'{os.path.basename(args.model_path).replace(".pth", "_results.txt")}')
with open(result_file, 'w') as f:
    f.write(f'Evaluation Results for {args.model_path}\n')
    f.write(f'='*80 + '\n')
    f.write(f'Model: {checkpoint["vit_class"]}\n')
    f.write(f'Embedding Dimension: {checkpoint["dim"]}\n')
    f.write(f'Dataset: {args.dataset}\n')
    f.write(f'Test Samples: {len(testset)}\n')
    f.write(f'Overall Accuracy: {accuracy:.2f}%\n')
    f.write(f'Average Loss: {avg_loss:.4f}\n')
    f.write(f'Evaluation Time: {eval_time:.2f}s\n')
    f.write(f'Parameters: {sum(p.numel() for p in net.parameters()):,}\n')
    f.write(f'\nPer-Class Accuracy:\n')
    for i in range(num_classes):
        if class_total[i] > 0:
            class_acc = 100. * class_correct[i] / class_total[i]
            f.write(f'  Class {i}: {class_acc:.2f}% ({class_correct[i]}/{class_total[i]})\n')

confusion_file = os.path.join('eval_results', f'{os.path.basename(args.model_path).replace(".pth", "_confusion_matrix.csv")}')
np.savetxt(confusion_file, confusion_matrix, fmt='%d', delimiter=',')

print(f'\nResults saved to: {result_file}')
print(f'Confusion matrix saved to: {confusion_file}')
print('\nEvaluation complete!')
