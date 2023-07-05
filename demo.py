import os
import time
import torch
import argparse
from utils import eigen, trace, accuracy, eigenvalue, eigenvalue_dataloader
from model.vgg import vgg11_bn
from model.ResNet import ResNet18
from dataset.cifar10 import CIFAR10_dataloader


parser = argparse.ArgumentParser(description='PyTorch CIFAR natural training')
parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--model', default='vgg11', choices=['vgg11', 'resnet', 'wrn', 'preresnet', 'resnet_ff'],
                    help='directory of model for saving checkpoint')
# checkpoint
parser.add_argument('--ckpt-path', type=str, default='./ckpt/', help='checkpoint path')
#/home/binxiao/nnet/AAAT/ckpt/vgg11/NT/best_NT_model.pt
parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100', 'svhn', 'tin'], help='dataset name')
parser.add_argument('--dataset-path', default='/home/binxiao/dataset/', help='dataset path')


args = parser.parse_args()

## load dataset
if args.dataset == 'cifar10':
    train_loader, val_loader, test_loader = CIFAR10_dataloader(args.dataset_path + 'cifar10', batch_size=args.batch_size)
    num_classes = 10
# get the batch for Hessian computation
for input, target in test_loader:
    break


# load model
if args.model =='resnet':
    model = ResNet18(num_classes=num_classes)
elif args.model == 'vgg11':
    model = vgg11_bn(num_classes=num_classes)


# load checkpoint
state = torch.load(args.ckpt_path)
ckpt = state["state_dict"]
newckpt = {}
for k,v in ckpt.items():
    if "module." in k:
        newckpt[k.replace("module.", "")] = v
    else:
        newckpt[k] = v
del ckpt
model.load_state_dict(newckpt, strict=True)
model.eval()
print("Successfully loaded model!")

# define the criterion
criterion = torch.nn.CrossEntropyLoss()

# check the accuracy
acc1 = accuracy(model(input), target)[0]
print("top-1 accuracy is %.4f %%." % (acc1))


# compute the eigenvalues, eigenvectors
values, vectors = eigen(model, input, target, criterion)
# values, vectors = eigenvalue(model, input, target, criterion)
# values, vectors = eigenvalue_dataloader(model, test_loader, criterion)
# compute the trace
traces = trace(model, input, target, criterion, iteration=20)



i = 0
for name, para in model.named_parameters():
    if 'weight' in name:
        print(name)
        # print('top-1 eigenvalue: ', values[0][i]) # for eigenvalue funciton
        print('top-1 eigenvalue: ', values[i])
        # print('Corresponding eigenvector: ', vectors[i])
        print('Trace is :', traces[i],)
        i += 1
