import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import os

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from Code.utils.utils import MyDataset, validate, show_confMat
from datetime import datetime
from models import FPCN
from torchvision import models
from datasets import TestDataset, TrainDataset, BalancedBatchSampler
from utils import accuracy, AverageMeter, save_checkpoint, save_distillModel
import argparse

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--exp_name', default=None, type=str,
                    help='name of experiment')
parser.add_argument('--data', metavar='DIR', default='',
                    help='path to dataset')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=20, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=1, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--evaluate-freq', default=10, type=int,
                    help='the evaluation frequence')
parser.add_argument('--resume', default='./distillModel.pth.tar', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--n_classes', default=3, type=int,
                    help='the number of classes')
parser.add_argument('--n_samples', default=2, type=int,
                    help='the number of samples per class')

best_prec1 = 0
args = parser.parse_args()


class anNet(nn.Module):
    def __init__(self):
        super(anNet, self).__init__()
        self.resnet34 = models.resnet34(pretrained=True)
        self.fc = nn.Linear(1000, 200)

    def forward(self, images, targets=None, flag='train'):
        softmax_out = self.resnet34(images)
        logits_out = self.fc(softmax_out).squeeze()
        return logits_out


correct_ratio = []
alpha = 0.5

train_dataset = TrainDataset(transform=transforms.Compose([
    transforms.Resize([512, 512]),
    transforms.RandomCrop([448, 448]),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )]))

train_sampler = BalancedBatchSampler(train_dataset, n_classes=3, n_samples=1)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_sampler=train_sampler,
    num_workers=args.workers, pin_memory=True)

val_dataset = TestDataset(transform=transforms.Compose([
    transforms.Resize([512, 512]),
    transforms.CenterCrop([448, 448]),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )]))
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=16, shuffle=False,
    num_workers=args.workers, pin_memory=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
teach_model = FPCN()
teach_model.cuda()
teach_model = teach_model.to(device)
teach_model.conv = nn.DataParallel(teach_model.conv)
# teach_model.load_state_dict(torch.load('teach_net_params_0.9895.pkl'))
checkpoint = torch.load('./checkpoint.pth.tar')
teach_model.load_state_dict(checkpoint['state_dict'])

model = anNet()
model = model.to(device)
if os.path.isfile(args.resume):
    checkpoint2 = torch.load('./distillModel.pth.tar')
    model.load_state_dict(checkpoint2['state_dict'])

criterion = nn.CrossEntropyLoss()
criterion2 = nn.KLDivLoss()

optimizer = optim.Adam(model.parameters(), lr=0.0001)
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

for epoch in range(200):
    loss_sigma = 0.0
    correct = 0.0
    total = 0.0
    loss_sigma2 = 0.0
    correct2 = 0.0
    total2 = 0.0
    for i, (input, target) in enumerate(train_loader):
        teach_model.eval()
        model.train()
        input_var = input.to(device)
        target_var = target.to(device).squeeze()
        optimizer.zero_grad()
        # teacher output
        with torch.no_grad():
            teacher_outputs = teach_model(input_var, targets=None, flag='val')

        # student output
        outputs = model(input_var)

        # true loss
        loss1 = loss1 = criterion(outputs, target_var)

        # soft loss
        T = 2
        outputs_S = F.softmax(outputs / T, dim=1)
        outputs_T = F.softmax(teacher_outputs / T, dim=1)
        loss2 = criterion2(outputs_S, outputs_T) * T * T
        loss = loss1 * (1 - alpha) + loss2 * alpha
        loss.backward()
        optimizer.step()

        _, predicted2 = torch.max(teacher_outputs.data, dim=1)
        total2 += target_var.size(0)
        correct2 += (predicted2.cpu() == target_var.cpu()).squeeze().sum().numpy()
        loss_sigma2 += loss.item()
        if i % 100 == 0:
            loss_avg2 = loss_sigma2 / 10
            loss_sigma2 = 0.0
            print('teacher loss_avg:{:.2}   Acc:{:.2%}'.format(loss_avg2, correct2 / total2))

        _, predicted = torch.max(outputs.data, dim=1)
        total += target_var.size(0)
        correct += (predicted.cpu() == target_var.cpu()).squeeze().sum().numpy()
        loss_sigma += loss.item()
        if i % 100 == 0:
            loss_avg = loss_sigma / 10
            loss_sigma = 0.0
            print('loss_avg:{:.2}   Acc:{:.2%}'.format(loss_avg, correct / total))

    top1 = AverageMeter()
    top5 = AverageMeter()
    # eval

    num_correct = 0
    num_samples = 0

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input_var = input.to(device)
            target_var = target.to(device).squeeze()
            logits = model(input_var)
            num_correct += (target_var == logits).sum()
            num_samples += logits.size(0)
        acc = (num_correct / num_samples).item()
    print('Epoch:{}\t Accuracy:{:.2f}'.format(epoch + 1, acc))

    if epoch % 2 == 0:
        loss_sigma = 0.0
        cls_num = 200
        conf_mat = np.zeros([cls_num, cls_num])  # 混淆矩阵
        model.eval()
        with torch.no_grad():
            for i, (input, target) in enumerate(val_loader):
                input_var = input.to(device)
                target_var = target.to(device).squeeze()
                logits = model(input_var)

                prec1 = accuracy(logits, target_var, 1)
                top1.update(prec1, logits.size(0))

        print(' * StudentNet Test Prec@1 ' + str(top1.avg))