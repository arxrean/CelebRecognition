import os
from PIL import Image
import argparse
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import numpy as np
from tensorboard_logger import configure, log_value
import re
import shutil

from lfw_eval import read_pairs, get_paths, lfw_evaluate_acc
from model import LightCNN_29Layers_v3, LightCNN_29Layers_without_arc


def p_args():
    parser = argparse.ArgumentParser(description='lightcnn_train_gt',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--session_name', default='light_no_arc')
    parser.add_argument(
        '--csv_path', default='./save/csv2/valid_data_done.csv')
    parser.add_argument('--pretrained_weights_path', default='./save/weights')

    # train
    parser.add_argument('--pretrained', default=True, type=bool)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--cuda', default=False, type=bool)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_memory', default=True, type=bool)
    parser.add_argument('--epoch', default=1000, type=int)
    parser.add_argument('--trainval_split', default=0.1, type=int)

    # adjust
    parser.add_argument('--lr', default=1e-6, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)

    args = parser.parse_args()
    return args


class FamousClassification(Dataset):
    def __init__(self, args, transform=None, mode='train'):
        # img_id(dir_file),path,label,is_train
        self.args = args
        self.csv = pd.read_csv(args.csv_path)
        self.transform = transform

        self.mode = mode
        if self.mode == 'train':
            self.csv = self.csv[self.csv['is_train'] == 1]
            self.class_num = len(
                set(self.csv[self.csv['is_train'] == 1]['trainlabel']))
        elif self.mode == 'val':
            self.csv = self.csv[self.csv['is_train'] == 0]
            self.class_num = len(
                set(self.csv[self.csv['is_train'] == 0]['trainlabel']))
        else:
            self.csv = self.csv[self.csv['is_train'] == 2]
        self.csv.reset_index(drop=True, inplace=True)

    def __getitem__(self, index):
        item = self.csv.loc[index]
        img = Image.open(item['path']).convert('L')
        img = self.transform(img)

        label = int(item['trainlabel'])

        return item['path'], img, label

    def __len__(self):
        return len(self.csv)


def data_loader_from_folder(root_dir, transform, args):
    data = torchvision.datasets.ImageFolder(root_dir, transform=transform)
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(args.trainval_split * num_train))
    if args.trainval_shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        data, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=args.num_workers, pin_memory=args.pin_memory,
    )
    valid_loader = torch.utils.data.DataLoader(
        data, batch_size=args.batch_size, sampler=valid_sampler,
        num_workers=args.num_workers, pin_memory=args.pin_memory,
    )

    return train_loader, valid_loader


def get_finetune_optimizer(args, model, epoch, mode='arc'):
    if mode == 'arc':
        lr = args.lr
        fc_bias = []
        other_bias = []
        arc_w = []
        other_w = []
        for name, value in model.named_parameters():
            if 'bias' in name:
                if 'fc' in name:
                    fc_bias.append(value)
                else:
                    other_bias.append(value)
            else:
                if 'weights_arc' in name:
                    arc_w.append(value)
                else:
                    other_w.append(value)

        opt = optim.SGD([{'params': fc_bias, 'lr': lr, 'weight_decay': 0},
                         {'params': other_bias, 'lr': lr, 'weight_decay': 0},
                         {'params': arc_w, 'lr': 10*lr},
                         {'params': other_w, 'lr': lr}], momentum=0.9, weight_decay=args.weight_decay)

        return opt, lr

    elif mode == 'fc2':
        lr = args.lr
        last = []
        others = []
        for name, value in model.named_parameters():
            if 'fc2' in name:
                last.append(value)
            else:
                others.append(value)
        opt = optim.SGD([{'params': last, 'lr': 5 * lr},
                         {'params': others, 'lr': lr}], momentum=0.9, weight_decay=args.weight_decay)

        return opt, lr

    elif mode == 'none':
        lr = args.lr

        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=args.weight_decay), lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def train(args):
    print(os.getcwd()+': train m 0')
    train_img_transform = transforms.Compose([
        # transforms.Grayscale(),
        transforms.ColorJitter(0.1),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    val_img_transform = transforms.Compose([
        # transforms.Grayscale(),
        transforms.ToTensor()
    ])

    transform = {'train': train_img_transform, 'val': val_img_transform}

    train_loader = torch.utils.data.DataLoader(FamousClassification(args=args, transform=transform['train']),
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.num_workers,
                                               pin_memory=args.pin_memory)

    val_loader = torch.utils.data.DataLoader(FamousClassification(args=args, transform=transform['val'], mode='val'),
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.num_workers,
                                             pin_memory=args.pin_memory)

    model = LightCNN_29Layers_v3(
        num_classes=train_loader.dataset.class_num, args=args)
    if args.pretrained:
        pre_dict = torch.load(os.path.join(
            args.pretrained_weights_path, 'lightCNN_62_checkpoint.pth.tar'), map_location='cpu')['state_dict']
        pre_dict = {k[7:]: v for k, v in pre_dict.items(
        ) if 'fc2' not in k and 'arc' not in k}
        model_dict = model.state_dict()
        model_dict.update(pre_dict)
        model.load_state_dict(model_dict)
    # model.load_state_dict(torch.load('./save/weights/train_nom/train_arc_4.pt', map_location='cpu'))

    if args.cuda:
        model = nn.DataParallel(model).cuda()

    criterion = nn.CrossEntropyLoss()

    for p in range(args.epoch):
        total_loss = 0.
        optimizer, cur_lr = get_finetune_optimizer(args, model, p)
        for iter, pack in enumerate(train_loader):
            imgs = pack[1]
            labels = pack[2]
            if args.cuda:
                imgs = imgs.cuda()
                labels = labels.cuda()

            out, _ = model(imgs, labels)
            loss = criterion(out, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        res = []
        gt = []
        for iter, pack in enumerate(val_loader):
            imgs = pack[1]
            labels = pack[2]
            if args.cuda:
                imgs = imgs.cuda()
                labels = labels.cuda()

            out, _ = model(imgs, labels)
            res.append(out.detach().cpu().numpy())
            gt.append(labels.cpu().numpy())

        res = np.argmax(np.concatenate(res, 0), axis=1)
        gt = np.concatenate(gt)
        prec1 = np.mean(res == gt)
        print('epoch:{} loss:{} val_prec1:{} lr:{}'.format(
            p, total_loss, prec1, cur_lr), flush=True)
        torch.save(model.module.state_dict(), os.path.join(
            args.pretrained_weights_path, 'train_m0/train_arc_{}.pt'.format(p)))


def train_with_lfw(args):
    train_img_transform = transforms.Compose([
        # transforms.Grayscale(),
        transforms.ColorJitter(0.1),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    transform = {'train': train_img_transform}

    train_loader = torch.utils.data.DataLoader(FamousClassification(args=args, transform=transform['train']),
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.num_workers,
                                               pin_memory=args.pin_memory,
                                               drop_last=True)

    #################### val data #######################
    val_img_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    lfw_pairs = "/mnt/lustre/jiangsu/dlar/home/zyk17/data/LFW/pairs.txt"
    lfw_dir = "/mnt/lustre/jiangsu/dlar/home/zyk17/data/LFW/lfw_insight"

    pairs = read_pairs(os.path.expanduser(lfw_pairs))
    paths, actual_issame = get_paths(os.path.expanduser(lfw_dir), pairs)
    #####################################################

    model = LightCNN_29Layers_v3(
        num_classes=train_loader.dataset.class_num, args=args)

    if args.pretrained:
        pre_dict = torch.load(os.path.join(
            args.pretrained_weights_path, 'lightCNN_62_checkpoint.pth.tar'), map_location='cpu')['state_dict']
        pre_dict = {k[7:]: v for k, v in pre_dict.items() if 'arc' not in k}
        model_dict = model.state_dict()
        model_dict.update(pre_dict)
        model.load_state_dict(model_dict)

    if args.cuda:
        model = nn.DataParallel(model).cuda()

    criterion = nn.CrossEntropyLoss()
    if os.path.exists('./save/weights/train_with_lfw_dropout'):
        shutil.rmtree('./save/weights/train_with_lfw_dropout')
    os.makedirs('./save/weights/train_with_lfw_dropout')

    train_loss = []
    train_acc = []
    val_acc = []
    for p in range(args.epoch):
        optimizer, cur_lr = get_finetune_optimizer(args, model, p)

        train_loss.append(0)
        epoch_out = []
        epoch_label = []
        model = model.train()
        for iter, pack in enumerate(train_loader):
            imgs = pack[1]
            labels = pack[2]
            if args.cuda:
                imgs = imgs.cuda()
                labels = labels.cuda()

            out, _ = model(imgs, labels)
            loss = criterion(out, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss[-1] += loss.item()
            epoch_out.append(out.detach().cpu().numpy())
            epoch_label.append(labels.cpu().numpy())

        epoch_out = np.concatenate(epoch_out, axis=0)
        epoch_label = np.concatenate(epoch_label)
        train_acc.append(np.mean(np.argmax(epoch_out, axis=1) == epoch_label))

        model = model.eval()
        embeddings = []
        for path in paths:
            img = Image.open(path).convert('L')
            img = val_img_transform(img)
            if args.cuda:
                img = img.cuda()
            _, fc = model(img.unsqueeze(0))
            embeddings.append(fc.detach().cpu().numpy())

        embeddings = np.concatenate(embeddings, 0)
        accuracy = lfw_evaluate_acc(
            embeddings, actual_issame, nrof_folds=10, distance_metric=1, subtract_mean=False)

        snap_shot = {'epoch': p, 'train_loss': train_loss, 'train_acc': train_acc,
                     'val_acc': np.mean(accuracy), 'state_dict': model.module.state_dict()}

        print('epoch:{} train_loss:{} train_acc:{} val_acc:{} cur_lr:{}'.format(
            p, train_loss[-1], train_acc[-1], np.mean(accuracy), cur_lr))

        torch.save(
            './save/weights/train_with_lfw_dropout/{}.pth.tar'.format(p), snap_shot)


def train_without_arc(args):
    print('train_without_arc')
    train_img_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ColorJitter(0.1),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    val_img_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor()
    ])

    transform = {'train': train_img_transform, 'val': val_img_transform}

    train_loader = torch.utils.data.DataLoader(FamousClassification(args=args, transform=transform['train']),
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.num_workers,
                                               pin_memory=args.pin_memory)

    val_loader = torch.utils.data.DataLoader(FamousClassification(args=args, transform=transform['val'], mode='val'),
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.num_workers,
                                             pin_memory=args.pin_memory)

    model = LightCNN_29Layers_without_arc(
        num_classes=train_loader.dataset.class_num, args=args)
    if args.pretrained:
        pre_dict = torch.load(os.path.join(
            args.pretrained_weights_path, 'lightCNN_62_checkpoint.pth.tar'), map_location='cpu')['state_dict']
        pre_dict = {k[7:]: v for k, v in pre_dict.items(
        ) if 'fc2' not in k and 'arc' not in k}
        model_dict = model.state_dict()
        model_dict.update(pre_dict)
        model.load_state_dict(model_dict)

    if args.cuda:
        model = nn.DataParallel(model).cuda()

    criterion = nn.CrossEntropyLoss()

    for p in range(args.epoch):
        total_loss = 0.
        optimizer, cur_lr = get_finetune_optimizer(args, model, p, mode='fc2')
        for iter, pack in enumerate(train_loader):
            imgs = pack[1]
            labels = pack[2]
            if args.cuda:
                imgs = imgs.cuda()
                labels = labels.cuda()

            out, _ = model(imgs)
            loss = criterion(out, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        res = []
        gt = []
        for iter, pack in enumerate(val_loader):
            imgs = pack[1]
            labels = pack[2]
            if args.cuda:
                imgs = imgs.cuda()
                labels = labels.cuda()

            out, _ = model(imgs)
            res.append(out.detach().cpu().numpy())
            gt.append(labels.cpu().numpy())

        res = np.argmax(np.concatenate(res, 0), axis=1)
        gt = np.concatenate(gt)
        prec1 = np.mean(res == gt)
        print('epoch:{} loss:{} val_prec1:{} lr:{}'.format(
            p, total_loss, prec1, cur_lr), flush=True)
        torch.save(model.module.state_dict(), os.path.join(
            args.pretrained_weights_path, 'train_noarc/train_noarc_{}.pt'.format(p)))


if __name__ == '__main__':
    args = p_args()
    print(args)
    train_with_lfw(args)
