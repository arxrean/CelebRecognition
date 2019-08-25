import os
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
from tqdm import tqdm
# from tensorboard_logger import configure, log_value

from main import FamousClassification, accuracy
from model import LightCNN_29Layers_v3, LightCNN_29Layers_without_arc


def p_args():
    parser = argparse.ArgumentParser(description='lightcnn_train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--session_name', default='light_cnn_train')
    parser.add_argument('--csv_path', default='./save/csv2/valid_data_done.csv')
    parser.add_argument('--pretrained_weights_path', default='./save/weights')

    # train
    parser.add_argument('--cuda', default=False, type=bool)

    args = parser.parse_args()
    return args


def infer_train_star_cls(args):
    val_img_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor()
    ])

    transform = {'train': val_img_transform}
    val_loader = torch.utils.data.DataLoader(FamousClassification(
        args=args, transform=transform['train'], mode='train'), batch_size=1, shuffle=False)
    print('load data success!', flush=True)

    model = LightCNN_29Layers_without_arc(num_classes=8642, args=args)
    model.load_state_dict(torch.load(os.path.join(args.pretrained_weights_path,
                                                  'light_no_arc_best.pt'), map_location='cpu'))
    if args.cuda:
        model = model.cuda()

    res = []
    gt = []
    path = []
    for iter, pack in enumerate(val_loader):
        imgs = pack[1]
        labels = pack[2]
        if args.cuda:
            imgs = imgs.cuda()
            labels = labels.cuda()
        out, _ = model(imgs, labels)
        res.append(out.detach().cpu().numpy())
        gt.append(labels.cpu().numpy())
        path.append(pack[0][0])
    res = np.concatenate(res, axis=0)
    gt = np.asarray(np.concatenate(gt, axis=0))
    prec1, prec5 = accuracy(torch.from_numpy(
        res), torch.from_numpy(gt), topk=(1, 5))
    print('prec1:{}'.format(prec1), flush=True)
    print('prec5:{}'.format(prec5), flush=True)
    exit(0)
    res_max = np.argmax(res, axis=1)
    path = np.asarray(path)
    with open('./save/train_res.txt', 'w') as f:
        for i in range(len(res_max)):
            f.write('res: {} gt:{} raw_res: {} raw_gt:{} path:{}'.format(
                    res_max[i], gt[i], get_rawlabel_from_label(loader=val_loader, label=res_max[i]), get_rawlabel_from_label(loader=val_loader, label=gt[i]), path[i])+'\n')


def infer_val_star_cls(args):
    val_img_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor()
    ])

    transform = {'test': val_img_transform}
    val_loader = torch.utils.data.DataLoader(FamousClassification(
        args=args, transform=transform['test'], mode='test'), batch_size=1, shuffle=False)
    print('load data success!', flush=True)

    model = LightCNN_29Layers_v3(num_classes=8642, args=args)
    model.load_state_dict(torch.load(os.path.join(args.pretrained_weights_path,
                                                  'light_cnn_arc_loss_best.pt'), map_location='cpu'))
    if args.cuda:
        model = model.cuda()

    res = []
    gt = []
    path = []
    for iter, pack in enumerate(val_loader):
        imgs = pack[1]
        labels = pack[2]
        if args.cuda:
            imgs = imgs.cuda()
            labels = labels.cuda()
        out, _ = model(imgs, labels)
        res.append(out.detach().cpu().numpy())
        gt.append(labels.cpu().numpy())
        path.append(pack[0][0])
    res = np.concatenate(res, axis=0)
    gt = np.asarray(np.concatenate(gt, axis=0))
    prec1, prec5 = accuracy(torch.from_numpy(
        res), torch.from_numpy(gt), topk=(1, 5))
    print('prec1:{}'.format(prec1))
    print('prec5:{}'.format(prec5))
    if 'val' in transform:
        res_max = np.argmax(res, axis=1)
        path = np.asarray(path)
        with open('./save/val_res.txt', 'w') as f:
            for i in range(len(res_max)):
                f.write('res: {} gt:{} raw_res: {} raw_gt:{} path:{}'.format(
                        res_max[i], gt[i], get_rawlabel_from_label(loader=val_loader, label=res_max[i]), get_rawlabel_from_label(loader=val_loader, label=gt[i]), path[i])+'\n')


def gen_train_mean_feature():
    print(os.getcwd()+': gen_train_mean_feature', flush=True)
    val_img_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor()
    ])

    transform = {'train': val_img_transform}
    data = pd.read_csv(args.csv_path)

    model = LightCNN_29Layers_v3(num_classes=8642, args=args)
    pre_dict = torch.load(os.path.join(
        args.pretrained_weights_path, 'lightCNN_62_checkpoint.pth.tar'), map_location='cpu')['state_dict']
    pre_dict = {k[7:]: v for k, v in pre_dict.items(
    ) if 'fc2' not in k and 'arc' not in k}
    model_dict = model.state_dict()
    model_dict.update(pre_dict)
    model.load_state_dict(model_dict)

    if args.cuda:
        model = model.cuda()

    for name, group in data.groupby('rawlabel'):
        imgs = [val_img_transform(Image.open(path).convert('L'))
                for path in list(group['path'])]
        if args.cuda:
            imgs = [img.cuda() for img in imgs]
        fcs = [model(img.unsqueeze(0))[1].detach().cpu().numpy()
               for img in imgs]
        fc_mean = np.mean(fcs, 0)
        np.save('./save/register/train_avg/baseline_valid/{}.npy'.format(name), fc_mean)


def gen_train_center_feature():
    print(os.getcwd()+': gen_train_center_feature', flush=True)
    val_img_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor()
    ])

    transform = {'train': val_img_transform}
    data = pd.read_csv(args.csv_path)

    model = LightCNN_29Layers_v3(num_classes=8642, args=args)
    pre_dict = torch.load(os.path.join(
        args.pretrained_weights_path, 'lightCNN_62_checkpoint.pth.tar'), map_location='cpu')['state_dict']
    pre_dict = {k[7:]: v for k, v in pre_dict.items(
    ) if 'fc2' not in k and 'arc' not in k}
    model_dict = model.state_dict()
    model_dict.update(pre_dict)
    model.load_state_dict(model_dict)

    if args.cuda:
        model = model.cuda()

    mean_feature_list = []
    for name, group in data[(data['is_train']==1) | (data['is_train']==0)].groupby('rawlabel'):
        paths = list(group['path'])
        img = torch.cat([val_img_transform(Image.open(
            path).convert('L')).unsqueeze(0) for path in paths])
        if args.cuda:
            img = img.cuda()
        _, fc = model(img)
        fc = fc.detach()

        best_index = -1
        best_sim = -1
        for i in range(fc.size()[0]):
            sum_sim = np.sum(
                [torch.cosine_similarity(fc[i].unsqueeze(0), f.unsqueeze(0)).item() for f in fc])
            if best_index == -1 or sum_sim > best_sim:
                best_index = i
                best_sim = sum_sim

        np.save('./save/register/train_center/baseline/{}.npy'.format(name),
                fc[i].unsqueeze(0).cpu().numpy())
        print('register {} success!'.format(name), flush=True)


def gen_train_key_feature():
    print(os.getcwd()+': gen_train_key_feature', flush=True)
    val_img_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor()
    ])

    transform = {'train': val_img_transform}
    data = pd.read_csv(args.csv_path)

    model = LightCNN_29Layers_v3(num_classes=8642, args=args)
    pre_dict = torch.load(os.path.join(
        args.pretrained_weights_path, 'lightCNN_62_checkpoint.pth.tar'), map_location='cpu')['state_dict']
    pre_dict = {k[7:]: v for k, v in pre_dict.items(
    ) if 'fc2' not in k and 'arc' not in k}
    model_dict = model.state_dict()
    model_dict.update(pre_dict)
    model.load_state_dict(model_dict)

    if args.cuda:
        model = model.cuda()

    for name, group in data[(data['is_train']==1) | (data['is_train']==0)].groupby('rawlabel'):
        imgs = [val_img_transform(Image.open(path).convert('L'))
                for path in list(group['path']) if 'key' in path]
        if len(imgs) == 1:
            if args.cuda:
                imgs = [img.cuda() for img in imgs]
            fcs = [model(img.unsqueeze(0))[1].detach().cpu().numpy()
                   for img in imgs]
            fc_key = fcs[0]
            np.save('./save/register/key/baseline/{}.npy'.format(name), fc_key)

        else:
            imgs = [val_img_transform(Image.open(path).convert('L'))
                    for path in list(group['path'])]
            if args.cuda:
                imgs = [img.cuda() for img in imgs]
            fcs = [model(img.unsqueeze(0))[1].detach().cpu().numpy()
                   for img in imgs]
            fc_mean = np.mean(fcs, 0)
            np.save('./save/register/key/baseline/{}.npy'.format(name), fc_mean)


def infer_cls_by_feature(args, mode='baseline_avg'):
    print(os.getcwd()+': infer_cls_by_feature '+mode, flush=True)
    val_img_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor()
    ])
    transform = {'val': val_img_transform}
    val_loader = torch.utils.data.DataLoader(FamousClassification(
        args=args, transform=transform['val'], mode='val'), batch_size=1, shuffle=False)

    model = LightCNN_29Layers_v3(num_classes=9301, args=args)
    model.load_state_dict(torch.load(os.path.join(args.pretrained_weights_path,
                                                  'light_cnn_arc_best_v2_9301.pt'), map_location='cpu'))
    train_feat_dict = dict()
    if mode == 'baseline_key':
        for npy in os.listdir('./save/register/key/baseline'):
            train_feat_dict[npy[:-4]
                            ] = np.load(os.path.join('./save/register/key/baseline', npy))
    if mode == 'baseline_avg':
        for npy in os.listdir('./save/register/train_avg/baseline'):
            train_feat_dict[npy[:-4]
                            ] = np.load(os.path.join('./save/register/train_avg/baseline', npy))

    if mode == 'baseline_center':
        for npy in os.listdir('./save/register/train_center/baseline'):
            train_feat_dict[npy[:-4]
                            ] = np.load(os.path.join('./save/register/train_center/baseline', npy))

    star_cate_list = np.array(list(train_feat_dict.keys()))
    star_feat_list = torch.from_numpy(
        np.concatenate(list(train_feat_dict.values()), axis=0))
    if args.cuda:
        star_feat_list = star_feat_list.cuda()

    if args.cuda:
        model = model.cuda()

    img_path = []
    res = []
    res_5 = []
    gt = []
    for iter, pack in enumerate(val_loader):
        imgs = pack[1]
        labels = pack[2]
        if args.cuda:
            imgs = imgs.cuda()
            labels = labels.cuda()
        _, fc = model(imgs)
        cos_sim = [torch.cosine_similarity(
            fc, f.unsqueeze(0), dim=1).item() for f in star_feat_list]

        res.append(int(star_cate_list[np.argmax(cos_sim)]))
        res_5.append(list(map(int, star_cate_list[np.argsort(cos_sim)[-5:]])))
        gt.append(labels.item())
        if res[-1] != gt[-1]:
            print('path:{} res:{} gt:{}'.format(pack[0], int(
                star_cate_list[np.argmax(cos_sim)]), labels.item()), flush=True)
    res = np.array(res)
    gt = np.array(gt)
    res_5 = np.asarray(res_5)

    print('prec1:{}'.format(np.mean(res == gt)))
    print('prec5:{}'.format(
        np.mean([gt[i] in res_5[i] for i in range(len(gt))])))