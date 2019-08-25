# from model import LightCNN_29Layers_v3, LightCNN_29Layers_without_arc
import pickle
from distutils.dir_util import copy_tree
from torchvision import transforms
import numpy as np
import pandas as pd
from PIL import Image
import os
from shutil import copyfile
import torch
import argparse
import sys
import scipy.spatial
import pandas as pd
from matplotlib import pyplot as plt


def p_args():
    parser = argparse.ArgumentParser(description='lightcnn_train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--session_name', default='light_cnn_train')
    parser.add_argument('--csv_path', default='./save/csv2/data_done.csv')
    parser.add_argument('--pretrained_weights_path', default='./save/weights')

    # train
    parser.add_argument('--cuda', default=False, type=bool)

    args = parser.parse_args()
    return args


def pad_0(id):
    while len(id) < 4:
        id = '0'+id

    return id


def gen_lfw_deep_csv_file(input_dir, output_file):
    csv_data = []
    for d in os.listdir(input_dir):
        dir_path = os.path.join(input_dir, d)
        for file in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file)
            csv_data.append([file, file_path, d])

    csv_data = pd.DataFrame(csv_data, columns=['name', 'path', 'rawlabel'])
    rawlabel_list = list(set(list(csv_data['rawlabel'])))
    csv_data['label'] = -1
    for index in csv_data.index:
        csv_data.at[index, 'label'] = rawlabel_list.index(
            csv_data.loc[index]['rawlabel'])

    csv_data.to_csv(output_file, index=False)


def gen_data_csv(img_root_dir):
    # imgname | path | rawlabel | label
    print(os.getcwd()+': gen_data_csv')
    res = []
    for star_dir in os.listdir(img_root_dir):
        star_dir_path = os.path.join(img_root_dir, star_dir)
        for img_file in os.listdir(star_dir_path):
            if '.jpg' in img_file:
                img_path = os.path.join(star_dir_path, img_file)
                label = star_dir
                res.append([img_file, img_path, label])

    csv = pd.DataFrame(res, columns=['id', 'path', 'rawlabel'])
    csv.to_csv('./save/csv2/data.csv', index=False)


def split_data_csv(data_csv_path, pair_csv_path):
    data = pd.read_csv(data_csv_path)
    pairs = pd.read_csv(pair_csv_path)
    pair_list = list(set(list(pairs['s1'])+list(pairs['s2'])))

    data['is_train'] = 1
    data.at[data['path'].isin(pair_list), 'is_train'] = -1
    for name, group in data[data['is_train'] == 1].groupby('rawlabel'):
        if len(group) < 4:
            data.at[group.index, 'is_train'] = -2

    data['trainlabel'] = -1
    labels = list(set(data[data['is_train'] == 1]['rawlabel']))
    for index in data[data['is_train'] == 1].index:
        print(data.at[index, 'rawlabel'])
        data.at[index, 'trainlabel'] = labels.index(
            data.at[index, 'rawlabel'])

    for name, group in data[data['is_train'] == 1].groupby('trainlabel'):
        samples = group[group['id'] != 'key.jpg'].sample(frac=0.2)
        if len(samples) > 0:
            data.at[samples.index, 'is_train'] = 0

    data.to_csv('./save/csv2/data_done.csv', index=False)


def gen_pair_csv(data_csv_path):
    data_csv = pd.read_csv(data_csv_path)
    data_csv = data_csv[data_csv['is_train'] == 0]
    labels = np.array(list(set(data_csv['rawlabel'])))
    pair_csv = []
    flag = 0
    ex = []
    while True:
        label = np.random.choice(labels, 1)[0]
        # different
        if flag == 0:
            another_label = np.random.choice(
                labels[np.where(labels != label)], 1)[0]
            sample1 = list(data_csv[data_csv['rawlabel']
                                    == label].sample(1)['path'])[0]
            sample2 = list(data_csv[data_csv['rawlabel'] == another_label].sample(1)[
                'path'])[0]
            if '{}-{}'.format(sample1, sample2) in ex:
                continue
            pair_csv.append([sample1, sample2, 0])
            flag = 1
            ex.append('{}-{}'.format(sample1, sample2))
            ex.append('{}-{}'.format(sample2, sample1))
        else:
            try:
                samples = list(data_csv[data_csv['rawlabel']
                                        == label].sample(2)['path'])
                if '{}-{}'.format(samples[0], samples[1]) in ex:
                    continue
                ex.append('{}-{}'.format(samples[0], samples[1]))
                ex.append('{}-{}'.format(samples[1], samples[0]))
                pair_csv.append([samples[0], samples[1], 1])

                flag = 0
            except:
                print('err label:{}'.format(label), flush=True)
                continue

        if len(pair_csv) == 10000:
            break

    pair_csv = pd.DataFrame(pair_csv, columns=['s1', 's2', 'label'])
    pair_csv.to_csv('./save/csv2/val_pair.csv', index=False)


def extract_wrong_2_win10(wrong_slurm_path, ex_dir_pkl_path, img_root_path):
    print(os.getcwd()+': extract_wrong_2_win10', flush=True)
    all_dirs = []
    ex_dirs = pickle.load(open(ex_dir_pkl_path, 'rb'))
    for line in open(wrong_slurm_path).readlines():
        if 'path1' in line and 'path2' in line:
            path1, path2 = line.split(' ')[0], line.split(' ')[1]
            dir1, dir2 = path1.split('/')[-2], path2.split('/')[-2]
            if dir1 not in ex_dirs:
                all_dirs.append(dir1)
            if dir2 not in ex_dirs:
                all_dirs.append(dir2)

    for d in set(all_dirs):
        copy_tree(os.path.join(img_root_path, d), os.path.join('/mnt/lustre/jiangsu/dlar/home/zyk17/tmp',
                                                               d))


def gen_rgb_data(csv='./save/csv2/data_done.csv'):
    csv = pd.read_csv(csv)
    rgb_key_repo = '/mnt/lustre/jiangsu/dlar/home/zyk17/data/famous/Key/face_rgb'
    rgb_data_repo = '/mnt/lustre/jiangsu/dlar/home/zyk17/data/famous/OnlySingleFaceImages_rgb/align_multi'
    root_path = '/mnt/lustre/jiangsu/dlar/home/zyk17/data/famous/MergeTrainImages_v13_rgb'
    for index in csv.index:
        gray_path = csv.at[index, 'path']
        star = gray_path.split('/')[-2]
        if not os.path.exists(os.path.join(root_path, star)):
            os.makedirs(os.path.join(root_path, star))
        rgb_img = None
        if 'key' in gray_path:
            rgb_img = os.path.join(rgb_key_repo, star+'.jpg')
            copyfile(rgb_img, os.path.join(root_path, star, 'key.jpg'))
        else:
            rgb_img = os.path.join(rgb_data_repo, star,
                                   gray_path.split('/')[-1])
            copyfile(rgb_img, os.path.join(
                root_path, star, gray_path.split('/')[-1]))


def reduce_intra_distance(args, csv_path):
    csv = pd.read_csv(csv_path)
    val_csv = csv[csv['is_train'] == 0]
    csv.at[val_csv.index, 'is_train'] = 1

    trans = transforms.Compose([
        transforms.ToTensor()
    ])

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

    path_list = []
    feat_list = []
    for name, group in csv.groupby('rawlabel'):
        path = np.array(list(group['path']))
        imgs = torch.cat([trans(Image.open(p)).unsqueeze(0)
                          for p in path], dim=0)
        if args.cuda:
            imgs = imgs.cuda()
        _, feat = model(imgs)
        feat = feat.detach().cpu().numpy()
        if len(feat) == 1:
            path_list.append(path)
            feat_list.append(feat)

        else:
            sim_matrix = np.zeros((feat.shape[0], feat.shape[0]))
            for i in range(feat.shape[0]):
                for j in range(feat.shape[0]):
                    if i == j:
                        sim_matrix[i, j] = 0
                    else:
                        sim_matrix[i, j] = 1 - \
                            scipy.spatial.distance.cosine(feat[i], feat[j])

            sim_matrix = np.sum(sim_matrix, 0)/(len(sim_matrix)-1)
            valid_path = path[sim_matrix > 0.55]
            valid_feat = feat[sim_matrix > 0.55]

            path_list.append(valid_path)
            feat_list.append(valid_feat)

    np.save('./save/log/inter_intra/path_list.npy', path_list)
    np.save('./save/log/inter_intra/feat_list.npy', feat_list)


def reduce_dulicate():
    path_list = np.load('./save/log/inter_intra/path_list.npy')
    feat_list = np.load('./save/log/inter_intra/feat_list.npy')

    valid_path_list = []
    valid_feat_list = []

    for i in range(len(path_list)):
        print('start:{}'.format(i), flush=True)
        star_path = path_list[i]
        star_feat = feat_list[i]

        if len(star_path) == 0:
            continue
        elif len(star_path) == 1:
            valid_path_list.append(star_path)
            valid_feat_list.append(star_feat)
        else:
            duplicate_list = []
            for i in range(len(star_path)):
                for j in range(i+1, len(star_path)):
                    sim = 1 - \
                        scipy.spatial.distance.cosine(
                            star_feat[i], star_feat[j])
                    if sim > 0.95:
                        duplicate_list.append(i)

            star_path = [star_path[i]
                         for i in range(len(star_path)) if i not in duplicate_list]
            star_feat = [star_feat[i]
                         for i in range(len(star_feat)) if i not in duplicate_list]
            valid_path_list.append(star_path)
            valid_feat_list.append(star_feat)

    np.save('./save/log/inter_intra/nond_path_list.npy', valid_path_list)
    np.save('./save/log/inter_intra/nond_feat_list.npy', valid_feat_list)


def reduce_inter_distance():
    path_list = np.load('./save/log/inter_intra/nond_path_list.npy')
    feat_list = np.load('./save/log/inter_intra/nond_feat_list.npy')
    # max_intra_sim = np.load('./save/log/inter_intra/max_intra_sim.npy')

    valid_path = []
    bad_path = []
    for i in range(len(path_list)):
        print('start:{}'.format(i), flush=True)
        star_path = path_list[i]
        star_feat = feat_list[i]
        if len(star_path) == 0:
            continue
        elif len(star_path) == 1:
            img_path = star_path[0]
            img_feat = star_feat[0]
            flag = True

            for k in range(i+1, len(path_list)):
                star_path_2 = path_list[k]
                star_feat_2 = feat_list[k]
                if len(star_feat_2) == 0:
                    continue

                min_inter_sim = np.min(
                    [1-scipy.spatial.distance.cosine(img_feat, f) for f in star_feat_2])

                if min_inter_sim >= 0.6:
                    bad_path.append(img_path)
                    flag = False
                    break

            if flag:
                valid_path.append(img_path)
            else:
                bad_path.append(img_path)
        else:
            for j in range(len(star_path)):
                img_path = star_path[j]
                img_feat = star_feat[j]
                flag = True

                valid_intra_sim = [1-scipy.spatial.distance.cosine(
                    img_feat, star_feat[i]) for i in range(len(star_feat)) if i != j]
                max_intra_sim = np.max(valid_intra_sim)

                for k in range(i+1, len(path_list)):
                    star_path_2 = path_list[k]
                    star_feat_2 = feat_list[k]
                    if len(star_feat_2) == 0:
                        continue

                    min_inter_sim = np.min(
                        [1-scipy.spatial.distance.cosine(img_feat, f) for f in star_feat_2])

                    if max_intra_sim <= min_inter_sim:
                        bad_path.append(img_path)
                        flag = False
                        break

                if flag:
                    valid_path.append(img_path)
                else:
                    bad_path.append(img_path)

    np.save('./save/log/inter_intra/valid_inter_intra_path.npy', valid_path)
    np.save('./save/log/inter_intra/bad_inter_intra_path.npy', bad_path)


def count_bin_dir(dir_path):
    res = []
    for dir in os.listdir(dir_path):
        img_num = len(os.listdir(os.path.join(dir_path, dir)))
        res.append(img_num)

    plt.hist(res)
    plt.xlabel('image nums per star')
    plt.ylabel('star nums')
    plt.savefig('./save/face_dataset_hist.png')


def gen_calfw_csv(txt_path):
    root_dir = '/mnt/lustre/jiangsu/dlar/home/zyk17/data/CALFW'
    lines = open(txt_path).readlines()
    res = []
    for i in range(0, len(lines), 2):
        l1 = lines[i].strip()
        l2 = lines[i+1].strip()
        if int(l1.split(' ')[-1]) != 0:
            res.append(
                [os.path.join(root_dir, l1.split(' ')[0]), os.path.join(root_dir, l2.split(' ')[0]), 1])
        else:
            res.append([os.path.join(root_dir, l1.split(' ')[0]),
                        os.path.join(root_dir, l2.split(' ')[0]), 0])

    data = pd.DataFrame(res, columns=['s1', 's2', 'label'])
    data.to_csv('./save/csv2/ca_lfw.csv', index=False)


if __name__ == '__main__':
    count_bin_dir(
        '/mnt/lustre/jiangsu/dlar/home/zyk17/data/famous/MergeTrainImages_v13')
