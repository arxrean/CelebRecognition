from bs4 import BeautifulSoup
import urllib.request
import re
import json
import os
import pdb
import pickle
import time
from tqdm import tqdm
import threading
from fake_useragent import UserAgent
import random
import numpy as np
import urllib

import single
import utils

INFO_PATH = '/pic/info'
SMALL_INFO_PATH = 'small_info/'

'''
info{
    'id',
    'name',
    'sex',
    'constellation',
    'birth',
    'imdb',
    'des',
    'fans',
    'img_path'
}
'''


class myThread(threading.Thread):
    def __init__(self, threadID, step, ua, ips):
        threading.Thread.__init__(self)
        self.threadID = threadID

        self.step = step
        self.base = 0
        self.ua = ua
        self.ips = ips

    def run(self):
        start(low=self.base+self.threadID*self.step-self.step,
              high=self.base+self.threadID*self.step, ua=ua, ips=ips)


class mySingleThread(threading.Thread):
    def __init__(self, threadID, step, ua, ips, file_list):
        threading.Thread.__init__(self)
        self.threadID = threadID

        self.step = step
        self.base = 0
        self.ua = ua
        self.ips = ips
        self.file_list = file_list

    def run(self):
        start(low=self.base+self.threadID*self.step-self.step,
              high=self.base+self.threadID*self.step, ua=ua, ips=ips, file_list=self.file_list)


def save_single(info):
    try:
        img_path = info['img_path']
        if not img_path or len(img_path) < 10:
            with open(os.path.join(SMALL_INFO_PATH, info['id'])+'.pkl', 'wb') as f:
                pickle.dump(info, f)
        else:
            with open(os.path.join(INFO_PATH, info['id'])+'.pkl', 'wb') as f:
                pickle.dump(info, f)
        print('{}-{}:success parsed!'.format(info['id'], info['name']))

    except KeyboardInterrupt:
        exit(0)

    except Exception as e:
        print('{}:{}'.format(id, e))


def save_single_img(info):
    try:
        img_path = info['img_path']
        with open(os.path.join(INFO_PATH, info['id'])+'.pkl', 'wb') as f:
            pickle.dump(info, f)
        print('{}:success parsed!'.format(info['id']))

    except KeyboardInterrupt:
        exit(0)

    except Exception as e:
        print('{}:{}'.format(id, e))


def save_raw_img(info):
    id = info['id']
    img_path = info['img_path']

    if not os.path.exists(os.path.join(INFO_PATH, id)):
        os.mkdir(os.path.join(INFO_PATH, id))

    for i, path in enumerate(img_path):
        urllib.request.urlretrieve(path, os.path.join(
            os.path.join(INFO_PATH, id), str(i))+'.jpg')
        time.sleep(random.uniform(0, 2))

    print('{}:raw success parsed!'.format(info['id']))


def start_single(start_html, ua, ips, file_list):
    id = utils.get_only_numbers(start_html)[0]
    if id+'.pkl' not in file_list:
        try:
            response = utils.custom_get(start_html, ua, ips, False)
            html = response.read()

            # parse basic info
            soup = BeautifulSoup(html, "html.parser")
            info = single.parse_single_man(soup)
            info['id'] = id

            # parse imgs
            # single.parse_imgs(start_html, info, ua)
            info['img_path'] = []
            save_single(info)

            # if np.random.uniform() < 0.1:
            #     ips.refresh_ips()

        except KeyboardInterrupt:
            exit(0)

        except Exception as e:
            print('{}:{}'.format(id, e))

        finally:
            pass


def start_single_imgs(start_html, ua, ips, file_list):
    id = utils.get_only_numbers(start_html)[0]
    if id not in file_list:
        try:
            response = utils.custom_get(start_html, ua, ips, False)
            html = response.read()

            # parse basic info
            soup = BeautifulSoup(html, "html.parser")
            info = dict()
            info['id'] = id

            # parse imgs
            single.parse_imgs(soup, info, ua, ips)

            # save_single_img(info)
            save_raw_img(info)

            # if np.random.uniform() < 0.1:
            #     ips.refresh_ips()

        except KeyboardInterrupt:
            exit(0)

        except Exception as e:
            print('{}:{}'.format(id, e))

        finally:
            pass
    else:
        print('{} exist!'.format(id))


def start(low, high, ua, ips, file_list):
    for i in tqdm(range(low, high)):
        # start_single(
        #     'http://www.happyjuzi.com/star-{}/'.format(i), ua, ips, file_list)
        start_single_imgs(
            'http://www.happyjuzi.com/star-picture-{}/'.format(i), ua, ips, file_list)


if __name__ == '__main__':
    if not os.path.exists(INFO_PATH):
        os.mkdir(INFO_PATH)
    if not os.path.exists(SMALL_INFO_PATH):
        os.mkdir(SMALL_INFO_PATH)

    ua = UserAgent()
    # ips = utils.IP(ua=ua)
    ips = None
    file_list = utils.get_all_file_name_in_dir(INFO_PATH)

    # for i in range(1, 24000):
    #     start_single_imgs('http://www.happyjuzi.com/star-picture-{}/'.format(i), ua, ips)

    for i in range(1, 9):
        thread = mySingleThread(threadID=i, step=3000,
                                ua=ua, ips=ips, file_list=file_list)
        thread.start()

    # start_single_imgs(
    #     'http://www.happyjuzi.com/star-picture-{}/'.format(103), ua, ips, file_list)
