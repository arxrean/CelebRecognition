import re
# from fake_useragent import UserAgent
import urllib.request
import time
import random
# from bs4 import BeautifulSoup
import numpy as np
import requests
import threading
import datetime
from tqdm import tqdm
import pandas as pd
import pdb
import os
import json


def get_only_chinese(text):
	return re.sub("[A-Za-z0-9\!\%\[\]\,\。]", "", text)


def remove_all_punctuation(text):
	return re.sub(r'[^\w\s]', '', text).strip()


def get_only_numbers(text):
	return re.findall("\d+", text)


def custom_get(url, ua, ips, use_agent=True):
	# headers = {
	#     'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36'}
	headers = {'User-Agent': ua.random}
	request = urllib.request.Request(url=url, headers=headers)

	if use_agent:
		random_ip = ips.get_random_ip()
		httpproxy_handler = urllib.request.ProxyHandler({
			'http': 'http://{}'.format(random_ip),
			'https': 'https://{}'.format(random_ip)
		})
		opener = urllib.request.build_opener(httpproxy_handler)
		# urllib.request.install_opener(opener)
		time.sleep(random.uniform(3, 6))

		response = opener.open(request, timeout=3)

		return response
	else:
		response = urllib.request.urlopen(request)
		time.sleep(random.uniform(3, 6))

		return response


def get_all_file_name_in_dir(dir):
	file_list = []
	for file in os.listdir(dir):
		file_list.append(file)

	return file_list


class IP:
	def __init__(self, ua):
		self.headers = {'User-Agent': ua.random}
		self.ip_list = self.get_ip_list('https://www.xicidaili.com/nn/')
		print('成功获取ip池,共{}个ip'.format(len(self.ip_list)))

	def get_ip_list(self, ip_pool):
		request = urllib.request.Request(url=ip_pool, headers=self.headers)
		response = urllib.request.urlopen(request)
		time.sleep(random.uniform(3, 6))

		ip_list = []
		html = response.read()
		soup = BeautifulSoup(html, "html.parser")
		ip_table = soup.find('table', id='ip_list')
		for tr in tqdm(ip_table.find_all('tr')):
			ths = tr.find_all('td')
			if len(ths) > 0:
				ip = '{}:{}'.format(
					ths[1].get_text().strip(), ths[2].get_text().strip())
				if self.checkip(ip):
					ip_list.append(ip)

		return ip_list

	def get_random_ip(self):
		return np.random.choice(self.ip_list, 1)[0]

	def refresh_ips(self):
		self.ip_list = self.get_ip_list('https://www.xicidaili.com/nn/')
		print('成功刷新ip池,共{}个ip'.format(len(self.ip_list)))

	def checkip(self, ip):
		proxies = {"http": "http://"+ip, "https": "https://"+ip}  # 代理ip
		try:
			response = requests.get(
				url='http://www.baidu.com', proxies=proxies, headers=self.headers, timeout=3).status_code
			if response == 200:
				return True
			else:
				return False
		except:
			return False


def gen_star_csv(csv_path='./properties/item.json'):
	# data = json.load(open(csv_path))
	# csv_data = []
	# for item in data:
	# 	item_data = []
	# 	item_data.append(item['id'])
	# 	if 'name' in item:
	# 		item_data.append(item['name'])
	# 	else:
	# 		item_data.append('-1')

	# 	if 'e_name' in item:
	# 		item_data.append(item['e_name'])
	# 	else:
	# 		item_data.append('-1')

	# 	if 'star_type' in item:
	# 		item_data.append(item['star_type'])
	# 	else:
	# 		item_data.append('-1')

	# 	if 'star_other' in item:
	# 		item_data.append(item['star_other'])
	# 	else:
	# 		item_data.append('-1')

	# 	if 'area' in item:
	# 		item_data.append(item['area'])
	# 	else:
	# 		item_data.append('-1')

	# 	if 'desc' in item:
	# 		item_data.append(item['desc'])
	# 	else:
	# 		item_data.append('-1')

	# 	if 'reels' in item:
	# 		item_data.append(item['reels'])
	# 	else:
	# 		item_data.append('-1')

	# 	csv_data.append(item_data)

	# csv = pd.DataFrame(csv_data, columns=['id','name','e_name','star_type','star_other','area','desc','reels'])
	# csv.to_csv('star.csv',index=False)

	data = pd.read_csv('star.csv')
	data['sex'] = '-1'
	sex_json = json.load(open('./properties/sex.json'))
	for item in sex_json:
		id = item['id']
		target = data[data['id'] == int(id)]
		if len(target) > 0:
			data.at[target.index[0], 'sex'] = item['sex']

	data.drop([0], inplace=True)
	data.to_csv('star.csv', index=False)


def extract_key_imgs(json_file):
	data = json.load(open(json_file))
	exists=[x[:-4] for x in os.listdir('./save/key')]
	for item in data:
		try:
			id = item['id']
			if id not in exists:
				if 'key_src' in item:
					src = item['key_src']
					urllib.request.urlretrieve(src, './save/key/{}.jpg'.format(id))
					print('success:{}'.format(id))
		except:
			print('err')


if __name__ == '__main__':
	# gen_star_csv()
	extract_key_imgs('./properties/key.json')
