import utils
from urllib.parse import urljoin
import urllib.request
from bs4 import BeautifulSoup
import random
import time
import pdb


def parse_single_man(soup):
    content = soup
    result = dict()

    parse_famous_name(content, result)
    parse_famous_basic_info(content, result)
    parse_description(content, result)
    parse_best_movies(content, result)

    return result


def parse_famous_name(content, result):
    name = content.find('strong', attrs={'class': 'name_starindex'})
    e_name = content.find('strong', attrs={'class': 'ename_starindex'})
    if name:
        result['name'] = name.get_text().strip()
    elif e_name:
        result['name'] = e_name.get_text().strip()
    else:
        result['name'] = 'None'


def parse_famous_basic_info(content, result):
    result['type'] = ''

    start_type = content.find('p', attrs={'class': 'p_type_starindex'})
    if start_type:
        for a in start_type.find_all('a'):
            result['type'] += ' {}'.format(a.get_text().strip())
        result['type'] = result['type'].strip()

    address = content.find('p', attrs={'class': 'area_starindex'})
    if address:
        result['address'] = address.a.get_text().strip()


def parse_description(content, result):
    des = content.find('p', attrs={'class': 'p_desc_starindex'})
    result['des'] = des.get_text().strip()


def parse_best_movies(content, result):
    best_movies = content.find('div', attrs={'class': 'reels_starindex'})
    result['best_movies'] = []
    if best_movies:
        for item in best_movies.find_all(name='p', attrs={'class': 'name_year_starindex'}):
            movie = item.get_text().strip()
            result['best_movies'].append(movie)
            if len(result['best_movies']) == 5:
                break


def parse_imgs(soup, result, ua, ips):
    parse_imgs_in_one_page(soup, result, ua, ips)


def parse_imgs_in_one_page(soup, result, ua, ips):
    result['img_path'] = []
    content = soup.find('div', attrs={'class': 'center_hotstar'})
    if content:
        for pic in content.find_all('div', attrs={'class': 'star-pic'}):
            result['img_path'].append(pic['data-src'])

    # if next page
    paginator = content.find('div', attrs={'class': 'tcdPageCode'})
    if paginator and paginator.find('span', attrs={'class': 'current'}):
        while True:
            next_page = paginator.find(
                'span', attrs={'class': 'current'}).next_sibling
            if next_page and next_page.get('class')[0] == 'num':
                time.sleep(random.uniform(3, 6))

                page_num = next_page.get_text().strip()
                next_page_url = 'http://www.happyjuzi.com/star-picture-{}/p{}.html'.format(
                    result['id'], page_num)
                response = utils.custom_get(next_page_url, ua, ips, False)
                html = response.read()

                # parse basic info
                soup = BeautifulSoup(html, "html.parser")
                content = soup.find('div', attrs={'class': 'center_hotstar'})
                if content:
                    for pic in content.find_all('div', attrs={'class': 'star-pic'}):
                        result['img_path'].append(pic['data-src'])
                    if len(result['img_path'])>60:
                        break

            else:
                break

    # for item in content.ul.find_all(name='li'):
    #     img_tag = item.find('div', attrs={'class': 'cover'}).a.img
    #     result['img_path'].append(img_tag['src'].strip())

    # # get imgs in 2end page if has
    # paginator = content.find('div', attrs={'class': 'paginator'})
    # if paginator:
    #     next_page = paginator.find('a')
    #     if next_page.get_text().strip() == '2':
    #         next_page = next_page['href']
    #         time.sleep(random.uniform(3, 6))
    #         response = utils.custom_get(next_page, ua)
    #         body = response.read()

    #         soup = BeautifulSoup(body, "html.parser")
    #         content = soup.find('div', id='content')
    #         for item in content.ul.find_all(name='li'):
    #             img_tag = item.find('div', attrs={'class': 'cover'}).a.img
    #             result['img_path'].append(img_tag['src'].strip())
