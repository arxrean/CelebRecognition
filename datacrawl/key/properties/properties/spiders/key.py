import scrapy
from properties.items import PropertiesItem
import pdb
import re
import json


class KeySpider(scrapy.Spider):
    name = 'key'
    allowed_domain = ['http://www.happyjuzi.com']
    # start_urls = []
    # existed_stars = json.load(
    #     open('/pic/code/famous/juzi1ip/properties/item.json'))
    # existed_stars = [str(star['id']).strip() for star in existed_stars]
    all_stars = [line.strip() for line in open(
        '/pic/code/famous/juzi1ip/file_id.txt').readlines()]

    # pdb.set_trace()
    # for star in all_stars:
    #     if not star.strip() in existed_stars:
    #       start_urls.append('http://www.happyjuzi.com/star-{}'.format(star))

    # start_urls = ['http://www.happyjuzi.com/star-ku-0-0-0-0-0-0-0/']
    start_urls = [
        'http://www.happyjuzi.com/star-{}/'.format(id) for id in all_stars]

    # pdb.set_trace()

    def parse(self, response):
        try:
            item = PropertiesItem()
            item['id'] = re.findall("\d+", response.request.url)[0]

            # pdb.set_trace()
            img_src = response.xpath(
                '//img[@class="i_starimg_starindex"]/@src').extract_first()
            if '?' in img_src:
                img_src = img_src[:img_src.index('?')]

            item['key_src'] = img_src
            print('success:{}'.format(item['id']))

            return item

        except:
            item = PropertiesItem()
            item['id'] = re.findall("\d+", response.request.url)[0]

            return item
