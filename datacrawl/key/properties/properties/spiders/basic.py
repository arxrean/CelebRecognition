import scrapy
from properties.items import PropertiesItem
import pdb
import re
import json


class BasicSpider(scrapy.Spider):
    name = 'basic'
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
    #     	start_urls.append('http://www.happyjuzi.com/star-{}'.format(star))

    # start_urls = ['http://www.happyjuzi.com/star-3563/']
    start_urls=['http://www.happyjuzi.com/star-{}'.format(star) for star in all_stars]

    def parse(self, response):
        item = PropertiesItem()
        try:
            item['id'] = re.findall("\d+", response.request.url)[0]
            item['name'] = response.xpath(
                '//strong[contains(@class,"name")][1]/text()').extract()[0]
            item['e_name'] = response.xpath(
                '//strong[contains(@class,"ename_starindex")]/text()').extract()
            if len(item['e_name']) > 0:
                item['e_name'] = item['e_name'][0]
            item['star_type'] = response.xpath(
                'string(//p[contains(@class,"p_type_starindex")][1])').extract_first().strip()
            item['star_other'] = response.xpath(
                'string(//p[contains(@class,"p_other_starindex")][1])').extract_first().strip()
            item['area'] = response.xpath(
                'string(//p[contains(@class,"area_starindex")][1])').extract_first().strip()
            item['desc'] = response.xpath(
                'string(//p[contains(@class,"p_desc_starindex")][1])').extract_first().strip()
            item['reels'] = ' '.join(response.xpath(
                'string(//div[contains(@class,"reels_starindex")][1])').extract_first().strip().replace('\n', '').split())

            print('{}:{}'.format(item['id'], item['name']), flush=True)

        except Exception as e:
            # print(e)
            item = PropertiesItem()
            item['id'] = re.findall("\d+", response.request.url)[0]
            print('error:{}'.format(item['id']), flush=True)

        return item
