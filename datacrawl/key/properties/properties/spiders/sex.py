import scrapy
from properties.items import PropertiesItem
import pdb
import re
import json


class SexSpider(scrapy.Spider):
    name = 'sex'
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
    start_urls = ['http://www.happyjuzi.com/star-ku-0-0-0-0-0-0-0/'] + \
        ['http://www.happyjuzi.com/star-ku-0-0-0-0-0-0-0/p{}.html'.format(
            i) for i in range(2, 1431)]

    def parse(self, response):
        ids = response.xpath(
            '//div[@class="star_hotstar"]//a[@class="name_hotstar"]/@href').extract()
        # pdb.set_trace()
        return_items=[]
        # pdb.set_trace()
        for id in ids:
            try:
                item = PropertiesItem()
                star_id = re.findall("\d+", id)[0]

                sex = response.xpath(
                    '//div[@class="star_hotstar"]//a[@class="name_hotstar" and @href="http://www.happyjuzi.com/star-{}/"]/following-sibling::a[1]/text()'.format(star_id)).extract_first()

                item['id'] = star_id
                if sex == '男':
                    item['sex'] = 1
                elif sex == '女':
                    item['sex'] = 0
                else:
                    item['sex'] = -1

                print('id:{}'.format(star_id))
                return_items.append(item)

            except:
            	print('error')

        return return_items
