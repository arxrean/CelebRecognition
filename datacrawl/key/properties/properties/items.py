# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://doc.scrapy.org/en/latest/topics/items.html

import scrapy


class PropertiesItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    id = scrapy.Field()
    name = scrapy.Field()
    e_name = scrapy.Field()
    star_type = scrapy.Field()
    star_other = scrapy.Field()
    area = scrapy.Field()
    desc = scrapy.Field()
    reels = scrapy.Field()
    sex = scrapy.Field()
    key_src=scrapy.Field()
