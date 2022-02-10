# scrapy

1. link
   * [official site](https://docs.scrapy.org/en/latest/index.html)

## scrapy shell

1. enter shell
   * cmd: `scrapy shell "http://quotes.toscrape.com/page/1/"`
   * bash: `scrapy shell 'http://quotes.toscrape.com/page/1/'`

```python
view(response)
tmp1 = response.xpath('//div[@class="quote"]')
title = tmp1.xpath('span[@class="text"]/text()').extract()
author = tmp1.xpath('span/small[@class="author"]/text()').extract()
tag = [x.xpath('div[@class="tags"]/a[@class="tag"]/text()').extract() for x in tmp1]
next_page = response.xpath('//li[@class="next"]/a/@href').extract_first()
```

## tutorial

1. [scrapy tutorial](https://docs.scrapy.org/en/latest/intro/tutorial.html)
2. `scrapy startproject tutorial ws00`
   * `scrapy startproject <project> [project_dir]`
   * `scrapy startproject --help`
3. `ws00/tutorial/spiders/quotes_spider.py`
4. `scrapy crawl quotes`
5. scrapy shell
6. `scrapy.follow()`, `scrapy.Requests()`, `response.urljoin()`, `.css.get()`

```bash
scrapy runspider main01.py -o tbd01.json

scrapy crawl quotes -o tbd01.json
scrapy crawl quotes -o tbd01.jl #json lines format
```

## commands

1. get help
   * `scrapy -h`
   * `scrapy xxx -h`
2. `scrapy startproject xxx yyy`
3. `scrapy genspider -l`

## spider

TODO

```Python
'''
scrapy.Spider
    .name(str): unique
    .allowed_domains(list,str)
    .start_urls(list,str)
    .custom-settings(dict)
    .crawler: set by the from_crawler() classmethod
    .settings
    .logger
    .start_requests()
    .parse()
'''
```

## TODO

1. schedule.json
2. downloader middleware
