'''
reference: https://docs.scrapy.org/en/latest/intro/overview.html

cmd line:
    scrapy runspider main01.py -o tbd01.json
'''
import scrapy

class QuotesSpider(scrapy.Spider):
    name = 'quotes'
    start_urls = ['http://quotes.toscrape.com/tag/humor/']

    def parse(self, response):
        for x in response.css('div.quote'):
            yield {
                'text': x.css('span.text::text').extract_first(),
                'author': x.xpath('span/small/text()').extract_first(),
            }
        next_page = response.css('li.next a::attr("href")').extract_first()
        if next_page is not None:
            yield response.follow(next_page, self.parse)

