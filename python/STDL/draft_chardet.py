# -*- coding: utf-8 -*-
import chardet

chardet.detect(b'Hello, world!')


chardet.detect('离离原上草，一岁一枯荣'.encode('gbk'))


chardet.detect('离离原上草，一岁一枯荣'.encode('utf-8'))


chardet.detect('最新の主要ニュース'.encode('euc-jp'))
