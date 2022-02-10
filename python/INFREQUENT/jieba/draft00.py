import os
import jieba
import jieba.posseg as pseg


hf_data = lambda *x: os.path.join('data', *x)


print('/ '.join(jieba.cut('我来到北京清华大学', cut_all=True)))
print('/ '.join(jieba.cut('我来到北京清华大学')))
print('/ '.join(jieba.cut('他来到了网易行研大厦')))
print('/ '.join(jieba.cut_for_search('小明硕士毕业于中国科学院计算所，后在日本京都大学深造')))


jieba.load_userdict(hf_data('userdict.txt'))
jieba.add_word('石墨烯')
jieba.add_word('凱特琳')
jieba.del_word('自定义词')

test_sent = (
    "李小福是创新办主任也是云计算方面的专家; 什么是八一双鹿\n"
    "例如我输入一个带“韩玉赏鉴”的标题，在自定义词库中也增加了此词为N类\n"
    "「台中」正確應該不會被切開。mac上可分出「石墨烯」；此時又可以分出來凱特琳了。"
)
print('/ '.join(jieba.cut(test_sent)))
for w in pseg.cut(test_sent):
    print(w.word + w.flag + '/ ', end='')

print('/ '.join(jieba.cut('easy_install is great')))
print('/'.join(jieba.cut('python 的正则表达式是好用的')))

# test frequency tune
testlist = [
    ('今天天气不错', ('今天', '天气')),
    ('如果放到post中将出错。', ('中', '将')),
    ('我们中出了一个叛徒', ('中', '出')),
]
for sent, seg in testlist:
    print('/'.join(jieba.cut(sent, HMM=False)))
    word = ''.join(seg)
    print('%s Before: %s, After: %s' % (word, jieba.get_FREQ(word), jieba.suggest_freq(seg, True)))
    print('/'.join(jieba.cut(sent, HMM=False)))
    print("-"*40)
