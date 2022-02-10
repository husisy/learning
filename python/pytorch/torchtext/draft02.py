# TODO WMT2014 datasets
# src_tokenizer = torchtext.data.utils.get_tokenizer('spacy', language='de_core_news_sm')
# dst_tokenizer = torchtext.data.utils.get_tokenizer('spacy', language='en_core_web_sm')

# BOS_WORD = '<bos>'
# EOS_WORD = '<eos>'
# PAD_WORD = "<pad>"
# SRC = torchtext.legacy.data.Field(tokenize=src_tokenizer, init_token=BOS_WORD, eos_token=EOS_WORD, pad_token=PAD_WORD)
# TGT = torchtext.legacy.data.Field(tokenize=dst_tokenizer, init_token=BOS_WORD, eos_token=EOS_WORD, pad_token=PAD_WORD)
# # torchtext.utils.download_from_url('https://drive.google.com/uc?export=download&id=0B_bZck-ksdkpM25jRUN2X2UxMm8', 'wmt16_en_de.tar.gz')
# z0 = torchtext.legacy.datasets.WMT14.splits(root=TORCHTEXT_ROOT, exts=('.en','.de'), fields=(SRC,TGT))
