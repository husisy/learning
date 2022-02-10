# fairseq

1. link
   * [github](https://github.com/pytorch/fairseq)
   * [documentation](https://fairseq.readthedocs.io/en/latest/)
2. Byte Pair Encoding (BPE) vocabulary [arxiv-link](https://arxiv.org/abs/1508.07909)
3. 输出格式
   * `S-0`：原始输入句子
   * `W-0`
   * `H-0`: the hypothesis along with an average log-likelihood
   * `D-0`: the detokenized hypothesis
   * `P-0`: the positional score per token position, including the end-of-sentence marker
   * `T-0`: the reference target
   * `A-0`: alignment info
   * `E-0`: the history of generation steps

```bash
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install .
# python setup.py build_ext --inplace

conda install -c conda-forge sacremoses
# pip install sacremoses
pip install subword-nmt

cd examples/translation/
bash prepare-iwslt14.sh
TEXT=iwslt14.tokenized.de-en
fairseq-preprocess --source-lang de --target-lang en --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test --destdir data-bin/iwslt14.tokenized.de-en
mkdir -p checkpoints/fconv
fairseq-train data-bin/iwslt14.tokenized.de-en --lr 0.03 --clip-norm 0.1 --dropout 0.2 --max-tokens 4000 --arch fconv_iwslt_de_en --save-dir checkpoints/fconv --optimizer adam
```

data

```bash
curl -O https://dl.fbaipublicfiles.com/fairseq/models/wmt14.v2.en-fr.fconv-py.tar.bz2
tar xvjf wmt14.v2.en-fr.fconv-py.tar.bz2
MODEL_DIR=data/wmt14.en-fr.fconv-py
fairseq-interactive --path $MODEL_DIR/model.pt $MODEL_DIR --beam 5 --source-lang en --target-lang fr --tokenizer moses --bpe subword_nmt --bpe-codes $MODEL_DIR/bpecodes #loss could be lower than 10
```

```bash
from sacremoses import MosesTokenizer, MosesDetokenizer
mt = MosesTokenizer(lang='en')
text = u'This, is a sentence with weird\xbb symbols\u2026 appearing everywhere\xbf'
expected_tokenized = u'This , is a sentence with weird \xbb symbols \u2026 appearing everywhere \xbf'
mt.tokenize(text, return_str=True) == expected_tokenized


mt, md = MosesTokenizer(lang='en'), MosesDetokenizer(lang='en')
sent = "This ain't funny. It's actually hillarious, yet double Ls. | [] < > [ ] & You're gonna shake it off? Don't?"
expected_tokens = [u'This', u'ain', u'&apos;t', u'funny', u'.', u'It', u'&apos;s', u'actually', u'hillarious', u',', u'yet', u'double', u'Ls', u'.', u'&#124;', u'&#91;', u'&#93;', u'&lt;', u'&gt;', u'&#91;', u'&#93;', u'&amp;', u'You', u'&apos;re', u'gonna', u'shake', u'it', u'off', u'?', u'Don', u'&apos;t', u'?']
expected_detokens = "This ain't funny. It's actually hillarious, yet double Ls. | [] < > [] & You're gonna shake it off? Don't?"
mt.tokenize(sent) == expected_tokens
md.detokenize(expected_tokens) == expected_detokens
```
