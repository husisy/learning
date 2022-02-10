# sentencepiece

1. link
   * [github](https://github.com/google/sentencepiece)
   * [medium/introduction](https://jacky2wong.medium.com/understanding-sentencepiece-under-standing-sentence-piece-ac8da59f6b08)
   * [github/jupyter-tutorial](https://github.com/google/sentencepiece/blob/master/python/sentencepiece_python_module_example.ipynb)
2. installation
   * `conda install -c conda-forge sentencepiece`
   * `pip install sentencepiece`
3. special symbols
   * BERT special symbols `[SEP]`, `[CLS]`
   * user defined symbols: Always treated as one token in any context. These symbols can appear in the input sentence
   * control symbol: We only reserve ids for these tokens. Even if these tokens appear in the input text, they are not handled as one token. User needs to insert ids explicitly after encoding
