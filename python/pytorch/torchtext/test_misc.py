import numpy as np
import torch
import torchtext

import collections


def ngrams_iterator(token_list, ngrams):
    ret = [(x,) for x in token_list]
    for n in range(2, ngrams+1):
        ret += list(zip(*[token_list[i:] for i in range(n)]))
    ret = collections.Counter(ret)
    return ret


def my_bleu_score(candidate_corpus, references_corpus, max_n=4, weights=[0.25] * 4):
    assert max_n == len(weights)
    assert len(candidate_corpus) == len(references_corpus)
    clipped_counts = np.zeros(max_n, dtype=np.int64)
    total_counts = np.zeros(max_n, dtype=np.int64)
    weights = np.array(weights)
    candidate_len = 0.0
    refs_len = 0.0

    for (candidate, refs) in zip(candidate_corpus, references_corpus):
        candidate_len += len(candidate)

        # Get the length of the reference that's closest in length to the candidate
        tmp0 = [float(len(ref)) for ref in refs]
        refs_len += min(tmp0, key=lambda x: abs(len(candidate)-x))

        reference_counters = ngrams_iterator(refs[0], max_n)
        for ref in refs[1:]:
            reference_counters = reference_counters | ngrams_iterator(ref, max_n)

        candidate_counter = ngrams_iterator(candidate, max_n)

        clipped_counter = candidate_counter & reference_counters

        for ngram in clipped_counter:
            clipped_counts[len(ngram) - 1] += clipped_counter[ngram]

        for x in range(max_n):
            total_counts[x] += max(0, len(candidate)-x)

    if min(clipped_counts) == 0:
        ret = 0.0
    else:
        score = np.exp(np.sum(weights * np.log(clipped_counts / total_counts)))
        bp = np.exp(min(1 - refs_len / candidate_len, 0))
        ret = (bp * score).item()
    return ret

def test_bleu_source():
    tmp0 = ['My full pytorch test', 'Another Sentence']
    ex0_candidate = [x.split(' ') for x in tmp0]
    tmp1 = [
        ('My full pytorch test', 'Completely Different'),
        ('No Match',),
    ]
    ex0_reference = [[y.split(' ') for y in x] for x in tmp1]

    tmp0 = ['It is a guide to action which ensures that the military always obeys the commands of the party']
    ex1_candidate = [x.split(' ') for x in tmp0]
    tmp1 = [[
        'It is a guide to action that ensures that the military will forever heed Party commands',
        'It is the guiding principle which guarantees the military forces always being under the command of the Party',
        'It is the practical guide for the army always to heed the directions of the party',
    ]]
    ex1_reference = [[y.split(' ') for y in x] for x in tmp1]

    for x,y in [(ex0_candidate,ex0_reference), (ex1_candidate,ex1_reference)]:
        ret_ = torchtext.data.metrics.bleu_score(x, y)
        ret0 = my_bleu_score(x, y)
        assert abs(ret_ - ret0) < 1e-5
