import sys
import numpy as np
import feature_pb2 as feature
from stat_collector import StatCollector
from bisect import bisect_left


collector = StatCollector()

values = np.random.normal(0, 1, 1000)
for v in values:
    collector.collect_value('random', v)

float_value = -5.5
buckets = collector.build_bucket('random')
print('equal number (quantile):', buckets)
i = bisect_left(buckets, float_value)
print(f'index of {float_value} is {i}')
buckets = collector.build_bucket('random', method=feature.Discretize.EQUALLY_SPACED)
print('equally spaced:', buckets)
i = bisect_left(buckets, float_value)
print(f'index of {float_value} is {i}')


stopwords = set([
        '=',
        '==',
        '(%s)',
        '#',
        "'''",
        '%',
        '%s',
        '/',
        '//',
        '{',
        '}',
        '[',
        ']'
    ])

with open(sys.argv[1]) as fin:
    for line in fin:
        words = line.strip('\n').split()
        collector.collect_vocab('code', words)

vocab = collector.build_vocab('code', savefile='code_vocab.txt', minfreq=3,
                              stopwords=stopwords, limit=10)

print(vocab)
