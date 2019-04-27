from stat_collector import StatCollector
import feature_pb2 as feature
import random
import numpy as np
import sys


collector = StatCollector()

values = np.random.normal(0, 1, 1000)
for v in values:
    collector.collect_value('random', v)

buckets = collector.build_bucket('random')
print('equal number (quantile):', buckets)
buckets = collector.build_bucket('random', method=feature.Discretize.EQUAL_SPACING)
print('equal spacing:', buckets)


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
