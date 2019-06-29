from collections import defaultdict
from collections import Counter
import numpy as np
import feature_pb2 as feature


class StatCollector:
    ''' collect stats '''

    def __init__(self):
        self.vocab = defaultdict(Counter)
        self.mean = {}
        self.std = {}
        self.value = defaultdict(list)

    def collect_vocab(self, feature_name, words):
        if isinstance(words, list):
            self.vocab[feature_name].update(words)
        elif isinstance(words, str):
            self.vocab[feature_name][words] += 1
        else:
            raise ValueError(f'can only create vocab for string feature, got '
                             '{words} with type {type(words)}')

    def build_vocab(self, feature_name, stopwords=None, savefile=None, limit=0,
                    minfreq=1):
        complete_vocab = self.vocab[feature_name]
        words = list(complete_vocab.keys())
        for word in words:
            if complete_vocab[word] < minfreq:
                del complete_vocab[word]
            if word in stopwords:
                del complete_vocab[word]

        if limit > 0:
            vocab = complete_vocab.most_common(limit)
        else:
            vocab = complete_vocab.most_common()
        if savefile:
            with open(savefile, 'w') as fout:
                for w, c in vocab:
                    fout.write(f'{w}\t{c}\n')
        word2idx = {}
        for idx, (w, _) in enumerate(vocab):
            word2idx[w] = idx
        return word2idx

    def collect_value(self, feature_name, values):
        if isinstance(values, list):
            self.value[feature_name].extend(values)
        else:
            self.value[feature_name].append(values)

    def compute_mean_std(self, feature_name):
        if feature_name not in self.value:
            raise KeyError(f'feature {feature_name} does not exist')
        values = np.asarray(self.value[feature_name])
        return np.mean(values), np.std(values)

    def build_bucket(self, feature_name, level=10,
                     method=feature.Discretize.QUANTILE):
        if feature_name not in self.value:
            raise KeyError(f'feature {feature_name} does not exist')
        values = np.asarray(self.value[feature_name])
        if len(values) < level:
            raise ValueError(f'feature {feature_name} has only {len(values)} '
                             'values, less than discretize level {level}')

        # referred pandas.core.algorithms.quantile
        def _get_score(at):
            if not values.any():
                return np.nan

            idx = at * (len(values) - 1)
            if idx % 1 == 0:
                score = values[int(idx)]
            else:
                lower = values[int(idx)]
                higher = values[int(idx) + 1]
                score = lower + (higher - lower) * (idx % 1)

            return score

        if method == feature.Discretize.EQUALLY_SPACED:
            max_value, min_value = np.max(values), np.min(values)
            boundaries = np.linspace(min_value, max_value, level)
        elif method == feature.Discretize.QUANTILE:
            values = np.sort(values)
            quantiles = np.linspace(0, 1, level)
            boundaries = []
            for q in quantiles:
                boundaries.append(_get_score(q))
        return np.asarray(boundaries)
