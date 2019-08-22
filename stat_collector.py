from collections import defaultdict
from collections import Counter
from config import OOV_SYMBOL
import numpy as np
import feature_pb2 as feature


class StatCollector:
    ''' collect stats '''
    # TODO add missing_ratio collector

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
            if stopwords is not None and word in stopwords:
                del complete_vocab[word]

        if limit > 0:
            vocab = complete_vocab.most_common(limit)
        else:
            vocab = complete_vocab.most_common()
        if savefile:
            with open(savefile, 'w') as fout:
                for w, c in vocab:
                    fout.write(f'{w}\t{c}\n')
        word2idx = {OOV_SYMBOL: 0}
        for idx, (w, _) in enumerate(vocab, start=1):
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
        mean, std = np.mean(values), np.std(values)
        self.mean[feature_name] = mean
        self.std[feature_name] = std
        return mean, std

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
            # print('sorted values', values)
            quantiles = np.linspace(0, 1, level)
            boundaries = []
            for q in quantiles:
                boundaries.append(_get_score(q))
        return np.asarray(boundaries)

def collect_stats(samples, schema, collector, needed_stats, transformers):
    ''' collect needed stats and update corresponding transformers '''
    for sample in samples:
        for feat_name, feat_value in sample.feature.items():
            if feat_name not in needed_stats:
                continue
            for stat in needed_stats[feat_name]:
                if stat == 'bucket_info':
                    collector.collect_value(feat_name, feat_value.float_value)
                elif stat == 'vocab':
                    if feat_value.HasField('string_list_value'):
                        collector.collect_vocab(
                                feat_name,
                                list(feat_value.string_list_value.value))
                    elif feat_value.HasField('string_value'):
                        collector.collect_vocab(feat_name,
                                                feat_value.string_value)
                    else:
                        raise ValueError(f'no string_value/string_list_value in'
                                         ' feature {feat_name}')
                # TODO collect other stats
    for feat_name in needed_stats:
        for stat in needed_stats[feat_name]:
            if stat == 'bucket_info':
                trans = transformers[feat_name][stat]
                boundaries = collector.build_bucket(
                        feat_name,
                        trans.discretize.discretize_level,
                        trans.discretize.method)
                trans.discretize.boundaries.extend(boundaries)
            elif stat == 'vocab':
                trans = transformers[feat_name][stat]
                conf = trans.build_vocab_and_convert_to_id
                collector.build_vocab(feat_name,
                                      savefile=conf.vocab_file_name,
                                      minfreq=conf.min_freq,
                                      limit=conf.max_vocab_num)
            # TODO implement other stats calculations


