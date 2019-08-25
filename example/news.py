#!/usr/bin/env python3

from __future__ import print_function
import sys
sys.path.insert(0, '..')
import csv
import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
import numpy as np
import random

from google.protobuf import text_format

import feature_pb2 as feature
from analyzer import Analyzer
from transform import transform
from validator import validate
from stat_collector import StatCollector, collect_stats

valid_cates = ['ent', 'sport', 'finacial', 'society', 'mil', 'stock', 'tech']
def define_schema():
    schema = feature.Schema()
    schema.version = 1

    # age
    validator = feature.Validator()
    validator.float_min = 0.0
    validator.float_max = 100.0
    new_feature = schema.feature.add(type=feature.Feature.FLOAT, name='age',
                                     validator=validator)
    new_feature.desc = "age of users"
    trans = new_feature.transformer.add(
            discretize=feature.Discretize(discretize_level=5))

    # gender
    validator = feature.Validator()
    validator.one_of_string.extend(['male', 'female'])
    new_feature = schema.feature.add(type=feature.Feature.STRING, name='gender',
                                     validator=validator)
    new_feature.desc = "user gender"

    cate_validator = feature.Validator()
    cate_validator.one_of_string.extend(valid_cates)

    # user cates
    new_feature = schema.feature.add(type=feature.Feature.STRING_LIST,
                                     name='ucate',
                                     validator=cate_validator)
    new_feature.desc = 'tags preference in user profile'
    build_vocab = feature.BuildVocabAndConvertToId()
    build_vocab.vocab_file_name = 'cate_to_id.txt'
    new_feature.transformer.add(build_vocab_and_convert_to_id=build_vocab)

    # news cates
    new_feature = schema.feature.add(type=feature.Feature.STRING_LIST,
                                     name='ncate',
                                     validator=cate_validator)
    new_feature.desc = 'news tags'
    build_vocab = feature.BuildVocabAndConvertToId()
    build_vocab.init_vocab_file = 'cate_to_id.txt'  # use the same one
    new_feature.transformer.add(build_vocab_and_convert_to_id=build_vocab)
    new_feature.dependency_feature.append('ucate')  # depend on vocab

    # cross feature
    features_to_cross = sorted(['ucate', 'ncate'])
    new_feature = schema.feature.add(type=feature.Feature.CROSS,
                                     name='_X_'.join(features_to_cross))
    new_feature.dependency_feature.extend(features_to_cross)
    new_feature.desc = 'cross ucate and ncate'
    new_feature.transformer.add(hash_to_interval=feature.HashToInterval(modulus=11))

    return schema


def load_samples():
    samples = []
    for i in range(100):
        age = feature.Feature(type=feature.Feature.FLOAT)
        age.float_value = np.round(np.random.uniform(20, 70), 2)
        gender = feature.Feature(type=feature.Feature.STRING)
        gender.string_value = random.choice(['male', 'female'])
        ucate = feature.Feature(type=feature.Feature.STRING_LIST)
        ucate.string_list_value.value.extend(random.sample(valid_cates, 2)) # 2 user cates
        ncate = feature.Feature(type=feature.Feature.STRING_LIST)
        ncate.string_list_value.value.extend(random.sample(valid_cates, 1))  # 1 news cates
        features = {'age': age, 'gender': gender, 'ucate': ucate, 'ncate': ncate}
        samples.append(feature.Sample(feature=features))
    return samples

if __name__ == '__main__':
    schema = define_schema()
    with open('original_newsschema.txt', 'w') as fout:
        fout.write(text_format.MessageToString(schema, as_utf8=True))
    data = load_samples()

    schema_analyzer = Analyzer(schema)
    schema_analyzer.topo_sort()
    print('features in topological order:',
          schema_analyzer.topo_sorted_feature)
    schema_analyzer.collect_needed_stats_type()
    print('statistics to be collected:', schema_analyzer.stats_to_collect)

    collector = StatCollector()

    collect_stats(data, schema, collector,
            schema_analyzer.stats_to_collect,
            schema_analyzer.corresponding_transformers)
    with open('newsschema.txt', 'w') as fout:
        fout.write(text_format.MessageToString(schema, as_utf8=True))

    with open('original_news_sample_0.txt', 'w') as fout:
        fout.write(text_format.MessageToString(data[0], as_utf8=True))
    # validate/transform data
    for sample in data:
        validate(sample, schema, feature.Validator.BEFORE_TRANSFORM)
        transform(sample, schema, collector.collected_stats)
        validate(sample, schema, feature.Validator.AFTER_TRANSFORM)
    with open('transformed_news_sample_0.txt', 'w') as fout:
        fout.write(text_format.MessageToString(data[0], as_utf8=True))
