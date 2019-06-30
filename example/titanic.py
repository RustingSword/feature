#!/usr/bin/env python3
#!coding: utf8
''' kaggle titanic data '''

from __future__ import print_function
import sys
sys.path.insert(0, '..')
import csv
import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

from google.protobuf import text_format

import feature_pb2 as feature
from analyzer import Analyzer
from transform import transform
from validator import validate
from stat_collector import StatCollector, collect_stats

def define_schema():
    ''' there are 11 features:
    1. PassengerId: int
    2. Pclass: int
    3. Name: string
    4. Sex: string
    5. Age: int/float
    6. SibSp: int
    7. Parch: int
    8. Ticket: string
    9. Fare: float
    10. Cabin: string
    11. Embarked: string
    use several to demonstrate usage
    '''
    schema = feature.Schema()
    schema.version = 1

    # age
    validator = feature.Validator()
    validator.float_min = 0.0
    validator.float_max = 80.0
    new_feature = schema.feature.add(type=feature.Feature.FLOAT, name='age',
                                     validator=validator)
    new_feature.desc = "age of passangers"
    trans = new_feature.transformer.add(
            discretize=feature.Discretize(discretize_level=5))

    # sex
    validator = feature.Validator()
    validator.one_of_string.extend(['male', 'female'])
    new_feature = schema.feature.add(type=feature.Feature.STRING, name='sex',
                                     validator=validator)
    new_feature.desc = "sex of passangers"
    new_feature.is_intermediate_feature = True

    # pclass
    new_feature = schema.feature.add(type=feature.Feature.INT, name='pclass')
    new_feature.desc = 'class of ticket'

    # fare
    new_feature = schema.feature.add(type=feature.Feature.FLOAT, name='fare')
    new_feature.desc = 'ticket fare'

    # cross feature
    features_to_cross = sorted(['sex', 'pclass'])
    new_feature = schema.feature.add(type=feature.Feature.CROSS,
                                     name='_X_'.join(features_to_cross))
    new_feature.dependency_feature.extend(features_to_cross)
    new_feature.desc = 'cross sex and pclass'

    return schema


def load_samples(filename):
    samples = []
    with open(filename) as fin:
        csv_reader = csv.DictReader(fin)
        for sample in csv_reader:
            age = feature.Feature(type=feature.Feature.FLOAT)
            try:
                age.float_value = float(sample['Age'])
            except ValueError as e:
                age.float_value = 0.0
            sex = feature.Feature(type=feature.Feature.STRING)
            sex.string_value = sample['Sex']
            pclass = feature.Feature(type=feature.Feature.INT)
            pclass.int_value = int(sample['Pclass'])
            fare = feature.Feature(type=feature.Feature.FLOAT)
            try:
                fare.float_value = float(sample['Fare'])
            except ValueError as e:
                fare.float_value = 0.0
            features = {'age': age, 'sex': sex, 'pclass': pclass, 'fare': fare}
            samples.append(feature.Sample(feature=features))
    return samples

if __name__ == '__main__':
    # TODO simplify the whold process, it's still cumbersome right now
    schema = define_schema()
    train_data = load_samples('train.csv')
    test_data = load_samples('test.csv')

    schema_analyzer = Analyzer(schema)
    schema_analyzer.topo_sort()
    print('features in topological order:',
          schema_analyzer.topo_sorted_feature)
    schema_analyzer.collect_needed_stats_type()
    print('statistics to be collected:', schema_analyzer.stats_to_collect)

    collector = StatCollector()

    collect_stats(train_data, schema, collector,
            schema_analyzer.stats_to_collect,
            schema_analyzer.corresponding_transformers)
    with open('schema.txt', 'w') as fout:
        fout.write(text_format.MessageToString(schema, as_utf8=True))

    # validate/transform data
    for sample in train_data:
        validate(sample, schema, feature.Validator.BEFORE_TRANSFORM)
        transform(sample, schema)
        validate(sample, schema, feature.Validator.AFTER_TRANSFORM)
    with open('transformed_train_sample_0.txt', 'w') as fout:
        fout.write(text_format.MessageToString(train_data[0], as_utf8=True))
