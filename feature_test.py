#!/usr/bin/env python3
#!coding: utf8
''' test feature.proto '''

from __future__ import print_function
import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

from google.protobuf import text_format

import feature_pb2 as feature
from analyzer import Analyzer
from transform import transform
from validator import validate
from config import CURRENT_STAGE, MINIMUM_SCHEMA_VERSION

def define_schema():
    ''' define a schema '''
    schema = feature.Schema()
    schema.version = 2

    # age
    validator = feature.Validator()
    validator.int_min = 0
    validator.int_max = 99
    validator.phase = feature.Validator.SKIPPED
    new_feature = schema.feature.add(type=feature.Feature.INT, name='age',
                                     validator=validator,
                                     lifecycle_stage=feature.Feature.ALPHA)
    new_feature.desc = "user's age"
    trans = new_feature.transformer.add()
    trans.clip.int_min_value = 20
    trans.clip.int_max_value = 30

    # sex
    validator = feature.Validator()
    validator.one_of_int.extend([0, 1, 2])
    new_feature = schema.feature.add(type=feature.Feature.INT, name='sex',
                                     validator=validator,
                                     lifecycle_stage=feature.Feature.BETA)
    new_feature.desc = "user's sex"

    # deprecated feature
    new_feature = schema.feature.add(type=feature.Feature.INT,
                                     name='deprecated_feature',
                                     lifecycle_stage=feature.Feature.DEPRECATED)
    new_feature.desc = 'deprecated feature example'

    # string feature
    validator = feature.Validator()
    validator.max_missing_ratio = 0.1
    validator.one_of_string.extend(['北京', '上海', '广州', '深圳'])
    new_feature = schema.feature.add(type=feature.Feature.STRING, name='city',
                                     validator=validator)
    new_feature.desc = "user's city"
    hash_method = feature.HashToInteger(hash_function=feature.HashToInteger.XXHASH)
    trans = new_feature.transformer.add(hash_to_integer=hash_method)

    # dense (embedding)
    validator = feature.Validator()
    validator.dense_value_dim = 3
    new_feature = schema.feature.add(type=feature.Feature.DENSE, name='embedding',
                                     validator=validator,
                                     lifecycle_stage=feature.Feature.PRODUCTION)
    new_feature.desc = 'some embedding feature'

    # dense 2
    new_feature = schema.feature.add(type=feature.Feature.DENSE, name='random')
    new_feature.desc = 'random dense feature'

    # sparse
    new_feature = schema.feature.add(type=feature.Feature.SPARSE, name='ntags')
    new_feature.desc = 'tags of news'

    # sparse
    new_feature = schema.feature.add(type=feature.Feature.SPARSE, name='utags')
    new_feature.desc = 'tags of user'

    # float
    validator = feature.Validator(float_min=0, float_max=1)
    new_feature = schema.feature.add(type=feature.Feature.FLOAT, name='ctr',
                                     validator=validator)
    new_feature.desc = 'ctr of news'

    # cross feature
    validator = feature.Validator(int_min=0, int_max=2000,
                                  phase=feature.Validator.AFTER_TRANSFORM)
    features_to_cross = sorted(['utags', 'ntags', 'intermediate'])
    new_feature = schema.feature.add(type=feature.Feature.CROSS,
                                     name='_X_'.join(features_to_cross),
                                     validator=validator)
    new_feature.dependency_feature.extend(features_to_cross)
    new_feature.desc = 'a cross feature example'

    # cross 2
    features_to_cross = sorted(['embedding', 'random'])
    new_feature = schema.feature.add(type=feature.Feature.CROSS,
                                     name='_X_'.join(features_to_cross),
                                     validator=validator)
    new_feature.dependency_feature.extend(features_to_cross)
    new_feature.cross_value.cross_method = feature.CrossDomain.COSINE_SIMILARITY
    new_feature.desc = 'another cross feature example'

    # sparse
    new_feature = schema.feature.add(type=feature.Feature.SPARSE, name='intermediate',
                                     is_intermediate_feature=True)
    new_feature.desc = 'some intermediate feature'

    return schema


def generate_feature():
    ''' Generate a sample. In model training/serving pipeline, this comes from
    log/online data '''
    # age
    age = feature.Feature(type=feature.Feature.INT)
    age.int_value = 50
    # sex
    sex = feature.Feature(type=feature.Feature.INT)
    sex.int_value = 2
    # dense
    tensor = feature.Feature(type=feature.Feature.DENSE)
    tensor.dense_value.value.extend([1.0, 2.0, 3.0])
    # sparse
    utags = feature.Feature(type=feature.Feature.SPARSE)
    utags.sparse_value.value.extend([10, 20, 30, 40])
    ntags = feature.Feature(type=feature.Feature.SPARSE)
    ntags.sparse_value.value.extend([1, 2, 3, 4])
    # city
    city = feature.Feature(type=feature.Feature.STRING)
    city.string_value = '北京'
    # ctr
    ctr = feature.Feature(type=feature.Feature.FLOAT)
    ctr.float_value = 1

    # intermediate
    intermediate = feature.Feature(type=feature.Feature.SPARSE)
    intermediate.sparse_value.value.extend([3, 2, 1])

    # random
    random_vec = feature.Feature(type=feature.Feature.DENSE)
    random_vec.dense_value.value.extend([0.3, 0.2, 0.5])

    # put together
    features = {'age': age, 'sex': sex, 'embedding': tensor, 'city': city,
                'utags': utags, 'ntags': ntags, 'ctr': ctr,
                'intermediate': intermediate, 'random': random_vec}

    sample = feature.Sample(feature=features)
    return sample

def main():
    ''' entry '''
    schema_demo = define_schema()
    if schema_demo.version < MINIMUM_SCHEMA_VERSION:
        raise RuntimeError(f'schema version {schema_demo.version} < required mininum schema version {MINIMUM_SCHEMA_VERSION}')
    schema_analyzer = Analyzer(schema_demo)
    schema_analyzer.topo_sort()
    print(text_format.MessageToString(schema_demo, as_utf8=True))
    print('features in topological order:',
          schema_analyzer.topo_sorted_feature)
    schema_analyzer.collect_needed_stats_type()
    print('statistics to be collected:', schema_analyzer.stats_to_collect)
    sample_demo = generate_feature()
    validate(sample_demo, schema_demo, feature.Validator.BEFORE_TRANSFORM)
    transform(sample_demo, schema_demo, {})
    validate(sample_demo, schema_demo, feature.Validator.AFTER_TRANSFORM)
    print(sample_demo)


if __name__ == '__main__':
    main()
