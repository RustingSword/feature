#!/usr/bin/env python3
#!coding: utf8
''' test feature.proto '''

from __future__ import print_function

from functools import reduce  # pylint: disable=redefined-builtin
from itertools import product

from google.protobuf import text_format
from numpy import dot
from numpy.linalg import norm
from operator import mul

import feature_pb2 as feature
from analyzer import Analyzer
from transform import Transform
from validator import Validator

CURRENT_STAGE = feature.Feature.ALPHA


def define_schema():
    ''' define a schema '''
    schema = feature.Schema()
    schema.version = 1

    # age
    validator = feature.Validator()
    validator.int_min = 0
    validator.int_max = 99
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
    trans = new_feature.transformer.add(hash_by_md5=feature.HashByMd5())

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
    trans = new_feature.transformer.add()
    trans.discretize.discretize_level = 10

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


def get_feature_by_name(fname, schema):
    ''' features in schema is stored in a list, iterate through them to find
    feature with given name '''
    for feat in schema.feature:
        if feat.name == fname:
            return feat
    return None


def cross_feature_sanity_check(sample, feat, schema):
    ''' sanity check:
    - dependency features must exist in sample
    - lifecycle stage of dependency features should be equal or
      later than CURRENT_STAGE and current feature's lifecycle stage
    '''
    for name in feat.dependency_feature:
        if name not in sample.feature:
            raise ValueError('dependency feature %s not in sample' % (name))
        cross_feat = get_feature_by_name(name, schema)
        if cross_feat is None:
            raise ValueError('dependency feature %s not in schema' % (name))
        if cross_feat.lifecycle_stage < feat.lifecycle_stage:
            raise ValueError('dependency feature %s in stage %s cannot be '
                             'used for cross feature %s in stage %s' % (
                                 name,
                                 feature.LifecycleStage.Name(
                                     cross_feat.lifecycle_stage),
                                 feat.name,
                                 feature.LifecycleStage.Name(feat.lifecycle_stage)))
        if cross_feat.lifecycle_stage < CURRENT_STAGE:
            raise ValueError('dependency feature %s in stage %s cannot be '
                             'used in current stage %s' % (
                                 name,
                                 feature.LifecycleStage.Name(
                                     cross_feat.lifecycle_stage),
                                 feature.LifecycleStage.Name(CURRENT_STAGE)))


def add_cartesian_cross_feature(sample, feat):
    ''' cartesian product of two or more features '''
    feature_values = []
    for name in feat.dependency_feature:
        current_feature = sample.feature[name]
        if current_feature.type == feature.Feature.INT:
            feature_values.append([current_feature.int_value])
        elif current_feature.type == feature.Feature.SPARSE:
            feature_values.append(current_feature.sparse_value.value)
        elif current_feature.type == feature.Feature.CROSS:
            feature_values.append(current_feature.cross_value.list_value)
        else:
            raise ValueError(
                'only accept one of ["INT", "SPARSE", "CROSS"] features, \
                        got %s' % feature.FeatureType.Name(
                    current_feature.type))
    prod = product(*feature_values)  # pylint: disable=star-args
    result = []
    for crossed_value in prod:
        result.append(reduce(mul, crossed_value))
    return result


def add_dot_cross_feature(sample, feat):
    ''' dot product/cosine similarity of two dense features '''
    assert len(feat.dependency_feature) == 2
    feat1, feat2 = feat.dependency_feature
    feat1, feat2 = sample.feature[feat1], sample.feature[feat2]
    assert feat1.type == feature.Feature.DENSE
    assert feat2.type == feature.Feature.DENSE
    value1 = feat1.dense_value.value
    value2 = feat2.dense_value.value
    assert len(value1) == len(value2)
    if feat.cross_value.cross_method == feature.CrossDomain.COSINE_SIMILARITY:
        norm1 = norm(value1)
        norm2 = norm(value2)
        if norm1 == 0 or norm2 == 0:
            crossed_value = 0.0
        else:
            crossed_value = dot(value1, value2) / (norm1 * norm2)
    elif feat.cross_value.cross_method == feature.CrossDomain.DOT_PRODUCT:
        crossed_value = dot(value1, value2)
    else:
        raise ValueError('unsupported cross operation %s' % (
            feature.CrossMethod.Name(feat.cross_value.cross_method)))
    return crossed_value


def add_cross_feature(sample, feat, schema):
    ''' add a crossed feature '''
    cross_feature_sanity_check(sample, feat, schema)
    cross_method = feat.cross_value.cross_method
    new_feature = feature.Feature(type=feature.Feature.CROSS)
    if cross_method == feature.CrossDomain.CARTESIAN_PRODUCT:
        crossed_values = add_cartesian_cross_feature(sample, feat)
        new_feature.cross_value.list_value.extend(crossed_values)
    else:  # both COSINE_SIMILARITY and DOT_PRODUCT require two DENSE features
        crossed_value = add_dot_cross_feature(sample, feat)
        new_feature.cross_value.point_value = crossed_value
    sample.feature[feat.name].CopyFrom(new_feature)


def transform(sample, schema):
    ''' transform '''
    for feat in schema.feature:
        if feat.type == feature.Feature.CROSS:
            add_cross_feature(sample, feat, schema)
        feature_in_sample = sample.feature[feat.name]
        for trans in feat.transformer:
            Transform.apply(feature_in_sample, trans)


def validate(sample, schema, stage=feature.Validator.BEFORE_TRANSFORM):
    ''' validate sample against schema '''
    print('='*20)
    print('validation of stage %s' % (feature.Validator.ValidatePhase.Name(stage)))
    print('='*20)
    for feat in schema.feature:
        name = feat.name
        if feat.lifecycle_stage < CURRENT_STAGE:
            print('ignore feature %s (lifecycle stage %s)' %
                  (name, feature.Feature.LifecycleStage.Name(feat.lifecycle_stage)))
            continue
        validator = feat.validator
        if validator.phase not in (stage, feature.Validator.BEFORE_AND_AFTER_TRANSFORM):
            continue
        if name not in sample.feature:
            if feat.type == feature.Feature.CROSS or validator.allow_missing:
                # CROSS feature will be generated in transformation stage
                continue
            raise ValueError('feature %s is missing' % name)
        feature_in_sample = sample.feature[name]
        if feature_in_sample.type != feat.type:
            raise ValueError('feature type unmatch (%s != %s)' %
                             (feature.FeatureType.Name(feature_in_sample.type),
                              feature.FeatureType.Name(feat.type)))
        if feature_in_sample.type == feature.Feature.INT:
            value = feature_in_sample.int_value
            Validator.validate_int(value, validator)
        elif feature_in_sample.type == feature.Feature.FLOAT:
            value = feature_in_sample.float_value
            Validator.validate_float(value, validator)
        elif feature_in_sample.type == feature.Feature.STRING:
            value = feature_in_sample.string_value
            Validator.validate_string(value, validator)
        elif feature_in_sample.type == feature.Feature.DENSE:
            value = feature_in_sample.dense_value
            Validator.validate_dense(value, validator)
        elif feature_in_sample.type == feature.Feature.SPARSE:
            value = feature_in_sample.sparse_value
            Validator.validate_sparse(value, validator)
        elif feature_in_sample.type == feature.Feature.CROSS:
            value = feature_in_sample.cross_value
            Validator.validate_cross(value, validator)
        # print('value of %s (%s): %s' % (name, feat.desc, value))


def print_final_sample(sample, schema):
    ''' print feature values in sample '''
    print('='*20)
    print('feature valuse of sample')
    print('='*20)
    for feat in schema.feature:
        if feat.lifecycle_stage < CURRENT_STAGE:
            print('ignore feature %s (lifecycle stage %s)' %
                  (feat.name, feature.Feature.LifecycleStage.Name(feat.lifecycle_stage)))
            continue
        if feat.is_intermediate_feature:
            print('ignore intermediate feature %s' % (feat.name))
            continue
        feature_in_sample = sample.feature[feat.name]
        if feature_in_sample.type == feature.Feature.INT:
            value = feature_in_sample.int_value
        elif feature_in_sample.type == feature.Feature.FLOAT:
            value = feature_in_sample.float_value
        elif feature_in_sample.type == feature.Feature.STRING:
            value = feature_in_sample.string_value
        elif feature_in_sample.type == feature.Feature.DENSE:
            value = feature_in_sample.dense_value
        elif feature_in_sample.type == feature.Feature.SPARSE:
            value = feature_in_sample.sparse_value
        elif feature_in_sample.type == feature.Feature.CROSS:
            value = feature_in_sample.cross_value
        print('value of %s (%s): %s' % (feat.name, feat.desc, value))


def main():
    ''' entry '''
    schema_demo = define_schema()
    print(text_format.MessageToString(schema_demo, as_utf8=True))
    schema_analyzer = Analyzer(schema_demo)
    schema_analyzer.topo_sort()
    print('features in topological order:',
          schema_analyzer.topo_sorted_feature)
    schema_analyzer.collect_needed_stats()
    print('statistics to be collected:', schema_analyzer.stats_to_collect)
    sample_demo = generate_feature()
    validate(sample_demo, schema_demo, feature.Validator.BEFORE_TRANSFORM)
    transform(sample_demo, schema_demo)
    validate(sample_demo, schema_demo, feature.Validator.AFTER_TRANSFORM)
    print_final_sample(sample_demo, schema_demo)


if __name__ == '__main__':
    main()
