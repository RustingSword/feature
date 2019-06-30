''' validators '''
import feature_pb2 as feature
from config import CURRENT_STAGE

def validate_int(value, validator, feature_name):
    ''' validate feature of int value '''
    if validator.HasField('int_min'):
        assert value >= validator.int_min, \
               f'{feature_name}: {value} < {validator.int_min}'
    if validator.HasField('int_max'):
        assert value <= validator.int_max, \
               f'{feature_name}: {value} > {validator.int_max}'
    if validator.one_of_int:
        assert value in validator.one_of_int, \
               f'{feature_name}: {value} not in {validator.one_of_int}'

def validate_float(value, validator, feature_name):
    ''' validate feature of float value '''
    if validator.HasField('float_min'):
        assert value >= validator.float_min, \
               f'{feature_name}: {value} < {validator.float_min}'
    if validator.HasField('float_max'):
        assert value <= validator.float_max, \
               f'{feature_name}: {value} > {validator.float_max}'

def validate_string(value, validator, feature_name):
    ''' validate feature of string value '''
    if validator.one_of_string:
        assert value in validator.one_of_string, \
               f'{feature_name}: {value} not in {validator.one_of_string}'

def validate_dense(value, validator, feature_name):
    ''' validate feature of dense value '''
    if validator.HasField('dense_value_dim'):
        assert len(value.value) == validator.dense_value_dim, \
               f'{feature_name} {len(value.value)} != {len(validator.dense_value_dim)}'
    for v in value.value:
        validate_float(v, validator, feature_name)

def validate_sparse(value, validator, feature_name):
    ''' validate feature of sparse value '''
    for v in value.value:
        validate_int(v, validator, feature_name)

def validate_cross(value, validator, feature_name):
    ''' validate feature of sparse value '''
    for v in value.list_value:
        validate_int(v, validator, feature_name)
    validate_float(value.point_value, validator, feature_name)

def validate(sample, schema, stage=feature.Validator.BEFORE_TRANSFORM):
    ''' validate sample against schema '''
    for feat in schema.feature:
        name = feat.name
        if feat.lifecycle_stage < CURRENT_STAGE:
            print('ignore feature %s (lifecycle stage %s)' %
                  (name, feature.Feature.LifecycleStage.Name(feat.lifecycle_stage)))
            continue
        validator = feat.validator
        if validator.phase == feature.Validator.SKIPPED:
            print(f'skip validator of {feat.name}')
            continue
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
            validate_int(value, validator, name)
        elif feature_in_sample.type == feature.Feature.FLOAT:
            value = feature_in_sample.float_value
            validate_float(value, validator, name)
        elif feature_in_sample.type == feature.Feature.STRING:
            value = feature_in_sample.string_value
            validate_string(value, validator, name)
        elif feature_in_sample.type == feature.Feature.DENSE:
            value = feature_in_sample.dense_value
            validate_dense(value, validator, name)
        elif feature_in_sample.type == feature.Feature.SPARSE:
            value = feature_in_sample.sparse_value
            validate_sparse(value, validator, name)
        elif feature_in_sample.type == feature.Feature.CROSS:
            value = feature_in_sample.cross_value
            validate_cross(value, validator, name)
        # TODO validate collective stats, such as missing ratio


