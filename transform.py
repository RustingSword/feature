''' transformers '''
from functools import reduce  # pylint: disable=redefined-builtin
from itertools import product
from operator import mul
from numpy import dot
from numpy.linalg import norm
from bisect import bisect_left
from hashlib import md5
import mmh3
import xxhash
from config import CURRENT_STAGE, OOV_SYMBOL
import feature_pb2 as feature

##################################
# transformations
##################################

def clip(feat, spec):
    ''' clip feature value '''
    if spec.HasField('int_min_value'):
        feat.int_value = max(spec.int_min_value, feat.int_value)
    if spec.HasField('int_max_value'):
        feat.int_value = min(spec.int_max_value, feat.int_value)
    if spec.HasField('float_min_value'):
        feat.float_value = max(spec.float_min_value, feat.float_value)
    if spec.HasField('float_max_value'):
        feat.float_value = min(spec.float_max_value, feat.float_value)

def normalize(feat, spec):
    ''' normalize feature value '''
    if feat.type == feature.Feature.FLOAT:
        feat.float_value = (feat.float_value - spec.mean) / spec.std
    elif feat.type == feature.Feature.DOUBLE:
        feat.double_value = (feat.double_value - spec.mean) / spec.std
    elif feat.type == feature.Feature.DENSE:
        for index in range(len(feat.dense_value.value)):
            feat.dense_value.value[index] = \
                (feat.dense_value.value[index] - spec.mean) / spec.std
    elif feat.type == feature.Feature.CROSS:
        feat.cross_value.point_value = \
            (feat.cross_value.point_value - spec.mean) / spec.std


def discretize(feat, spec):
    if not spec.boundaries:
        raise RuntimeError(f'no discretization boundaries for {feat.name}')
    feat.int_value = bisect_left(spec.boundaries, feat.float_value)
    # NOTE feature type changed to INT
    feat.type = feature.Feature.INT

def build_vocab_and_convert_to_id(feat, word_to_index):
    ''' build vocab and convert to id '''
    if feat.type == feature.Feature.STRING:
        feat.int_value = word_to_index.get(feat.string_value, 0)
        feat.type = feature.Feature.INT
    elif feat.type == feature.Feature.STRING_LIST:
        feat.sparse_value.value = []  # clear
        for word in feat.string_list_value.value:
            feat.sparse_value.value.append(word_to_index.get(word, 0))
        feat.type = feature.Feature.SPARSE

def hash_to_interval(feat, spec):
    ''' hash int/sparse/cross features to interval '''
    if feat.type == feature.Feature.INT:
        feat.int_value = feat.int_value % spec.modulus + spec.offset
    elif feat.type == feature.Feature.SPARSE:
        for index in range(len(feat.sparse_value.value)):
            feat.sparse_value.value[index] = \
                feat.sparse_value.value[index] % spec.modulus + spec.offset
    elif feat.type == feature.Feature.CROSS:
        for index in range(len(feat.cross_value.list_value)):
            feat.cross_value.list_value[index] = \
                feat.cross_value.list_value[index] % spec.modulus + spec.offset

def hash_string_to_64bit_int(string, method=feature.HashToInteger.XXHASH):
    ''' string to int '''
    return hash_bytes_to_64bit_int(string.encode('utf8'), method)

def hash_bytes_to_64bit_int(bytes_data, method=feature.HashToInteger.XXHASH):
    ''' bytes to int '''
    if method == feature.HashToInteger.XXHASH:
        return xxhash.xxh64_intdigest(bytes_data)
    if method == feature.HashToInteger.MD5:
        return int(md5(bytes_data).hexdigest()[:16], 16)
    if method == feature.HashToInteger.MURMURHASH3:
        return mmh3.hash64(bytes_data, signed=False)[0]
    raise RuntimeError(f'unsupported hash method '
                       '{feature.HashStringToInteger.Name(method)}')

def load_vocab(vocab_file):
    word_to_index = {OOV_SYMBOL: 0}
    with open(vocab_file) as fin:
        for index, line in enumerate(fin, start=1):
            word, _ = line.strip('\n').split('\t')
            word_to_index[word] = index
    return word_to_index

def apply_transformation(feat, trans):
    ''' apply transformations on features '''
    if trans.HasField('clip'):
        clip(feat, trans.clip)
    if trans.HasField('normalize'):
        normalize(feat, trans.normalize)
    if trans.HasField('discretize'):
        discretize(feat, trans.discretize)
    if trans.HasField('build_vocab_and_convert_to_id'):
        word_to_index = load_vocab(
                            trans.build_vocab_and_convert_to_id.vocab_file_name)
        build_vocab_and_convert_to_id(feat, word_to_index)
    if trans.HasField('hash_to_interval'):
        hash_to_interval(feat, trans.hash_to_interval)
    if trans.HasField('hash_to_integer'):  # XXX will not be used directly?
        if feat.type == feature.Feature.STRING:
            feat.int_value = hash_string_to_64bit_int(feat.string_value)
            feat.type = feature.Feature.INT
        elif current_feature.type == feature.Feature.STRING_LIST:
            feat.sparse_value.value = []
            for string in feat.string_list_value.value:
                feat.sparse_value.value.append(hash_string_to_64bit_int(string))
            feat.type = feature.Feature.SPARSE
        elif current_feature.type == feature.Feature.BYTES:
            feat.int_value = hash_bytes_to_64bit_int(feat.bytes_value)
            feat.type = feature.Feature.INT

##################################
# cross features
##################################

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
            raise ValueError(f'dependency feature {name} not in sample')
        cross_feat = get_feature_by_name(name, schema)
        if cross_feat is None:
            raise ValueError(f'dependency feature {name} not in schema')
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


def truncated_mul(*value_list):
    ''' keep result in 64 bit range '''
    return mul(*value_list) & 0xffffffffffffffff

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
        elif current_feature.type == feature.Feature.STRING:
            feature_values.append([hash_string_to_64bit_int(
                current_feature.string_value)])
        elif current_feature.type == feature.Feature.STRING_LIST:
            hashed_features = []
            for string in current_feature.string_list_value.value:
                hashed_features.append(hash_string_to_64bit_int(string))
            feature_values.append(hashed_features)
        elif current_feature.type == feature.Feature.BYTES:
            feature_values.append([hash_bytes_to_64bit_int(
                current_feature.bytes_value)])
        else:
            raise ValueError(
                'unsupported feature type %s' % feature.FeatureType.Name(
                            current_feature.type))
    prod = product(*feature_values)  # pylint: disable=bad-option-value
    result = []
    for crossed_value in prod:
        # FIXME maybe should use some other operator instead of mul, which will
        # introduce duplicated feature values, e.g., 1 * 2 = 2
        result.append(reduce(truncated_mul, crossed_value))
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
    ''' transform
    first apply transformations on features, then add cross features
    '''
    for feat in schema.feature:
        feature_in_sample = sample.feature[feat.name]
        for trans in feat.transformer:
            apply_transformation(feature_in_sample, trans)
    for feat in schema.feature:
        if feat.type == feature.Feature.CROSS:
            add_cross_feature(sample, feat, schema)
