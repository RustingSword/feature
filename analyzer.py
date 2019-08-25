''' analyzer '''
from __future__ import print_function

from collections import defaultdict


class Analyzer:
    ''' analyze schema
    1. create DAG of features
    2. collect statistics that are needed for validator/transformer
    '''

    def __init__(self, schema):
        self.schema = schema
        self.features_in_topo_order = []
        self.needed_stats = defaultdict(set)
        self.transformers = defaultdict(dict)

    @property
    def topo_sorted_feature(self):
        ''' feature names in topological order '''
        return self.features_in_topo_order

    @property
    def stats_to_collect(self):
        ''' stats to be collected '''
        return self.needed_stats

    @property
    def corresponding_transformers(self):
        return self.transformers

    def topo_sort(self):
        ''' rearrange feature in topological order for transformation '''

        precursor = defaultdict(list)
        successor = defaultdict(list)
        temporary = set()
        feature_set = set()

        def _visit(feat):
            ''' helper function '''
            if feat in self.features_in_topo_order:
                return
            if feat in temporary:
                raise ValueError('cyclic dependency detected on feature %s' % feat)
            temporary.add(feat)
            for suc in successor[feat]:
                _visit(suc)
            temporary.remove(feat)
            self.features_in_topo_order.insert(0, feat)

        for feat in self.schema.feature:
            if feat.name in feature_set:
                raise ValueError('detected duplicate feature name %s' %
                                 (feat.name))
            feature_set.add(feat.name)
            for dep in feat.dependency_feature:
                successor[dep].append(feat.name)
                precursor[feat.name].append(dep)

        # topological sort by depth first search
        while feature_set:
            current_feature = feature_set.pop()
            _visit(current_feature)

        self.schema.topo_sorted_feature.extend(self.features_in_topo_order)

    def collect_stats_by_transforms(self, feat):
        ''' stats needed by transformers '''
        for trans in feat.transformer:
            if trans.HasField('discretize'):
                if not trans.discretize.boundaries:
                    if not trans.discretize.HasField('discretize_level'):
                        raise ValueError('discretize level not specified')
                    self.needed_stats[feat.name].add('bucket_info')
                    self.transformers[feat.name]['bucket_info'] = trans
            if trans.HasField('normalize'):
                if not trans.normalize.HasField('mean'):
                    self.needed_stats[feat.name].add('mean')
                if not trans.normalize.HasField('std'):
                    self.needed_stats[feat.name].add('std')
            if trans.HasField('clip'):
                if not (trans.clip.HasField('float_min_value') or
                        trans.clip.HasField('float_max_value') or
                        trans.clip.HasField('int_min_value') or
                        trans.clip.HasField('int_max_value')):
                    raise ValueError('clip transform of feature %s not '
                                     'initialized' % feat.name)
            if trans.HasField('build_vocab_and_convert_to_id'):
                transform = trans.build_vocab_and_convert_to_id
                if not transform.HasField('init_vocab_file'):
                    self.needed_stats[feat.name].add('vocab')
                    self.transformers[feat.name]['vocab'] = trans

    def collect_stats_by_validators(self, feat):
        ''' stats needed by validators '''
        if feat.validator.HasField('max_missing_ratio'):
            self.needed_stats[feat.name].add('missing_ratio')

    def collect_needed_stats_type(self):
        ''' collect statistics that needed for validators and transformers '''
        if self.schema.is_online_mode:
            return

        for feat in self.schema.feature:
            # check for transformers
            self.collect_stats_by_transforms(feat)

            # check for validators
            self.collect_stats_by_validators(feat)
