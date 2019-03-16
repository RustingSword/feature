
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
        self.precursor = defaultdict(list)
        self.successor = defaultdict(list)
        self.feature_set = set()
        self.temporary = set()
        self.needed_stats = defaultdict(set)

    @property
    def topo_sorted_feature(self):
        return self.features_in_topo_order

    @property
    def stats_to_collect(self):
        return self.needed_stats

    def visit(self, feat):
        if feat in self.features_in_topo_order:
            return
        if feat in self.temporary:
            raise ValueError('cyclic dependency detected on feature %s' % feat)
        self.temporary.add(feat)
        for suc in self.successor[feat]:
            self.visit(suc)
        self.temporary.remove(feat)
        self.features_in_topo_order.insert(0, feat)

    def topo_sort(self):
        ''' rearrange feature in topological order for transformation '''
        for feat in self.schema.feature:
            if feat.name in self.feature_set:
                raise ValueError('detected duplicate feature name %s' %
                                 (feat.name))
            self.feature_set.add(feat.name)
            for dep in feat.dependency_feature:
                self.successor[dep].append(feat.name)
                self.precursor[feat.name].append(dep)

        # topological sort by depth first search
        while self.feature_set:
            current_feature = self.feature_set.pop()
            self.visit(current_feature)

    def collect_by_transforms(self, feat):
        for trans in feat.transformer:
            if trans.HasField('discretize'):
                if not trans.discretize.boundaries:
                    if not trans.discretize.HasField('discretize_level'):
                        raise ValueError('discretize level not specified')
                    self.needed_stats[feat.name].add('bucket_info')
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
                if transform.HasField('init_vocab_file'):
                    continue
                self.needed_stats[feat.name].add('vocab')

    def collect_by_validators(self, feat):
        if feat.validator.HasField('max_missing_ratio'):
            self.needed_stats[feat.name].add('missing_ratio')

    def collect_needed_stats(self):
        ''' collect statistics that needed for validators and transformers '''
        if self.schema.is_online_mode:
            return

        for feat in self.schema.feature:
            # check for transformers
            self.collect_by_transforms(feat)

            # check for validators
            self.collect_by_validators(feat)
