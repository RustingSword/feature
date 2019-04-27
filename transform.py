
class Transform:
    ''' Transform features as specified in a given schema '''
    def __init__(self):
        pass

    @classmethod
    def clip(self, feat, spec):
        ''' clip feature value '''
        if spec.HasField('int_min_value'):
            feat.int_value = max(spec.int_min_value, feat.int_value)
        if spec.HasField('int_max_value'):
            feat.int_value = min(spec.int_max_value, feat.int_value)
        if spec.HasField('float_min_value'):
            feat.float_value = max(spec.float_min_value, feat.float_value)
        if spec.HasField('float_max_value'):
            feat.float_value = min(spec.float_max_value, feat.float_value)

    @classmethod
    def apply(self, feat, trans):
        ''' apply transformations on features '''
        if trans.HasField('clip'):
            self.clip(feat, trans.clip)
