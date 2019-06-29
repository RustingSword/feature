''' validators '''
class Validator:
    ''' Validate features against a given schema '''
    def __init__(self):
        pass

    @classmethod
    def validate_int(cls, value, validator):
        ''' validate feature of int value '''
        if validator.HasField('int_min'):
            assert value >= validator.int_min, '%d < %d' % \
                (value, validator.int_min)
        if validator.HasField('int_max'):
            assert value <= validator.int_max, '%d > %d' % \
                (value, validator.int_max)
        if validator.one_of_int:
            assert value in validator.one_of_int, '%d not in %s' % \
                (value, validator.one_of_int)

    @classmethod
    def validate_float(cls, value, validator):
        ''' validate feature of float value '''
        if validator.HasField('float_min'):
            assert value >= validator.float_min, '%f < %f' % \
                (value, validator.float_min)
        if validator.HasField('float_max'):
            assert value <= validator.float_max, '%f > %f' % \
                (value, validator.float_max)

    @classmethod
    def validate_string(cls, value, validator):
        ''' validate feature of string value '''
        if validator.one_of_string:
            assert value in validator.one_of_string, '%s not in %s' % \
                (value, validator.one_of_string)

    @classmethod
    def validate_dense(cls, value, validator):
        ''' validate feature of dense value '''
        if validator.HasField('dense_value_dim'):
            assert len(value.value) == validator.dense_value_dim, '%d != %d' % \
                (len(value.value), validator.dense_value_dim)
        for v in value.value:
            cls.validate_float(v, validator)

    @classmethod
    def validate_sparse(cls, value, validator):
        ''' validate feature of sparse value '''
        for v in value.value:
            cls.validate_int(v, validator)

    @classmethod
    def validate_cross(cls, value, validator):
        ''' validate feature of sparse value '''
        for v in value.list_value:
            cls.validate_int(v, validator)
        cls.validate_float(value.point_value, validator)
