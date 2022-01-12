from enum import Enum


class Method(Enum):
    ASLT = 'aslt'
    AVG = 'avg'
    MLN_PRAC = 'mlnprac'
    MLN_ALCH = 'mlnalch'
    BIN_MEMM = 'binmemm'
    FLOAT_MEMM = 'floatmemm'
    REDUCED_BIN_MEMM = 'redbinmemm'
    REDUCED_FLOAT_MEMM = 'redfloatmemm'
    PARENT_SENS_FLOAT_MEMM = 'parentmemm'
