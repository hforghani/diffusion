from enum import Enum


class Method(Enum):
    ASLT = 'aslt'
    AVG = 'avg'
    MLN_PRAC = 'mlnprac'
    MLN_ALCH = 'mlnalch'
    BIN_MEMM = 'binmemm'
    TD_MEMM = 'tdmemm'
    REDUCED_BIN_MEMM = 'redbinmemm'
    REDUCED_TD_MEMM = 'redtdmemm'
    PARENT_SENS_TD_MEMM = 'parentmemm'
    LONG_PARENT_SENS_TD_MEMM = 'longparentmemm'


class Criterion(Enum):
    NODES = 'nodes'
    EDGES = 'edges'
