from enum import Enum


class Method(Enum):
    ASLT = 'aslt'
    CTIC = 'ctic'
    EMIC = 'emic'
    DAIC = 'daic'
    MLN_PRAC = 'mlnprac'
    MLN_ALCH = 'mlnalch'
    BIN_MEMM = 'bmemm'
    TD_MEMM = 'tdmemm'
    REDUCED_BIN_MEMM = 'rbmemm'
    REDUCED_TD_MEMM = 'rtdmemm'
    PARENT_SENS_TD_MEMM = 'prtdmemm'
    LONG_PARENT_SENS_TD_MEMM = 'lprtdmemm'
    AVG = 'avg'


class Criterion(Enum):
    NODES = 'nodes'
    EDGES = 'edges'
