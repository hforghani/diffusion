from enum import Enum


class Method(Enum):
    ASLT = 'aslt'
    CTIC = 'ctic'
    EMIC = 'emic'
    DAIC = 'daic'
    MLN_PRAC = 'mlnprac'
    MLN_ALCH = 'mlnalch'
    LONG_MEMM = 'lmemm'
    BIN_MEMM = 'bmemm'
    TD_MEMM = 'tdmemm'
    PARENT_SENS_TD_MEMM = 'ptdmemm'
    LONG_PARENT_SENS_TD_MEMM = 'lptdmemm'
    FULL_TD_MEMM = 'ftdmemm'
    TD_EDGE_MEMM = 'tdememm'
    AVG = 'avg'


class Criterion(Enum):
    NODES = 'nodes'
    EDGES = 'edges'
