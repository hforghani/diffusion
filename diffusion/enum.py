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
    MULTI_STATE_LONG_MEMM = 'mlmemm'
    MULTI_STATE_BIN_MEMM = 'mtdmemm'
    MULTI_STATE_TD_MEMM = 'mbmemm'
    PARENT_SENS_TD_MEMM = 'ptdmemm'
    LONG_PARENT_SENS_TD_MEMM = 'lptdmemm'
    FULL_TD_MEMM = 'ftdmemm'
    TD_EDGE_MEMM = 'tdememm'
    LONG_CRF = 'lcrf'
    BIN_CRF = 'bcrf'
    TD_CRF = 'tdcrf'
    MULTI_STATE_TD_CRF = 'mtdcrf'
    AVG = 'avg'


class Criterion(Enum):
    NODES = 'nodes'
    EDGES = 'edges'
