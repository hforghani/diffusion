from enum import Enum


class Method(Enum):
    ASLT = 'aslt'
    CTIC = 'ctic'
    EMIC = 'emic'
    DAIC = 'daic'
    LONG_MEMM = 'lmemm'
    BIN_MEMM = 'bmemm'
    TD_MEMM = 'tdmemm'
    MULTI_STATE_LONG_MEMM = 'mlmemm'
    MULTI_STATE_BIN_MEMM = 'mbmemm'
    MULTI_STATE_TD_MEMM = 'mtdmemm'
    FULL_TD_MEMM = 'ftdmemm'
    LONG_CRF = 'lcrf'
    BIN_CRF = 'bcrf'
    TD_CRF = 'tdcrf'
    FULL_MULTI_STATE_BIN_CRF = 'fmbcrf'
    PAR_MULTI_STATE_LONG_CRF = 'mlcrf'
    PAR_MULTI_STATE_BIN_CRF = 'mbcrf'
    PAR_MULTI_STATE_TD_CRF = 'mtdcrf'
    MLN_TUFFY = 'mlntuffy'
    # MLN_PRAC = 'mlnprac'
    # MLN_ALCH = 'mlnalch'
    UNI_MRF = 'unimrf'


class Criterion(Enum):
    NODES = 'nodes'
    EDGES = 'edges'
