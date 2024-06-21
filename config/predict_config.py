from typing import Dict, Tuple, List, Union

import settings
from diffusion.enum import Method, Criterion


class Singleton(object):
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Singleton, cls).__new__(cls)
        return cls.instance


class PredictConfig(Singleton):
    project: str
    method: Method
    criterion: Criterion = Criterion.NODES
    init_depth: int = 0
    max_depth: int = None
    multiprocessed: bool = False
    eco: bool = False
    params: Dict[str, Union[float, Tuple[float, float]]] = {}
    n_iter: int = 100
    additional_metrics: List[str] = []
    saved: bool = False


def extract_param(param):
    new_param = {}
    for items in param:
        if len(items) == 3:
            param_name, start, end = items
            new_param[param_name] = (float(start), float(end))
        elif len(items) == 2:
            try:
                new_param[items[0]] = int(items[1])
            except ValueError:
                try:
                    new_param[items[0]] = float(items[1])
                except ValueError:
                    new_param[items[0]] = items[1]
        else:
            raise ValueError('invalid format for params option')

    return new_param


def set_config(method: str, project: str, criterion: str, initial_depth: int, max_depth: int,
               multiprocessed: bool, eco: bool, params: List[List[str]], n_iter: int, additional_metrics: List[str],
               saved: bool):
    config = PredictConfig()
    config.project = project
    config.method = Method(method)
    config.criterion = Criterion(criterion)
    config.init_depth = initial_depth
    config.max_depth = max_depth
    config.multiprocessed = multiprocessed
    config.eco = eco
    config.n_iter = n_iter
    config.additional_metrics = additional_metrics if additional_metrics is not None else []
    config.saved = saved
    if saved:
        config.params = get_saved_params(config.project, config.method)
    if params:
        config.params.update(extract_param(params))


def get_saved_params(project_name, method):
    if method in settings.PARAM:
        params = settings.PARAM[method]
        if project_name in params:
            return params[project_name]
    return {}
