from pympler.asizeof import asizeof
from settings import logger


def log_memory(locals, globals):
    var_dict = locals.copy()
    var_dict.update(globals)
    log = 'memory usage:'
    for var, val in var_dict.items():
        log += f'\n{var}: {asizeof(val)}'
    logger.debug(log)
