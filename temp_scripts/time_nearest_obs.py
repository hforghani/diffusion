from random import randint
from bson import ObjectId
import numpy as np
import sys

sys.path.append('.')

from db.managers import MEMMManager
from cascade.models import Project
from memm.memm import obs_to_array, array_to_obs, obs_to_str
from utils.time_utils import Timer, TimeUnit
from settings import logger


def current_nearest_obs(memm, obs, dim):
    observations = sorted(list(memm.map_obs_prob.keys()))[1:]
    dist = np.array([bin(obs ^ o).count('1') for o in observations])
    index = np.argmin(dist)
    sim = (dim - np.min(dist)) / dim
    return observations[index], sim


def nearest_obs(memm, obs, dim):
    new_obs_vec = obs_to_array(obs, dim)
    sim = np.count_nonzero(memm.all_obs_arr[1:, :] == np.tile(new_obs_vec, (memm.all_obs_arr.shape[0] - 1, 1)), axis=1)
    index = np.argmax(sim) + 1
    nearest_obs = array_to_obs(memm.all_obs_arr[index, :])
    return nearest_obs, np.max(sim) / dim


project = Project('mt-size364')
manager = MEMMManager(project)
memm = manager.fetch_one(ObjectId('5c5d12f28688770e5864ddc7'))
dim = len(memm.orig_indexes)
obs = randint(0, 2 ** dim - 1)
logger.info('obs = %s', obs)
logger.info('to str = %s', obs_to_str(obs, dim))
logger.debug('dim = %d', dim)
number = 10000

with Timer('current nearest obs'):
    for _ in range(number):
        obs2, sim = current_nearest_obs(memm, obs, dim)
logger.info('obs2 = %d, sim = %f', obs2, sim)
logger.info('to str = %s', obs_to_str(obs2, dim))

memm.all_obs_arr = memm.all_obs_arr.toarray()
with Timer('new nearest obs'):
    for _ in range(number):
        obs2, sim = nearest_obs(memm, obs, dim)
logger.info('obs2 = %d, sim = %f', obs2, sim)
logger.info('to str = %s', obs_to_str(obs2, dim))
