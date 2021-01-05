import multiprocessing
import traceback

from pymongo import MongoClient

from memm.db import EvidenceManager
from memm.memm import MEMM, MemmException
from settings import logger, DB_NAME
from utils.time_utils import Timer


def train_memms(evidences):
    user_ids = list(evidences.keys())
    logger.debugv('training memms started')
    memms = {}
    count = 0
    for uid in user_ids:
        count += 1
        ev = evidences.pop(uid)  # to free RAM
        #    logger.debug('training MEMM %d (user id: %s, dimensions: %d) ...', count, uid, ev[0])
        m = MEMM()
        try:
            m.fit(ev)
            memms[uid] = m
        except MemmException:
            logger.warn('evidences for user %s ignored due to insufficient data', uid)
        if count % 1000 == 0:
            logger.debug('%d memms trained', count)

    logger.debugv('training memms finished')
    return memms


def test_memms(children, parents_dic, observations, active_ids, memms, threshold):
    try:
        logger.debug('testing memms started')

        active_children = []

        j = 0
        for child_id in children:
            if child_id not in parents_dic:
                continue
            parents = parents_dic[child_id]
            # rel = mongodb.relations.find_one({'user_id': child_id}, {'_id': 0, 'parents': 1})
            # if rel is None:
            #     continue
            # parents = rel['parents']

            obs = observations[child_id]
            # logger.debug('child_id not in active_ids: %s, child_id in self.__memms: %s',
            #             child_id not in active_ids, child_id in self.__memms)

            if child_id not in active_ids and child_id in memms:
                memm = memms[child_id]
                # logger.debug('predicting cascade ...')
                new_state = memm.predict(obs, len(parents), threshold)
                if new_state == 1:
                    active_children.append(child_id)
                    active_ids.append(child_id)
                    # logger.debug('\ta reshare predicted')

            j += 1
            if j % 100 == 0:
                logger.debugv('%d / %d of children iterated', j, len(children))

        del memms, children, parents_dic, observations, active_ids

        logger.debug('testing memms finished')
        return active_children
    except:
        traceback.print_exc()
        raise


def test_memms_eco(children, parents_dic, observations, project, active_ids, threshold):
    """
    Economical version of test_memms. Consumes less RAM by training MEMMS in-place.
    :param children:        list of child user id's
    :param parents_dic:     dictionary of user id's to their parent id's
    :param observations:    dictionary of user id's to their observation vectors (int)
    :param active_ids:      list of current active user id's
    :param threshold:       model threshold
    :return:                newly active user id's
    """
    try:
        db_timer = Timer('getting evidence', silent=True)
        train_timer = Timer('training memm in eco mode', silent=True)
        test_timer = Timer('testing memm in eco mode', silent=True)

        logger.debug('testing memms started')

        active_children = []

        inactive_children = list((set(children) & set(parents_dic.keys())) - set(active_ids))
        with db_timer:
            # A new instance is created since instances of MongoClient must not be copied from
            # a parent process to a child process.
            evidences = EvidenceManager().get_many(project, inactive_children)

        j = 0
        for child_id, ev in evidences:
            parents = parents_dic[child_id]
            obs = observations[child_id]

            memm = MEMM()
            try:
                with train_timer:
                    memm.fit(ev)
                # logger.debug('predicting cascade ...')
                with test_timer:
                    new_state = memm.predict(obs, len(parents), threshold)
                del memm
                if new_state == 1:
                    active_children.append(child_id)
                    active_ids.append(child_id)
                    # logger.debug('\ta reshare predicted')
            except MemmException:
                logger.warn('evidences for user %s ignored due to insufficient data', child_id)

            j += 1
            if j % 100 == 0:
                logger.debug('%d / %d of children iterated', j, len(inactive_children))

        del children, parents_dic, observations, active_ids

        for timer in [db_timer, train_timer, test_timer]:
            timer.report_sum()

        logger.debug('testing memms finished')
        return active_children

    except:
        traceback.print_exc()
        raise
