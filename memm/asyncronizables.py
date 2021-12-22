import traceback

from db.managers import DBManager, MEMMManager
from memm.memm import MEMM
from memm.exceptions import MemmException
from settings import logger


def train_memms(evidences, save_in_db=False, project=None):
    try:
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

                if save_in_db:
                    logger.debug('inserting MEMMs into db ...')
                    MEMMManager(project).insert(memms)
                    memms = {}

        logger.debugv('training memms finished')
        if not save_in_db:
            return memms
    except:
        logger.error(traceback.format_exc())
        raise


def extract_evidences(train_set, act_seqs):
    try:
        evidences = {}  # dictionary of user id's to list of the sequences of ObsPair instances.
        cascade_seqs = {}  # dictionary of user id's to the sequences of ObsPair instances for the current cascade
        parent_sizes = {}  # dictionary of user id's to number of their parents
        db = DBManager().db
        count = 0
        # act_seqs = project.load_or_extract_act_seq()

        # Iterate each activation sequence and extract sequences of (observation, state) for each user
        for cascade_id in train_set:
            act_seq = act_seqs[cascade_id]
            observations = {}  # current observation of each user
            activated = set()  # set of current activated users
            i = 0
            logger.info('cascade %d with %d users ...', count + 1, len(act_seq.users))

            for uid in act_seq.users:  # Notice users are sorted by activation time.
                activated.add(uid)
                rel = db.relations.find_one({'user_id': uid}, {'_id': 0, 'children': 1, 'parents': 1})
                parents_count = len(rel['parents']) if rel is not None else 0
                parent_sizes[uid] = parents_count
                logger.debug('extracting children ...')
                children = rel['children'] if rel is not None else []

                # Put the last observation with state 1 in the sequence of (observation, state) if exists.
                if parents_count:
                    observations.setdefault(uid, 0)  # initial observation: 0000000
                    cascade_seqs.setdefault(uid, [])
                    uid_cur_seqs = cascade_seqs[uid]
                    if uid_cur_seqs:
                        obs = uid_cur_seqs[-1][0]
                        del uid_cur_seqs[-1]
                        uid_cur_seqs.append((obs, 1))

                if children:
                    logger.debug('iterating on %d children ...', len(children))

                # Set the related coefficient of the children observations equal to 1.
                j = 0
                for child in children:
                    rel = db.relations.find_one({'user_id': child}, {'_id': 0, 'parents': 1})
                    if rel is not None:
                        child_parents = rel['parents']
                        parent_sizes[child] = len(child_parents)

                        obs = observations.setdefault(child, 0)
                        index = child_parents.index(uid)
                        obs |= 1 << index
                        observations[child] = obs
                        if child not in activated:
                            cascade_seqs.setdefault(child, [(0, 0)])
                            cascade_seqs[child].append((obs, 0))
                    j += 1
                    if j % 1000 == 0:
                        logger.debug('%d children done', j)

                i += 1
                logger.debug('%d users done', i)

            count += 1
            if count % 1000 == 0:
                logger.info('%d cascades done', count)

            # Add current sequence of pairs (observation, state) to the MEMM evidences.
            logger.debug('adding sequences of current cascade ...')
            for uid in cascade_seqs:
                dim = parent_sizes[uid]
                evidences.setdefault(uid, {
                    'dimension': dim,
                    'sequences': []
                })
                evidences[uid]['sequences'].append(cascade_seqs[uid])
            cascade_seqs = {}

        return evidences
    except:
        logger.error(traceback.format_exc())
        raise
