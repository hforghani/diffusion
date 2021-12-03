import traceback

from db.managers import EvidenceManager, DBManager, MEMMManager
from memm.memm import MEMM, MemmException
from settings import logger
from utils.time_utils import Timer


def train_memms(evidences, save_in_db=False, project=None):
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


def test_memms(children, parents_dic, observations, active_ids, memms, threshold):
    try:
        logger.debug('testing memms started')

        active_children = []

        j = 0
        for child_id in children:
            if child_id not in parents_dic:
                continue
            parents = parents_dic[child_id]
            obs = observations[child_id]

            if child_id not in active_ids and child_id in memms:
                memm = memms[child_id]
                logger.debugv('testing reshare to user %s ...', child_id)
                new_state, prob = memm.predict(obs, len(parents), threshold)
                if new_state == 1:
                    active_children.append(child_id)
                    active_ids.append(child_id)
                    logger.debug('a reshare predicted %f > %f', prob, threshold)
            else:
                if child_id in active_ids:
                    logger.debugv('user %s is already activated', child_id)
                elif child_id not in memms:
                    logger.debugv('user %s does not have any MEMM', child_id)

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
            evidences = EvidenceManager(project).get_many_generator(inactive_children)

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


def extract_evidences(train_set, act_seqs):
    evidences = {}  # dictionary of user id's to list of the sequences of ObsPair instances.
    cascade_seqs = {}  # dictionary of user id's to the sequences of ObsPair instances for this current cascade
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

            # Put the last observation with state 1 in the sequence of (observation, state).
            if parents_count:
                observations.setdefault(uid, 0)  # initial observation: 0000000
                cascade_seqs.setdefault(uid, [])
                if cascade_seqs[uid]:
                    obs = cascade_seqs[uid][-1][0]
                    del cascade_seqs[uid][-1]
                    cascade_seqs[uid].append((obs, 1))

            if children:
                logger.debug('iterating on %d children ...', len(children))

            # Set the related coefficient of the children observations equal to 1.
            j = 0
            for child in children:
                rel = db.relations.find_one({'user_id': child}, {'_id': 0, 'parents': 1})
                if rel is None:
                    continue
                child_parents = rel['parents']
                parent_sizes[child] = len(child_parents)

                obs = observations.setdefault(child, 0)
                index = child_parents.index(uid)
                obs |= 1 << (len(child_parents) - index - 1)
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
