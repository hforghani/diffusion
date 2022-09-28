from cascade.models import ActSequence, Project
from db.managers import DBManager
from settings import logger


def extract_act_seq(db):
    logger.debug('fetching cascade ids ...')
    cascade_ids = [c['_id'] for c in db.cascades.find({}, ['_id'])]
    users = {m: [] for m in cascade_ids}
    times = {m: [] for m in cascade_ids}

    # Iterate on posts to extract activation sequences.
    logger.debug('iterating over postcascades ...')
    pc_count = db.postcascades.count_documents({})
    i = 0
    for pc in db.postcascades.find().sort('datetime'):
        cascade_id = pc['cascade_id']
        users[cascade_id].append(pc['author_id'])
        times[cascade_id].append(pc['datetime'])
        i += 1
        if i % 10000 == 0:
            logger.debug('%d%% posts done' % (i * 100 / pc_count))

    logger.info('setting relative times and max times ...')
    max_t = {}
    i = 0
    for cascade in db.cascades.find({}, ['last_time', 'first_time']):
        cid = cascade['_id']
        times[cid] = [(t - cascade['first_time']).total_seconds() / (3600.0 * 24 * 30) for t in
                      times[cid]]  # number of months
        max_t[cid] = (cascade['last_time'] - cascade['first_time']).total_seconds() / (
                3600.0 * 24 * 30)  # number of months
        i += 1
        if i % 10 ** 5 == 0:
            logger.debug('%d%% done' % (i * 100 / len(cascade_ids)))

    sequences = {}
    for cid in cascade_ids:
        if users[cid]:
            sequences[cid] = ActSequence(users[cid], times[cid], max_t[cid])

    logger.info('saving act. sequences ...')
    project = Project('twitter-all')
    project.save_act_sequences(sequences)


def main():
    manager = DBManager('twitter')
    db = manager.db
    extract_act_seq(db)


if __name__ == '__main__':
    main()
