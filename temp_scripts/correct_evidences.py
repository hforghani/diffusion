from cascade.models import Project, ParamTypes
from neo4j.models import Neo4jGraph
from settings import logger

#def is_power_of_2(num):
#    while num > 1:
#        if num % 2 == 1:
#            return False
#        num >>= 1
#    if num == 1:
#        return True
#    else:
#        return False
#
#
#if __name__ == '__main__':
#    project = Project('weibo-size7')
#    graph = Neo4jGraph('User')
#
#    train_set, test_set = project.load_train_test()
#    samples = []
#    samples.extend(train_set)
#    samples.extend(test_set)
#
#    for cid in samples:
#        logger.info('loading MEMM evidences for cascade %s ...', cid)
#        fname = 'memm/evidence-{}'.format(cid)
#        tsets = project.load_param(fname, ParamTypes.JSON)
#
#        count = 0
#        for uid in tsets:
#            dim, sequences = tsets[uid]
#
#            for seq in sequences:
#                assert seq[0][0] == 0 or is_power_of_2(seq[0][0])
#                for i in range(1,len(seq)):
#                    assert seq[i][0] == 0 or is_power_of_2(seq[i][0])
#                    seq[i][0] |= seq[i-1][0]
#
#            count += 1
#            if count % 10000 == 0:
#                logger.info('%d memm data corrected', count)
#
#        logger.info('saving corrected evidences ...')
#        project.save_param(tsets, fname, ParamTypes.JSON)



def correct_evidences(fname):
    tsets = project.load_param(fname, ParamTypes.JSON)
    id_to_delete = []
    for uid in tsets:
        dim, sequences = tsets[uid]
        if not sequences:
            id_to_delete.append(uid)
    for uid in id_to_delete:
        del tsets[uid]
    logger.info('saving corrected evidences ...')
    project.save_param(tsets, fname, ParamTypes.JSON)


if __name__ == '__main__':
    project = Project('weibo-size7')
    graph = Neo4jGraph('User')

    train_set, val_set, test_set = project.load_sets()
    samples = train_set + val_set + test_set

    for cid in samples:
        logger.info('loading MEMM evidences for cascade %s ...', cid)
        fname = 'memm/evidence-{}'.format(cid)
        correct_evidences(fname)

    # logger.info('loading MEMM evidences for all training set ...')
    # correct_evidences('memm/evidence')