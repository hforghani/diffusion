from cascade.models import CascadeTree, Project
from db.managers import DBManager
from settings import logger


def extract_trees(db):
    logger.info('reading postcascades ...')
    post_to_cascade = {pc['post_id']: pc['cascade_id'] for pc in db.postcascades.find({}, ['post_id', 'cascade_id'])}
    logger.info('%d postcascades read', len(post_to_cascade))
    trees = {}
    resh_count = db.reshares.count_documents({})
    i = 0

    logger.info('reading reshares ...')
    for resh in db.reshares.find().sort('datetime'):
        cascade_id = post_to_cascade[resh['post_id']]
        tree = trees.get(cascade_id)
        if not tree:
            tree = CascadeTree()
            trees[cascade_id] = tree

        if not tree.get_node(resh['ref_user_id']):
            tree.add_node(resh['ref_user_id'], resh['ref_datetime'])
        tree.add_node(resh['user_id'], resh['datetime'], resh['ref_user_id'])
        i += 1
        if i % 10 ** 5 == 0:
            logger.info('%i%% done', i / resh_count * 100)

    project = Project('twitter-all')
    project.save_trees(trees)


def main():
    manager = DBManager('twitter')
    db = manager.db
    extract_trees(db)


if __name__ == '__main__':
    main()
