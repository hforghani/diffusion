from utils.db import DBManager


def main():
    unique_rels = {}
    ids_to_del = []
    i = 0
    del_count = 0
    db = DBManager().db

    with db.relations.find() as relations:
        for rel in relations:
            parent, child = rel['parent'], rel['child']
            parent = int(str(parent))
            child = int(str(child))
            #res = db.relations.find_one({'parent': parent, 'child': child})
            #if res['_id'] !=  rel['_id']:
            if parent in unique_rels and child in unique_rels[parent]:
                #db.relations.delete_one({'_id': rel['_id']})
                ids_to_del.append(rel['_id'])
                #logger.info('>>>> ({}, {}) deleted'.format(str(rel['parent']), str(rel['child'])))
                del_count += 1
                if del_count % 1000 == 0:
                    db.relations.delete_many({'_id': {'$in': ids_to_del}})
                    ids_to_del = []
                    logger.info('>>>> {} duplicates deleted'.format(del_count))
            else:
                unique_rels.setdefault(parent, set())
                unique_rels[parent].add(child)

            i += 1
            if i % 100000 == 0:
                logger.info('{} relations read'.format(i))

    db.relations.delete_many({'_id': {'$in': ids_to_del}})
    logger.info('>>>> {} duplicates deleted'.format(del_count))


if __name__ == '__main__':
    main()