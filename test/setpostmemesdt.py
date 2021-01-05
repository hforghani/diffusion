from memm.db import DBManager
from settings import logger

db = DBManager().db
post_ids = db.postmemes.find({'datetime': None}, {'post_id': 1, '_id': 0})
post_ids_done = set()
count = post_ids.count()
i = 0

for p in post_ids:
    i += 1
    if p['post_id'] in post_ids_done:
        continue
    dt = db.posts.find_one({'_id': p['post_id']}, ['datetime'])['datetime']
    db.postmemes.update_many({'post_id': p['post_id']}, {'$set': {'datetime': dt}})
    post_ids_done.add(p['post_id'])
    if i % 1000 == 0:
        logger.info('{:.0f}% done. {} datetimes set'.format(i / count * 100, i))

logger.info('100% done')
