from mongo import mongodb

post_ids = mongodb.postmemes.find({'datetime': None}, {'post_id': 1, '_id': 0})
post_ids_done = set()
count = post_ids.count()
i = 0

for p in post_ids:
    i += 1
    if p['post_id'] in post_ids_done:
        continue
    dt = mongodb.posts.find_one({'_id': p['post_id']}, ['datetime'])['datetime']
    mongodb.postmemes.update_many({'post_id': p['post_id']}, {'$set': {'datetime': dt}})
    post_ids_done.add(p['post_id'])
    if i % 1000 == 0:
        print('{:.0f}% done. {} datetimes set'.format(i / count * 100, i))

print('100% done')
