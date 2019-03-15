from mongo import mongodb

post_ids = mongodb.postmemes.find({'datetime': None}, {'post_id': 1, '_id': 0})
post_ids = list({p['post_id'] for p in post_ids})

step = 10 ** 6
count = len(post_ids)
for i in range(0, count, step):
    if i != 0:
        print('{:.0f}% done. {} datetimes set'.format(i / count * 100, i))
    posts = mongodb.posts.find({'_id': {'$in': post_ids[i: i + step]}}, ['datetime'])
    for p in posts:
        mongodb.postmemes.update_many({'post_id': p['_id']}, {'$set': {'datetime': p['datetime']}})

print('100% done')
