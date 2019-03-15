from mongo import mongodb

post_ids = mongodb.postmemes.find({'datetime': None}, {'post_id': 1, '_id': 0})
post_ids = list({p['post_id'] for p in post_ids})
datetimes = mongodb.posts.find({'_id': {'$in': post_ids}}, ['datetime'])
datetimes = {p['_id']: p['datetime'] for p in datetimes}

i = 0
count = mongodb.postmemes.find({'datetime': None}).count()

for pm in mongodb.postmemes.find({'datetime': None}, ['post_id']):
    mongodb.memes.find_one_and_update({'_id': pm['_id']}, {'$set': {'datetime': datetimes[pm['post_id']]}})
    i += 1
    if i % 1000 == 0:
        print('{:.0f}% done. {} postmemes updated'.format(i / count * 100, i))

print('100% done')
