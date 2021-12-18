import sys

import pymongo

client = pymongo.MongoClient()
db_name = sys.argv[1]
db = client.get_database(db_name)
print(f'started for {db_name}')
db.postcascades.update_many({}, {'$rename': {'meme_id': 'cascade_id'}})
print('finished')
