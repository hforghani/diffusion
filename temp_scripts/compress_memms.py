import sys
import pickle
import gridfs
import pymongo
from scipy.sparse import csr_matrix

sys.path.append('.')

from settings import logger


# import pydevd_pycharm
#
# pydevd_pycharm.settrace('194.225.227.132', port=12345, stdoutToServer=True, stderrToServer=True)


def eval_all_obs(all_obs_str):
    overflow_thr = 10 ** 9
    if len(all_obs_str) < overflow_thr:
        return eval(all_obs_str)
    else:
        logger.debug('all_obs is too large: %d', len(all_obs_str))
        # logger.debug('before eval: %s', all_obs_str)
        quote_char = all_obs_str[-1:]
        res = b''
        i = 0
        counter = 0
        while i < len(all_obs_str):
            if len(all_obs_str) - i < overflow_thr:
                index = len(all_obs_str)
            else:
                index = all_obs_str.rfind(b'\\', i + 1, i + overflow_thr)
                if index == -1:
                    index = i + overflow_thr
            substr = all_obs_str[i:index]
            if index < len(all_obs_str):
                substr += quote_char
            if i != 0:
                substr = b"b" + quote_char + substr
            res += eval(substr)
            i = index
            counter += 1
            if counter % 10 == 0:
                logger.debug('%d iterations done', counter)
        # logger.debug('after eval: %s', res)
        return res


def compress_data(doc):
    data = doc.read()

    # print('evaluated before compress:', eval(data))

    i2 = data.index(b'tpm')
    i3 = data.index(b'all_obs_arr')
    i4 = data.index(b'map_obs_index')

    # print('all_obs_arr dump =', data[i3 + 21: i4 - 7])

    all_obs = eval_all_obs(data[i3 + 21: i4 - 7])
    all_obs_arr = pickle.loads(all_obs)
    new_all_obs = pickle.dumps(csr_matrix(all_obs_arr), protocol=2)
    new_data = data[:i2 + 6] + data[i2 + 13:i3 - 5] + data[i3 - 1:i3 + 14] + bytes(str(new_all_obs),
                                                                                   encoding='utf-8') + data[i4 - 3:]
    # print('new_data =', new_data)

    return new_data


def main():
    if len(sys.argv) < 2:
        raise ValueError('Db name must be given')
    db_name = sys.argv[1]
    mongo_client = pymongo.MongoClient()
    mongo_client.drop_database(f'{db_name}_2')

    db = mongo_client[db_name]
    db2 = mongo_client[f'{db_name}_2']
    fs = gridfs.GridFS(db)
    fs2 = gridfs.GridFS(db2)
    logger.info('compressing and inserting documents into "%s" :', f'{db_name}_2')

    i = 0
    for doc in fs.find(no_cursor_timeout=True):
        new_data = compress_data(doc)
        # test_dict = eval(new_data)
        # print('evaluated after compress:', test_dict)
        fs2.put(new_data, user_id=doc.user_id)
        i += 1
        if i % 10000 == 0:
            logger.info(f'{i} documents inserted')


if __name__ == '__main__':
    main()
