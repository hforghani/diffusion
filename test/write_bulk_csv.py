import csv
import os
import time
from cascade.weibo import read_uidlist
from settings import logger
import settings


def write_bulk_csv(relations_file, uidlist_file):
    t0 = time.time()

    uid_list = read_uidlist(uidlist_file)

    # Write the users into a csv file.
    logger.info('writing users csv file ...')
    users_csv_file = os.path.join(settings.BASEPATH, 'data', 'weibo-users.csv')
    with open(users_csv_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow([':ID(User)', '_id'])
        i = 0
        for uid in uid_list:
            csv_writer.writerow([str(i), str(uid)])
            i += 1

    # Write PARENT_OF relations between the users into a csv file.
    logger.info('writing relations csv file ...')
    rel_csv_file = os.path.join(settings.BASEPATH, 'data', 'weibo-relations.csv')
    with open(rel_csv_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow([':START_ID(User)', ':END_ID(User)'])
        i = 0

        # Read the relations from the file.
        with open(relations_file, encoding='utf-8', errors='ignore') as f:
            f.readline()
            line = f.readline()

            while line:
                line = line.strip().split()
                u1_i = line[0]
                n = int(line[1])

                for j in range(n):
                    u2_i = line[2 + j * 2]
                    csv_writer.writerow([u2_i, u1_i])

                i += 1
                if i % 10000 == 0:
                    logger.info('%d lines read' % i)

                line = f.readline()

    logger.info('csv files wrote in %.2f min', (time.time() - t0) / 60.0)


if __name__ == '__main__':
    write_bulk_csv(settings.WEIBO_FOLLOWERS_PATH, settings.WEIBO_UIDLIST_PATH)
