import argparse
import json
import os
import pprint
import string
from email.parser import Parser

import settings
from settings import logger
from utils.time_utils import time_measure


def split_to_words(in_str):
    in_str = in_str.strip()
    for ch in string.punctuation:
        in_str = in_str.replace(ch, ' ')
    return in_str.split()


def jaccard(set1, set2):
    if not set1 or not set2:
        return 0
    else:
        return len(set1 & set2) / len(set1 | set2)


def find_thread(threads, email):
    words = set(split_to_words(email['subject']))
    for i in range(len(threads)):
        thread = threads[i]
        if all([jaccard(set(split_to_words(subject)), words) > 0.5 for subject in thread.subjects]):
            if email['to'] in thread.users or email['from'] in thread.users:
                return i
    return None


def extract_dir_file_names(path):
    file_names = []
    for root, dirs, files in os.walk(path):
        file_names.extend(os.path.join(root, f) for f in files)
    return file_names


def extract_file_names(root_dir, dir_name):
    if os.path.exists(os.path.join(root_dir, dir_name, 'all_documents')):
        logger.debug('files of all_documents returned')
        return extract_dir_file_names(os.path.join(root_dir, dir_name, 'all_documents'))
    else:
        file_names = []
        if os.path.exists(os.path.join(root_dir, dir_name, 'inbox')):
            file_names = extract_dir_file_names(os.path.join(root_dir, dir_name, 'inbox'))
        if os.path.exists(os.path.join(root_dir, dir_name, 'sent_items')):
            file_names.extend(extract_dir_file_names(os.path.join(root_dir, dir_name, 'sent_items')))
        logger.debug('files of inbox & sent_items returned')
        return file_names


class Thread:
    def __init__(self):
        self.emails = []
        self.paths = []
        self.users = set()
        self.subjects = set()

    def add_email(self, email, path):
        self.emails.append(email)
        self.paths.append(path)
        self.users.add(email['from'])
        if email['to']:
            email_to = email['to'].replace("\n", "").replace("\t", "").replace(" ", "").split(",")
            self.users.update(email_to)
        if email['subject']:
            self.subjects.add(email['subject'])


def save_file_names(threads):
    logger.info('saving threads ...')
    file_names = [thread.paths for thread in threads if len(thread.paths) > 1]
    with open(os.path.join(settings.BASEPATH, 'data', 'enron_threads.json'), 'w') as f:
        json.dump(file_names, f)


@time_measure()
def main(args):
    threads = []

    for dir_name in sorted(os.listdir(args.path)):
        logger.info('reading emails of %s ...', dir_name)

        file_names = extract_file_names(args.path, dir_name)
        logger.info('number of emails : %d', len(file_names))

        i = 0
        for fname in file_names:
            try:
                with open(fname) as f:
                    data = f.read()
            except UnicodeDecodeError:
                with open(fname, encoding="ISO-8859-1") as f:
                    data = f.read()
            email = Parser().parsestr(data)
            if email['subject']:
                thread_index = find_thread(threads, email)
                if thread_index is not None:
                    threads[thread_index].add_email(email, fname)
                elif email['subject']:
                    new_thread = Thread()
                    new_thread.add_email(email, fname)
                    threads.append(new_thread)
            i += 1
            if i % 100 == 0:
                logger.debug('%d emails read', i)

    save_file_names(threads)
    pprint.pprint({i: threads[i].subjects for i in range(len(threads))})


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Read Enron dataset')
    parser.add_argument("path", help="dataset path")
    args = parser.parse_args()
    main(args)
