import argparse
import json
import os
import pprint
import string
from datetime import timedelta
from email.message import Message
from email.parser import Parser

from networkx import DiGraph

import sampledata
import settings
from cascade.models import Project, ActSequence, CascadeTree, CascadeNode, ParamTypes
from db.managers import DBManager
from settings import logger
from utils.time_utils import time_measure, str_to_datetime


def split_to_words(in_str):
    in_str = in_str.strip()
    for ch in string.punctuation:
        in_str = in_str.replace(ch, ' ')
    return in_str.lower().split()


def jaccard(set1, set2):
    if not set1 or not set2:
        return 0
    else:
        return len(set1 & set2) / len(set1 | set2)


def find_thread(threads, email):
    words = set(split_to_words(email.subject))
    for i in range(len(threads)):
        thread = threads[i]
        if all([jaccard(set(split_to_words(subject)), words) > 0.5 for subject in thread.subjects]):
            if email.from_addr in thread.users or any([u in thread.users for u in email.to_addr]):
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


class Email:
    def __init__(self, path: str):
        try:
            with open(path) as f:
                data = f.read()
        except UnicodeDecodeError:
            with open(path, encoding="ISO-8859-1") as f:
                data = f.read()
        message = Parser().parsestr(data)

        self.from_addr = message['from']
        if message['to'] is None:
            self.to_addr = []
        else:
            self.to_addr = message['to'].replace("\n", "").replace("\t", "").replace(" ", "").split(",")
        self.date = str_to_datetime(message['date'][:-12], '%a, %d %b %Y %H:%M:%S', 'US/Pacific')
        self.subject = message['subject']
        self.path = path


class Thread:
    def __init__(self):
        self.emails = []
        self.users = set()
        self.subjects = set()

    def add_email(self, email):
        self.emails.append(email)
        self.users.add(email.from_addr)
        if email.to_addr:
            self.users.update(email.to_addr)
        if email.subject:
            self.subjects.add(email.subject)


def save_file_names(threads, save_path):
    logger.info('saving threads ...')
    file_names = [[email.dataset_path for email in thread.emails] for thread in threads]
    with open(save_path, 'w') as f:
        json.dump(file_names, f, indent=2)


def extract_threads(path):
    threads = []
    for dir_name in sorted(os.listdir(path)):
        logger.info('reading emails of %s ...', dir_name)

        file_names = extract_file_names(path, dir_name)
        logger.info('number of emails : %d', len(file_names))

        i = 0
        for fname in file_names:
            email = Email(fname)
            if email.subject and email.to_addr:
                thread_index = find_thread(threads, email)
                if thread_index is not None:
                    threads[thread_index].add_email(email)
                else:
                    new_thread = Thread()
                    new_thread.add_email(email)
                    threads.append(new_thread)
            i += 1
            if i % 100 == 0:
                logger.debug('%d emails read', i)

    threads = list(filter(lambda thread: len(thread.emails) > 1, threads))

    logger.info('removing the threads with subject Re, FW, or FWD ...')
    new_threads = []
    for thread in threads:
        # Remove the threads with single email. Remove the threads with subject Re, FW, or FWD.
        if len(thread.emails) < 2 or any(
                split_to_words(subj) in [['re'], ['fw'], ['fwd']] for subj in thread.subjects):
            continue
        new_threads.append(thread)

    return new_threads


def load_threads(save_path):
    logger.info('loading threads from the saved file ...')
    with open(save_path) as f:
        file_names = json.load(f)

    threads = []

    for thr_fnames in file_names:
        thr = Thread()

        for fname in thr_fnames:
            email = Email(fname)
            thr.add_email(email)
        threads.append(thr)

    return threads


def save_cascades(threads, db_name):
    manager = DBManager(db_name)

    # Drop the database.
    manager.client.drop_database(db_name)

    db = manager.db
    users_map = {}  # dictionary from email addresses (usernames) to their _id
    all_post_cascades = []
    all_reshares = []
    act_sequences = {}
    trees = {}

    i = 0
    logger.info('processing %d threads ...', len(threads))

    # Iterate on threads.
    for thread in threads:
        users = [{'username': user} for user in thread.users]
        db.users.insert_many(users)
        users_map.update({user['username']: user['_id'] for user in users})
        posts_map = {}  # dictionary from email addresses to their first posts in this thread.
        reshare_pairs = []  # list of the pairs (post1, post2) where post2 is a reshare of post1

        # Iterate on the emails of this thread in the order of their dates.
        for email in sorted(thread.emails, key=lambda m: m.date):
            if email.from_addr in posts_map:
                from_post = posts_map[email.from_addr]
                if from_post['datetime'] is None:
                    from_post['datetime'] = email.date
            else:
                from_post = {'author_id': users_map[email.from_addr], 'datetime': email.date}
                posts_map[email.from_addr] = from_post

            if email.to_addr:
                for to_addr in email.to_addr:
                    if to_addr not in posts_map:
                        to_post = {'author_id': users_map[to_addr], 'datetime': None}
                        posts_map[to_addr] = to_post
                        reshare_pairs.append((from_post, to_post))

        # Set the datetime of the posts without datetime.
        for source_post, dest_post in reshare_pairs:
            if dest_post['datetime'] is None:
                # The destination time is set 1 day after the source time if there is no datetime.
                dest_post['datetime'] = source_post['datetime'] + timedelta(days=1)

        # Insert the posts into db.
        posts = list(posts_map.values())
        db.posts.insert_many(posts)
        logger.debug('%d posts inserted', len(posts))

        # Extract the activation sequence of the thread.
        times = [post['datetime'] for post in posts]
        first_time, last_time = min(times), max(times)
        act_seq = extract_act_sequence(posts, first_time, last_time)

        # Add the thread reshares to the list of all reshares.
        reshares = [{
            'reshared_post_id': pair[0]['_id'],
            'post_id': pair[1]['_id'],
            'ref_user_id': pair[0]['author_id'],
            'user_id': pair[1]['author_id'],
            'ref_datetime': pair[0]['datetime'],
            'datetime': pair[1]['datetime']
        } for pair in reshare_pairs]
        all_reshares.extend(reshares)

        # Extract the cascade tree of the thread.
        tree = extract_tree(reshares)

        # Insert the cascade into db and add the activation sequence and the tree to the related dictionaries.
        res = db.cascades.insert_one({
            'size': len(thread.users),
            'count': len(posts_map),
            'depth': tree.depth,
            'first_time': first_time,
            'last_time': last_time
        })
        cascade_id = res.inserted_id
        act_sequences[cascade_id] = act_seq
        trees[cascade_id] = tree

        # Add the thread post_cascades to the list of all post_cascades.
        post_cascades = [{'post_id': post['_id'],
                          'cascade_id': cascade_id,
                          'author_id': post['author_id'],
                          'datetime': post['datetime']} for post in posts]
        all_post_cascades.extend(post_cascades)

        i += 1
        if i % 1000 == 0:
            logger.info('%d threads done', i)

    # Extract the graph
    graph = extract_graph(all_reshares)

    # Create a project containing all cascades.
    create_project(act_sequences, graph, trees, db_name)

    # Insert post_cascades and reshares into db.
    logger.info('inserting %d post_cascades', len(all_post_cascades))
    db.postcascades.insert_many(all_post_cascades)
    logger.info('done. inserting %d reshares', len(all_reshares))
    db.reshares.insert_many(all_reshares)
    logger.info('done')


def create_project(act_sequences, graph, trees, db_name):
    """
    Create a project containing all cascades (with random training, validation, and test sets). Then save the graph,
    trees, and activation sequences data in the project data directory.
    :param act_sequences:
    :param graph:
    :param trees:
    :param db_name:
    :return:
    """
    logger.info('creating project "enron-all" ...')
    project_name = 'enron-all'
    sampledata.Command().sample_data(db_name, project_name)
    project = Project(project_name)
    project.save_param(graph, 'graph', ParamTypes.GRAPH)
    project.save_trees(trees)
    project.save_act_sequences(act_sequences)


def extract_graph(all_reshares):
    edges = {(reshare['ref_user_id'], reshare['user_id']) for reshare in all_reshares}
    graph = DiGraph()
    graph.add_edges_from(edges)
    return graph


def extract_act_sequence(posts, first_time, last_time):
    sorted_posts = sorted(posts, key=lambda p: p['datetime'])
    users = [post['author_id'] for post in sorted_posts]
    times = [(post['datetime'] - first_time).total_seconds() / (3600.0 * 24 * 30) for post in
             sorted_posts]  # number of months
    rel_last_time = (last_time - first_time).total_seconds() / (3600.0 * 24 * 30)  # number of months
    return ActSequence(users, times, rel_last_time)


def extract_tree(reshares):
    roots = []
    nodes = {}

    for reshare in sorted(reshares, key=lambda resh: resh['datetime']):
        if reshare['ref_user_id'] not in nodes:
            node = CascadeNode(reshare['ref_user_id'], reshare['ref_datetime'], reshare['reshared_post_id'])
            nodes[node.user_id] = node
            roots.append(node)
        else:
            node = nodes[reshare['ref_user_id']]

        if reshare['user_id'] not in nodes:
            child_node = CascadeNode(reshare['user_id'], reshare['datetime'], reshare['post_id'],
                                     reshare['ref_user_id'])
            nodes[child_node.user_id] = child_node
            node.children.append(child_node)

    return CascadeTree(roots)


@time_measure()
def main(args):
    # threads = extract_threads(args.path)

    save_path = os.path.join(settings.BASE_PATH, 'data', 'enron_threads.json')
    # save_file_names(threads, save_path)
    # pprint.pprint({i: threads[i].subjects for i in range(len(threads))})

    threads = load_threads(save_path)

    save_cascades(threads, args.db)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Read Enron dataset')
    parser.add_argument("-p", "--path", required=True, help="dataset directory path")
    parser.add_argument('-d', '--db', required=True, help="db name in which the documents must be inserted")
    args = parser.parse_args()
    main(args)
