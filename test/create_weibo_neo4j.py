from py2neo import Graph, Node, NodeMatcher, Relationship
import time
from cascade.weibo import read_uidlist
from settings import logger


def get_nodes(graph, uid_list):
    uid_index_map = {str(uid_list[i]): i for i in range(len(uid_list))}
    nodes = {}
    matcher = NodeMatcher(graph)
    for u in matcher.match('User'):
        ind = uid_index_map[u['_id']]
        nodes[ind] = u
    nodes = [nodes[i] for i in range(len(uid_list))]
    return nodes


def create_relations(graph, relations_file, uidlist_file, user_ids=None):
    t0 = time.time()

    uid_list = read_uidlist(uidlist_file)

    matcher = NodeMatcher(graph)
    if len(matcher.match('User')) >= len(uid_list):
        # Extract the nodes.
        logger.info('fetching nodes ...')
        nodes = get_nodes(graph, uid_list)
    else:
        # Create each node if does not exist.
        logger.info('creating nodes ...')
        tx = graph.begin()
        nodes = []
        for uid in uid_list:
            n = Node('User', _id=str(uid))
            tx.merge(n, 'User', '_id')
            nodes.append(n)
        logger.info('commiting ...')
        tx.commit()
        logger.info('{} nodes created'.format(len(nodes)))

    user_ids_str = None
    if user_ids is not None:
        user_ids_str = [str[u] for u in user_ids]

    logger.info('reading relationships ...')
    i = 0
    rel_count = 0
    tx = graph.begin()

    # Read the relations from the file and create them.
    with open(relations_file, encoding='utf-8', errors='ignore') as f:
        f.readline()
        line = f.readline()

        while line:
            line = line.strip().split()
            u1_i = int(line[0])
            u1 = nodes[u1_i]
            n = int(line[1])

            for j in range(n):
                rel_count += 1
                u2_i = int(line[2 + j * 2])
                u2 = nodes[u2_i]
                if user_ids_str is None or u1['_id'] in user_ids_str or u2['_id'] in user_ids_str:
                    rel = Relationship(u2, 'PARENT_OF', u1)
                    tx.create(rel)

                if rel_count % 1000000 == 0:
                    tx.commit()
                    logger.info('%d relations created' % rel_count)
                    tx = graph.begin()

            i += 1
            if i % 1000 == 0:
                logger.info('%d lines read' % i)

            line = f.readline()

    tx.commit()
    logger.info('%d relations created' % rel_count)

    logger.info('graph extraction time: %.2f min', (time.time() - t0) / 60.0)


if __name__ == '__main__':
    graph = Graph(user='neo4j', password='123')
    relations_file = 'E:\\datasets\\Weibo-Net-Tweet\\weibo_network.txt'
    uidlist_file = 'E:\\datasets\\Weibo-Net-Tweet\\uidlist.txt'
    create_relations(graph, relations_file, uidlist_file)
