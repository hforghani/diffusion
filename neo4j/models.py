from bson import ObjectId
from networkx import DiGraph
from py2neo import Graph
import time

from settings import logger


class Neo4jGraph:
    def __init__(self, label):
        self.label = label
        self.graph = Graph(user='neo4j', password='123')

    def parents(self, user_id):
        cypher = 'match (n:{0}{{_id:"{1}"}})<--(m:{0}) return m._id'.format(self.label, str(user_id))
        data = self.graph.run(cypher).data()
        parents = [ObjectId(item['m._id']) for item in data]
        return parents

    def parents_of_list(self, user_ids):
        cypher = 'match (n:{0})<--(m:{0}) ' \
                 'where n._id in [{1}] ' \
                 'return m._id'.format(self.label, ','.join(['"%s"' % str(uid) for uid in user_ids]))
        data = self.graph.run(cypher).data()
        parents = [ObjectId(item['m._id']) for item in data]
        return parents

    def children(self, user_id):
        cypher = 'match (n:{0}{{_id:"{1}"}})-->(m:{0}) return m._id'.format(self.label, str(user_id))
        data = self.graph.run(cypher).data()
        children = [ObjectId(item['m._id']) for item in data]
        return children

    def children_of_list(self, user_ids):
        cypher = 'match (n:{0})-->(m:{0}) ' \
                 'where n._id in [{1}] ' \
                 'return m._id'.format(self.label, ','.join(['"%s"' % str(uid) for uid in user_ids]))
        data = self.graph.run(cypher).data()
        children = [ObjectId(item['m._id']) for item in data]
        return children

    def parents_count(self, user_id):
        cypher = 'match (n:{0}{{_id:"{1}"}})<--(m:{0}) return count(m)'.format(self.label, str(user_id))
        data = self.graph.run(cypher).data()[0]
        return data['count(m)']

    def children_count(self, user_id):
        cypher = 'match (n:{0}{{_id:"{1}"}})-->(m:{0}) return count(m)'.format(self.label, str(user_id))
        data = self.graph.run(cypher).data()[0]
        return data['count(m)']

    def create_memm_train_graph(self, ids_list):
        digraph = DiGraph()

        #data = self.graph.run('match (n:{0}) '
        #                 'optional match (n:{0}) <-- (m:{0}) '
        #                 'optional match (n:{0}) --> (s:{0}) '
        #                 'optional match (r:{0}) --> (s:{0}) '
        #                 'where n._id in {} '
        #                 'return n._id,m._id,r._id,s._id '.format(ids_list)).data()

        digraph.add_nodes_from(ids_list)
        ids_str_list = [str(uid) for uid in ids_list]

        t0 = time.time()
        data = self.graph.run('match (n:{0}) <-- (m:{0}) '
                              'where n._id in {1} '
                              'return n._id, m._id'.format(self.label, ids_str_list)).data()
        digraph.add_nodes_from([ObjectId(row['m._id']) for row in data])
        digraph.add_edges_from([(ObjectId(row['m._id']), ObjectId(row['n._id'])) for row in data])
        logger.info('%d parent edges added to graph in %d s' % (len(data), time.time() - t0))

        t0 = time.time()
        data = self.graph.run('match (n:{0}) --> (o:{0}) '
                              'where n._id in {} '
                              'return n._id, o._id'.format(ids_str_list)).data()
        children = [row['o._id'] for row in data]
        digraph.add_nodes_from([ObjectId(c) for c in children])
        digraph.add_edges_from([(ObjectId(row['n._id']), ObjectId(row['o._id'])) for row in data])
        logger.info('%d child edges added to graph in %d s' % (len(data), time.time() - t0))

        t0 = time.time()
        data = self.graph.run('match (p:{0}) --> (o:{0}) '
                              'where o._id in {} '
                              'return p._id, o._id'.format(children)).data()
        digraph.add_nodes_from([ObjectId(row['p._id']) for row in data])
        digraph.add_edges_from([(ObjectId(row['p._id']), ObjectId(row['o._id'])) for row in data])
        logger.info('%d brother edges added to graph in %d s' % (len(data), time.time() - t0))

        return digraph
