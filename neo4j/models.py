from bson import ObjectId
from py2neo import Graph


class Neo4jGraph:
    def __init__(self, label):
        self.label = label
        self.__graph = Graph(user='neo4j', password='123')
        self.__parents = {}
        self.__children = {}
        self.__id_to_int_map = {}
        self.__int_to_id = []

    def parents(self, user_id):
        cypher = 'match (n:{0}{{_id:"{1}"}})<--(m:{0}) return m._id'.format(self.label, str(user_id))
        data = self.__graph.run(cypher).data()
        parents = [ObjectId(item['m._id']) for item in data]
        return parents

    def parents_of_list(self, user_ids):
        cypher = 'match (n:{0})<--(m:{0}) ' \
                 'where n._id in [{1}] ' \
                 'return m._id'.format(self.label, ','.join(['"%s"' % str(uid) for uid in user_ids]))
        data = self.__graph.run(cypher).data()
        parents = [ObjectId(item['m._id']) for item in data]
        return parents

    def children(self, user_id):
        cypher = 'match (n:{0}{{_id:"{1}"}})-->(m:{0}) return m._id'.format(self.label, str(user_id))
        data = self.__graph.run(cypher).data()
        children = [ObjectId(item['m._id']) for item in data]
        return children

    def children_of_list(self, user_ids):
        cypher = 'match (n:{0})-->(m:{0}) ' \
                 'where n._id in [{1}] ' \
                 'return m._id'.format(self.label, ','.join(['"%s"' % str(uid) for uid in user_ids]))
        data = self.__graph.run(cypher).data()
        children = [ObjectId(item['m._id']) for item in data]
        return children

    def parents_count(self, user_id):
        cypher = 'match (n:{0}{{_id:"{1}"}})<--(m:{0}) return count(m)'.format(self.label, str(user_id))
        data = self.__graph.run(cypher).data()[0]
        return data['count(m)']

    def children_count(self, user_id):
        cypher = 'match (n:{0}{{_id:"{1}"}})-->(m:{0}) return count(m)'.format(self.label, str(user_id))
        data = self.__graph.run(cypher).data()[0]
        return data['count(m)']

    def get_or_fetch_parents(self, uid):
        uid_int = self.__id_to_int(uid)
        if uid_int in self.__parents:
            return [self.__int_to_id[u] for u in self.__parents[uid_int]]
        else:
            parents = self.parents(uid)
            self.__parents[uid_int] = [self.__id_to_int(u) for u in parents]
            return parents

    def get_or_fetch_children(self, uid):
        uid_int = self.__id_to_int(uid)
        if uid_int in self.__children:
            return [self.__int_to_id[u] for u in self.__children[uid_int]]
        else:
            children = self.children(uid)
            self.__children[uid_int] = [self.__id_to_int(u) for u in children]
            return children

    def __id_to_int(self, uid):
        if str(uid) in self.__id_to_int_map:
            return self.__id_to_int_map[str(uid)]
        else:
            i = len(self.__int_to_id)
            self.__int_to_id.append(uid)
            self.__id_to_int_map[str(uid)] = i
            return i
