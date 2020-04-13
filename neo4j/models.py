from bson import ObjectId
import psutil
from py2neo import Graph


class Neo4jGraph:
    MAX_RAM_PERCENT = 85    # Stop collecting parents and children cache if the memory usage is higher than this threshold
    CRITICAL_RAM_PERCENT = 90  # Remove some cache entries if the memory usage is higher than this threshold

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
        uid_int = self.__id_to_int(user_id)
        if uid_int in self.__parents:
            return len(self.__parents)
        else:
            cypher = 'match (n:{0}{{_id:"{1}"}})<--(m:{0}) return count(m)'.format(self.label, str(user_id))
            data = self.__graph.run(cypher).data()[0]
            return data['count(m)']

    def children_count(self, user_id):
        uid_int = self.__id_to_int(user_id)
        if uid_int in self.__children:
            return len(self.__children)
        else:
            cypher = 'match (n:{0}{{_id:"{1}"}})-->(m:{0}) return count(m)'.format(self.label, str(user_id))
            data = self.__graph.run(cypher).data()[0]
            return data['count(m)']

    def get_or_fetch_parents(self, uid):
        uid_int = self.__id_to_int(uid)
        if uid_int in self.__parents:
            return [self.__int_to_id[u] for u in self.__parents[uid_int]]
        else:
            parents = self.parents(uid)
            if psutil.virtual_memory().percent < self.MAX_RAM_PERCENT:
                self.__parents[uid_int] = [self.__id_to_int(u) for u in parents]
            else:
                self.__check_critical_ram()
            return parents

    def get_or_fetch_children(self, uid):
        uid_int = self.__id_to_int(uid)
        if uid_int in self.__children:
            return [self.__int_to_id[u] for u in self.__children[uid_int]]
        else:
            children = self.children(uid)
            if psutil.virtual_memory().percent < self.MAX_RAM_PERCENT:
                self.__children[uid_int] = [self.__id_to_int(u) for u in children]
            else:
                self.__check_critical_ram()

        return children

    def __id_to_int(self, uid):
        if str(uid) in self.__id_to_int_map:
            return self.__id_to_int_map[str(uid)]
        else:
            i = len(self.__int_to_id)
            self.__int_to_id.append(uid)
            self.__id_to_int_map[str(uid)] = i
            return i

    def __check_critical_ram(self):
        if psutil.virtual_memory().percent > self.CRITICAL_RAM_PERCENT:
            main_dict = self.__parents if len(self.__parents) > len(self.__children) else self.__children
            for i in range(len(main_dict) - 1, -1, -1):
                del main_dict[i]
                if psutil.virtual_memory().percent <= self.MAX_RAM_PERCENT:
                    break
            return True
        return False