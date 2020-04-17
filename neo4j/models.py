from bson import ObjectId
import psutil
from py2neo import Graph
from settings import logger
import settings


class Neo4jGraph:
    MAX_RAM_PERCENT = 85    # Stop collecting parents and children cache if the memory usage is higher than this threshold
    CRITICAL_RAM_PERCENT = 92  # Remove some cache entries if the memory usage is higher than this threshold

    def __init__(self, label):
        self.label = label
        self.__graph = Graph(user=settings.NEO4J_USER, password=settings.NEO4J_PASS)
        self.__parents = {}
        self.__children = {}
        self.__par_count = {}
        self.__child_count = {}
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
        if uid_int in self.__par_count:
            return self.__par_count[uid_int]
        else:
            cypher = 'match (n:{0}{{_id:"{1}"}})<--(m:{0}) return count(m)'.format(self.label, str(user_id))
            count = self.__graph.run(cypher).data()[0]['count(m)']
            self.__par_count[uid_int] = count
            return count

    def children_count(self, user_id):
        uid_int = self.__id_to_int(user_id)
        if uid_int in self.__child_count:
            return self.__child_count[uid_int]
        else:
            cypher = 'match (n:{0}{{_id:"{1}"}})-->(m:{0}) return count(m)'.format(self.label, str(user_id))
            count = self.__graph.run(cypher).data()[0]['count(m)']
            self.__child_count[uid_int] = count
            return count

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
            self.__par_count[uid_int] = len(parents)
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
            self.__child_count[uid_int] = len(children)
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
            if main_dict:
                count = 0
                for i in range(len(main_dict) - 1, -1, -1):
                    if i in main_dict:
                        del main_dict[i]
                        count += 1
                        if psutil.virtual_memory().percent <= self.MAX_RAM_PERCENT:
                            break
                logger.info('%d dict entries deleted to free RAM!', count)
            return True
        return False
