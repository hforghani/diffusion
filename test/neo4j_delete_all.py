from py2neo import Graph, NodeMatcher
graph = Graph(user='neo4j', password='123')
matcher = NodeMatcher(graph)
nodes_count = len(matcher.match())
step = 10000

while True:
    res = graph.run('match (n) with n limit {} detach delete n return count(n)'.format(step)).data()[0]
    count = res['count(n)']
    nodes_count -= step
    print('{} deleted. {} remained'.format(count, nodes_count))
    if count == 0:
        break
