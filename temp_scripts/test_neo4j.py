from py2neo import Graph, Node, NodeMatcher, Relationship, RelationshipMatcher


graph = Graph(user='neo4j', password='123')
print('database: {}'.format(graph.database))

tx = graph.begin()
nodes = []
for name in ["Alice", "Bob", "Carol"]:
    n = Node('Person', name=name)
    nodes.append(n)
    tx.create(n)

for i in range(len(nodes) - 1):
    tx.create(Relationship(nodes[i], 'PARENT_OF', nodes[i + 1]))
tx.commit()

rmatcher = RelationshipMatcher(graph)
for r in rmatcher.match():
    print(r)