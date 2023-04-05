from graph import Graph, graph_from_file, Tree, TreeNode, power_min, kruskal
import time
import random as rd
g = Graph([])
g.add_edge("Paris", "Palaiseau", 4, 20)
#print (g)

data_path = "input/"
file_name = "network.2.in"

g = graph_from_file(data_path + file_name)
n = len(g.graph)

# Q10
"""
Average of time to calculate the minimum power of one road
"""
T2 = time.perf_counter()
g.min_power(rd.randrange(1, n), rd.randrange(1, n))
T3 = time.perf_counter()
print((T3-T2))

#Q15
g0 = kruskal(g)
A = Tree.arbre(1, g0)

T0 = time.perf_counter()

power_min(rd.randrange(1, n), rd.randrange(1, n), A)
T1 = time.perf_counter()
print((T1-T0))
