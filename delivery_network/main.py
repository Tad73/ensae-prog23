from graph import Graph, graph_from_file
import time
import random as rd
g = Graph([])
g.add_edge("Paris", "Palaiseau", 4, 20)
#print (g)

data_path = "input/"
file_name = "network.1.in"

g = graph_from_file(data_path + file_name)
n = len(g.graph)

# Q10
"""
Average of time to calculate the minimum power of one road
"""
T0 = time.perf_counter()
for i in range(100):
    g.min_power(rd.randrange(1, n), rd.randrange(1, n))
T1 = time.perf_counter()
print((T1-T0)/100)
