import numpy as np
import copy
import time
import networkx as nx
import random
import matplotlib.pyplot as plt

class Graph:
    """
    A class representing graphs as adjacency lists and implementing various algorithms on the graphs. Graphs in the class are not oriented. 
    Attributes: 
    -----------
    nodes: NodeType
        A list of nodes. Nodes can be of any immutable type, e.g., integer, float, or string.
        We will usually use a list of integers 1, ..., n.
    graph: dict
        A dictionnary that contains the adjacency list of each node in the form
        graph[node] = [(neighbor1, p1, d1), (neighbor2, p2, d2), ...]
        where p1 is the minimal power on the edge (node, neighbor1) and d1 is the distance on the edge
    nb_nodes: int
        The number of nodes.
    nb_edges: int
        The number of edges. 
    """

    def __init__(self, nodes=[]):
        """
        Initializes the graph with a set of nodes, and no edges. 
        Parameters: 
        -----------
        nodes: list, optional
            A list of nodes. Default is empty.
        """
        self.nodes = nodes
        self.graph = dict([(n, []) for n in nodes])
        self.nb_nodes = len(nodes)
        self.nb_edges = 0
    

    def __str__(self):
        """Prints the graph as a list of neighbors for each node (one per line)"""
        if not self.graph:
            output = "The graph is empty"            
        else:
            output = f"The graph has {self.nb_nodes} nodes and {self.nb_edges} edges.\n"
            for source, destination in self.graph.items():
                output += f"{source}-->{destination}\n"
        return output
    
    def add_edge(self, node1, node2, power_min, dist=1):
        """
        Adds an edge to the graph. Graphs are not oriented, hence an edge is added to the adjacency list of both end nodes. 

        Parameters: 
        -----------
        node1: NodeType
            First end (node) of the edge
        node2: NodeType
            Second end (node) of the edge
        power_min: numeric (int or float)
            Minimum power on this edge
        dist: numeric (int or float), optional
            Distance between node1 and node2 on the edge. Default is 1.
        """
        if node1 not in self.graph:
            self.graph[node1] = []
            self.nb_nodes += 1
            self.nodes.append(node1)
        if node2 not in self.graph:
            self.graph[node2] = []
            self.nb_nodes += 1
            self.nodes.append(node2)

        self.graph[node1].append((node2, power_min, dist))
        self.graph[node2].append((node1, power_min, dist))
        self.nb_edges += 1
    
    def route2(node, current, length):
        if node == dest:
            if length < self.min_length:
                self.min_length = length
                resultat[0] = current
            return None
        if node in visit:
            return None
        else:
            visit.append(node)
            for neighbor in self.graph[node]:
                if neighbor[1] <= power:
                    route2(neighbor[0], current+[neighbor[0]], length + neighbor[2])

    def get_path_with_power(self, src, dest, power):
        """
        The algorithm goes through every nodes and for each node,
        it browses every edges thus the complexity is in O(V*E)
        where V and E represent respectively the number of nodes and edges.
        Only takes road with the minimal distance
        Args:
            src (int): the source of the road
            dest (int): the destination of the road
            power (int): the power of the truck
        Returns:
            list: Contains the different nodes which compose the road
        """
        resultat = [[]]
        visit = []
        self.min_length = float('inf')
        route2(src, [src], 0)
        if resultat[0]:
            return resultat[0]
        else:
            return None
    
    def route2(node, current):
        if node in visit:
            return None
        else:
            visit.add(node)
            for voisin in self.graph[node]:
                route(voisin[0], current+[voisin[0]])

    def connected_components_set(self):
        """
        The result should be a set of frozensets (one per component),
        For instance, for network01.in:
        {frozenset({1, 2, 3}), frozenset({4, 5, 6, 7})}
        """
        result = set()
        for node in self.graph.keys():
            if node not in result:
                visit = set()
                route(node, [node])
                result.add(frozenset(visit))
        return result

    def route3(node, current, power):
            if node == dest:
                if self.minimal_power > power:
                    self.minimal_power = power
                    resultat[0] = current
                    resultat[1] = power
                return
            if node in current[:len(current)-1]: 
                return 
            else: 
                for neighbor in self.graph[node]:
                    temp = max(power, neighbor[1])
                    route3(neighbor[0], current+[neighbor[0]], temp)
    
    def min_power(self, src, dest):
        """
        Should return path, min_power.
        Same complexity as get_path_with_power, O(V*E)
        """
        resultat = [[], 0]
        self.minimal_power = float('inf')
        route3(src, [src], 0)
        if resultat:
            return resultat
        else:
            return None


def graph_from_file(filename):
    """
    Reads a text file and returns the graph as an object of the Graph class.

    The file should have the following format:
        The first line of the file is 'n m'
        The next m lines have 'node1 node2 power_min dist' or 'node1 node2 power_min' (if dist is missing, it will be set to 1 by default)
        The nodes (node1, node2) should be named 1..n
        All values are integers.

    Parameters:
    -----------
    filename: str
        The name of the file

    Outputs:
    -----------
    g: Graph
        An object of the class Graph with the graph from file_name.
    """
    with open(filename, "r") as file:
        n, m = map(int, file.readline().split())  # picks line
        g = Graph(range(1, n+1))
        for _ in range(m):
            edge = list(map(int, file.readline().split()))
            if len(edge) == 3:  # no information about the distance, dist = 1 by default
                node1, node2, power_min = edge
                g.add_edge(node1, node2, power_min)
            elif len(edge) == 4:
                node1, node2, power_min, dist = edge
                g.add_edge(node1, node2, power_min, dist)
            else:
                raise Exception("Format incorrect")
    return g

def road(filename):
    with open(filename, "r") as file:
        m = int(file.readline())
        D = {}
        i = 0
        for _ in range(m):
            R = file.readline().split()
            s, d, p = int(R[0]), int(R[1]), int(R[2])
            D[i] = [s, d, p]
            i += 1
    return D

def open_trucks(filename):
    with open(filename, "r") as file:
        m = int(file.readline())
        D = {}
        i = 0
        for _ in range(m):
            R = file.readline().split()
            power, cost = int(R[0]), int(R[1])
            D[i] = [power, cost]
            i+=1
    return D

def plot_graph(G):
    g = G.graph
    M = nx.Graph()
    n_nodes = len(g.keys())
    for k in range(1, n_nodes):
        M.add_node(k)
    for key, value in g.items():        
        source = key
        for neighbour in value:           
            destination, power, distance = neighbour
            M.add_edge(source, destination)
    plt.figure()
    nx.draw(M, with_labels=True)
    plt.savefig("delivery_network/graph")
    plt.close()



class Union:
    """
    the goal of this class is to represent trees with two lists: an adjacence one and an other to know how deep are the nodes.

    n_nodes: int
       number of nodes.
    T: list
        The list for the trees
    D: list
        The list with the depth of the nodes
    """
    def __init__(self, n_nodes=1):
        """
        Initialization for the union (number of nodes, and no edges).
        """
        self.n_nodes = n_nodes
        self.T = [i for i in range(n_nodes)]
        self.D = [0 for i in range(n_nodes)]
 
    def represantant(self, x):
        """
        This function permits to have the representant of x in the Union
        x is a node (type=int)
        """
        U = self.T
        if x == self.T[x]:
            return x
        return (Union.represantant(self, U[x]))

    def depth(self, x):
        a = Union.represantant(self, x)
        return (self.D[a])

    def fus(self, x, y):
        """
        It permits to merge the two parts of x and y
        x,y are nodes
        """
        rx, ry = Union.represantant(self, x), Union.represantant(self, y)
        Rx, Ry = Union.depth(self, x), Union.depth(self, y)
        if Rx > Ry:
            self.T[ry] = rx
            self.D[rx] += 1
        else:
            self.T[rx] = ry
            self.D[ry] += 1

def kruskal(g):
    """
    Transforms a graph into a covering tree minimum using the Union_set class.
    The complexity of this function is thank to the optimised Union_set operation in O((n+a)log(n))
    where n is the number of nodes and a the number of edges

    g is a Graph

    return a Graph g0 of the minimum tree covering of g
    """
    n = g.nodes
    n_nodes = g.nb_nodes
    m = g.nb_edges
    L = []
    for elt1 in n:
        adj = g.graph[elt1]
        for elt2 in adj:
            L.append([elt2[1], elt1-1, elt2[0]-1])
    L.sort()
    u = Union(n_nodes)
    g = Graph([i+1 for i in range(n_nodes)])
    i, j = 0, 0
    while i < n_nodes and j < 2*m:
        p, x, y = int(L[j][0]), L[j][1], L[j][2]
        if Union.represantant(u, x) != Union.represantant(u, y):
            Union.fus(u, x, y)
            Graph.add_edge(g, x+1, y+1, p)
            i += 1
        j += 1
    return (g)

class TreeNode:
    def __init__(self, value, parent=None, enfant=[], power=0):#puissance de l'arête avec son parent
        self.value = value
        self.parent = parent
        self.enfant = enfant
        self.power = power

class Tree:
    def __init__(self, root):
        self.root = root
        self.nodes = {root.value: TreeNode(root.value, root.parent, root.enfant, root.power)}
    
    def arbre(valeurDepart, G):
        root = TreeNode(valeurDepart)
        tree = Tree(root)
        visite = set()
        pile = [valeurDepart]
        while len(pile) > 0:
            value = pile.pop()
            visite.add(value)
            for L in G.graph[value]:
                (nv, b, c) = L
                if nv not in visite:
                    pile.append(nv)
                    tree.nodes[nv] = TreeNode(nv, value, [], b)
                    tree.nodes[value].enfant.append(nv)
        return tree

def ancetre_commun(node1, node2, A):
    N1 = TreeNode(node1, A.nodes[node1].parent, A.nodes[node1].enfant, A.nodes[node1].power)
    N2 = TreeNode(node2, A.nodes[node2].parent, A.nodes[node2].enfant, A.nodes[node2].power)
    L = []
    n1 = N1.value
    n2 = N2.value
    while n1 != A.root.value:
        L.append(N1.value)
        N1 = A.nodes[N1.parent]
        n1 = N1.value
    L.append(A.root.value)
    while n2 not in L:
        N2 = A.nodes[N2.parent]
        n2 = N2.value
    return n2

def power_min(node1, node2, A):
    N1 = TreeNode(node1, A.nodes[node1].parent, A.nodes[node1].enfant, A.nodes[node1].power)
    N2 = TreeNode(node2, A.nodes[node2].parent, A.nodes[node2].enfant, A.nodes[node2].power)
    L = []
    M = []
    n1 = N1.value
    n2 = N2.value
    s1 = 0
    s2 = 0
    a_c = ancetre_commun(node1, node2, A)
    while n1 != a_c:
        L.append(N1.value)
        s1 += N1.power
        N1 = A.nodes[N1.parent]
        n1 = N1.value
    while n2 != a_c:
        M.append(N2.value)
        s2 += N2.power
        N2 = A.nodes[N2.parent]
        n2 = N2.value
    L.append(a_c)
    M.reverse()
    traj = L + M
    return (traj, s1 + s2)


def trajet_possible(s, d, C, A):
    (traj, power) = power_min(s, d, A)
    if power <= C[0]:
        return True
    else:
        return False

def profit(s, d, R):
    for i in R.keys():
        if R[i][0] == s and R[i][1] == d:
            return R[i][2]

def rapport(s, d, R, C):
    p = profit(s, d, R)
    return p/C[1]

def maximisation(R, T, A):
    n = len(R)
    m = len(T)
    dico = {}
    resultat = np.zeros((n, m))
    for i in R.keys():
        for j in T.keys():
            if trajet_possible(R[i][0], R[i][1], T[j], A):
                resultat[i, j] = rapport(R[i][0], R[i][1], R, T[j])
            else:
                resultat[i, j] = 0
    for traj in R.keys():
        cam = np.argmax(resultat[traj, :])
        dico[traj] = T[cam]
    return dico
