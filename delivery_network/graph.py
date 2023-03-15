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
        graph[node] = [(neighbor1, p1, d1), (neighbor1, p1, d1), ...]
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

        def route(node, current, length):
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
                        route(neighbor[0], current+[neighbor[0]], length + neighbor[2])
        route(src, [src], 0)
        if resultat[0]:
            return resultat[0]
        else:
            return None

    def connected_components_set(self):
        """
        The result should be a set of frozensets (one per component),
        For instance, for network01.in:
        {frozenset({1, 2, 3}), frozenset({4, 5, 6, 7})}
        """
        result = set()

        def route(node, current):
            if node in visit:
                return None
            else:
                visit.add(node)
                for voisin in self.graph[node]:
                    route(voisin[0], current+[voisin[0]])

        for node in self.graph.keys():
            if node not in result:
                visit = set()
                route(node, [node])
                result.add(frozenset(visit))
        return result
    
    def min_power(self, src, dest):
        """
        Should return path, min_power.
        Same complexity as get_path_with_power, O(V*E)
        """
        resultat = [[], 0]
        self.minimal_power = float('inf')

        def road(node, current, power):
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
                    road(neighbor[0], current+[neighbor[0]], temp)
        road(src, [src], 0)
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
    with open(filename, "r") as f:
        doc = f.read()  # Picks line
        L = doc.split('\n')  # creates graph with a good format
        n = L[0].split(' ')[0]
        nodes = [i+1 for i in range(int(n))]
        g = Graph(nodes)
        for i in range (1,len(L)):
            values = L[i].split(' ')
            if len(values) == 4:  # the distance is mentioned
                node1, node1, power_min, dist = values
                g.add_edge(int(node1), int(node2), int(power_min), int(dist))
            elif len(values) == 3:  # no information about the distance
                node1, node2, power_min = values
                g.add_edge(int(node1), int(node2), int(power_min))
            else:
                return "error"
        return g


def plot_graph(g):
    from graphviz import Source
    t = """ graph{ """
    visit = []
    for clef, valeurs in g.graph.items():
        visit.append(clef)
        for neighbor in valeurs:
            if neighbor[0] not in visit:
                t += str(clef) + "--" + str(neighbor[0]) + """[label= "p = """ + str(neighbor[1]) + ";d = " + str(neighbor[2]) + """ "]""" + ";" + "\n"
    t += """}"""
    s = Source(t, filename="Graph.gv", format="png")
    s.view()
