import sys
sys.path.append("delivery_network/")

import unittest
from graph import graph_from_file

class Test_GraphLoading2(unittest.TestCase):
    def test_network5(self):
        g = graph_from_file("input/network.05.in")
        self.assertEqual(g.nb_nodes, 4)
        self.assertEqual(g.nb_edges, 6)

if __name__ == '__main__':
    unittest.main()
