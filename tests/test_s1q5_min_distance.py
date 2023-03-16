import sys
sys.path.append("delivery_network/")

import unittest
from graph import Graph, graph_from_file

class test_Graph(unittest.TestCase):
    def test_network4(self):
        g = graph_from_file("input/network.04.in")
        self.assertEqual(g.get_path_with_power(1, 4, 11), [1, 4])
        self.assertEqual(g.get_path_with_power(1, 4, 7), [1, 2, 3, 4])

if __name__ == '__main__':
    unittest.main()
