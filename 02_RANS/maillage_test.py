
import sys
import unittest
import numpy as np
from maillage import *
sys.path.append(r"D:/OneDrive/Documents/11-Codes/overflow/02_RANS")

class TestMaillage(unittest.TestCase):

    def test_surface_triangle(self):
        node1 = np.array([0, 0])
        node2 = np.array([1, 0])
        node3 = np.array([0, 1])
        self.assertAlmostEqual(surface_triangle(node1, node2, node3), 0.5)

    def test_cell_volume(self):
        nodes = np.array([[0, 0], [1, 0], [0, 1]])
        cell = Cell(0, [], nodes, [])
        self.assertAlmostEqual(cell.volume, 0.5)

    def test_cell_centroid(self):
        nodes = np.array([[0, 0], [1, 0], [0, 1]])
        cell = Cell(0, [], nodes, [])
        np.testing.assert_array_almost_equal(cell.centroid, [1/3, 1/3])

    def test_cell_sort_nodes(self):
        nodes = np.array([[0, 0], [1, 0], [0, 1]])
        cell = Cell(0, [], nodes, [])
        sorted_nodes = np.array([[0, 0], [1, 0], [0, 1]])
        np.testing.assert_array_almost_equal(cell.nodes, sorted_nodes)

        nodes = np.array([[0, 0], [1, 0], [0, 1],[0,2], [1,2]])
        cell = Cell(0, [], nodes, [])
        sorted_nodes = np.array([[0, 0], [1, 0],[1,2],[0,2], [0, 1]])
        np.testing.assert_array_almost_equal(cell.nodes, sorted_nodes)

    def test_mesh_initialization(self):
        nodes1 = np.array([[0, 0], [1, 0], [0, 1]])
        nodes2 = np.array([[1, 0], [1, 1], [0, 1]])
        cell1 = Cell(0, [], nodes1, [])
        cell2 = Cell(1, [], nodes2, [])
        mesh = Mesh(np.array([cell1, cell2]))
        self.assertEqual(mesh.size, 2)

if __name__ == "__main__":
    unittest.main()
