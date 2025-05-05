
import sys
import unittest
import numpy as np
sys.path.append(r"D:/OneDrive/Documents/11-Codes/overflow/02_RANS")
from maillage import *

node1 = np.array([0, 0])
node2 = np.array([1, 0])
node3 = np.array([0, 1])
nodes = np.array([[0, 0], [1, 0], [0, 1],[1,1],[0,2], [1,2],[2,1],[2,0]])
f0 = Mesh.Face(0, [0,1],[0] ,nodes)
f1 = Mesh.Face(1, [1,2],[0,1] ,nodes)
f2 = Mesh.Face(2, [0,2],[0] ,nodes)
f3 = Mesh.Face(3, [1,3],[1,3] ,nodes)
f4 = Mesh.Face(4, [2,3],[1,2] ,nodes)
f5 = Mesh.Face(5, [2,4],[2] ,nodes)
f6 = Mesh.Face(6, [4,5],[2] ,nodes)
f7 = Mesh.Face(7, [3,5],[2] ,nodes)
f8 = Mesh.Face(8, [3,6],[3] ,nodes)
f9 = Mesh.Face(9, [6,7],[3] ,nodes)
f10 = Mesh.Face(10, [1,7],[3] ,nodes)

faces = np.array([f0,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10])

cell0 = Mesh.Cell(0, [0,1,2],[0,1,2], [1],nodes)
cell1 = Mesh.Cell(1, [1,3,4],[1,3,2], [0,2],nodes)
cell2 = Mesh.Cell(2, [4,5,6,7],[2,3,4,5], [1],nodes)
cell3 = Mesh.Cell(3, [3,7,9,10],[1,3,6,7], [1],nodes)

cells = np.array([cell0, cell1,cell2,cell3])

mesh = Mesh(cells =cells,nodes = nodes, faces = faces)

class TestMaillage(unittest.TestCase):

    def test_surface_triangle(self):
        self.assertAlmostEqual(surface_triangle(node1, node2, node3), 0.5)

    def test_cell_volume(self):
        self.assertAlmostEqual(cell0.volume, 0.5)

    def test_cell_centroid(self):
        np.testing.assert_array_almost_equal(cell0.centroid, [1/3, 1/3])
        np.testing.assert_array_almost_equal(cell2.centroid, [0.5, 1.5])
        np.testing.assert_array_almost_equal(cell3.centroid, [1.5, 0.5])

    def test_cell_sort_nodes(self):
        cell = Mesh.Cell(0, [0,1,2],[0,1,2], [],nodes)
        sorted_nodes = np.array([0,1,2])
        np.testing.assert_array_almost_equal(cell.nodes, sorted_nodes)

        cell = Mesh.Cell(0, [0,1,2,3],[0,1,2,3,4], [],nodes)
        sorted_nodes = np.array([0, 1,3,4, 2])
        np.testing.assert_array_almost_equal(cell.nodes, sorted_nodes)

    def test_mesh_initialization(self):
        
        self.assertEqual(mesh.size, len(cells))


    def test_length(self):
        out = f0.length(nodes)
        expected = 1.0
        self.assertEqual(out, expected)

        out  = f1.length(nodes)
        expected = np.sqrt(2)
        self.assertEqual(out, expected)
    
    def test_get_normal(self) :
        n0 = f0.get_normal(nodes)
        n1 = f1.get_normal(nodes)

        expected0 = np.array([0,1])
        expected1 = np.array([1,1])/np.sqrt(2)
        np.testing.assert_array_almost_equal(n0, expected0) 
        np.testing.assert_array_almost_equal(n1[0]*expected1[1]-n1[1]*expected1[0],0)
        self.assertAlmostEqual(np.linalg.norm(n1), 1)

    def test_set_owner(self):
        if f1.get_normal(nodes)[0] < 0 :
            self.assertEqual(f1.owner, 1)
        else:
            self.assertEqual(f1.owner, 0)

    

if __name__ == "__main__":
    try :
        unittest.main()
    except :
        ax = mesh.complete_plot()
        # plt.show()