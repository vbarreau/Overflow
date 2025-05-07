import sys
import numpy as np
import pytest

sys.path.append(r"D:/OneDrive/Documents/11-Codes/overflow/02_RANS")
from maillage import *

node1 = np.array([0, 0])
node2 = np.array([1, 0])
node3 = np.array([0, 1])
nodes = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [0, 2], [1, 2], [2, 1], [2, 0]])
f0 = Mesh.Face(0, [0, 1], [0], nodes)
f1 = Mesh.Face(1, [1, 2], [0, 1], nodes)
f2 = Mesh.Face(2, [0, 2], [0], nodes)
f3 = Mesh.Face(3, [1, 3], [1, 3], nodes)
f4 = Mesh.Face(4, [2, 3], [1, 2], nodes)
f5 = Mesh.Face(5, [2, 4], [2], nodes)
f6 = Mesh.Face(6, [4, 5], [2], nodes)
f7 = Mesh.Face(7, [3, 5], [2], nodes)
f8 = Mesh.Face(8, [3, 6], [3], nodes)
f9 = Mesh.Face(9, [6, 7], [3], nodes)
f10 = Mesh.Face(10, [1, 7], [3], nodes)

faces = np.array([f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10])

cell0 = Mesh.Cell(0, [0, 1, 2], [0, 1, 2], [1], nodes)
cell1 = Mesh.Cell(1, [1, 3, 4], [1, 3, 2], [0, 2], nodes)
cell2 = Mesh.Cell(2, [4, 5, 6, 7], [2, 3, 4, 5], [1], nodes)
cell3 = Mesh.Cell(3, [3, 8, 9, 10], [1, 3, 6, 7], [1], nodes)

cells = np.array([cell0, cell1, cell2, cell3])

mesh = Mesh(cells=cells, nodes=nodes, faces=faces)


def test_surface_triangle():
    assert surface_triangle(node1, node2, node3) == pytest.approx(0.5)


def test_cell_volume():
    assert cell0.volume == pytest.approx(0.5)


def test_cell_centroid():
    np.testing.assert_array_almost_equal(cell0.centroid, [1 / 3, 1 / 3])
    np.testing.assert_array_almost_equal(cell2.centroid, [0.5, 1.5])
    np.testing.assert_array_almost_equal(cell3.centroid, [1.5, 0.5])


def test_cell_sort_nodes():
    cell = Mesh.Cell(0, [0, 1, 2], [0, 1, 2], [], nodes)
    sorted_nodes = np.array([0, 1, 2])
    np.testing.assert_array_almost_equal(cell.nodes_index, sorted_nodes)

    cell = Mesh.Cell(0, [0, 1, 2, 3], [0, 1, 2, 3, 4], [], nodes)
    sorted_nodes = np.array([0, 1, 3, 4, 2])
    np.testing.assert_array_almost_equal(cell.nodes_index, sorted_nodes)


def test_mesh_initialization():
    assert mesh.size == len(cells)


def test_length():
    out = f0.length(nodes)
    expected = 1.0
    assert out == pytest.approx(expected)

    out = f1.length(nodes)
    expected = np.sqrt(2)
    assert out == pytest.approx(expected)


def test_get_normal():
    n0 = f0.get_normal(nodes)
    n1 = f1.get_normal(nodes)

    expected0 = np.array([0, 1])
    expected1 = np.array([1, 1]) / np.sqrt(2)
    np.testing.assert_array_almost_equal(n0, expected0)
    np.testing.assert_array_almost_equal(n1[0] * expected1[1] - n1[1] * expected1[0], 0)
    assert np.linalg.norm(n1) == pytest.approx(1)


def test_set_owner():
    if f1.get_normal(nodes)[0] < 0:
        assert f1.owner == 1
    else:
        assert f1.owner == 0
    assert f0.owner == 0
    assert f2.owner == 0 


def test_contains():
    assert cell0.contains(0.1, 0.1, nodes)
    assert cell3.contains(1.5, 0.5, nodes)
    assert not cell3.contains(1.5, 1.5, nodes)


def test_find_cell():
    """Test de la fonction find_cell"""
    cell = mesh.find_cell(0.1, 0.1)
    assert cell == 0
    cell = mesh.find_cell(1.5, 0.5)
    assert cell == 3
    cell = mesh.find_cell(2, 2)
    assert cell is None

def test_surface():
    Sf = f0.surface
    assert type(Sf)==np.ndarray
    assert np.linalg.norm(Sf)==1

    Sf1 = f1.surface
    assert (np.linalg.norm(Sf1)==np.sqrt(2))
    assert Sf1[0] == -1
    assert Sf1[1] == -1
    assert f1.owner == 1


