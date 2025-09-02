import sys
import numpy as np
sys.path.append(r"D:/OneDrive/Documents/11-Codes/overflow/02_RANS")
from maillage_test import mesh
from colorama import Fore, Style, init
from geom import *
import pytest

# Initialize colorama
init(autoreset=True)


# l'objet mesh est importé depuis le fichier maillage.py
#  __
# | _|__
# |_\|__|

# mesh.complete_plot() pour afficher le maillage
def test_surface_triangle_area():
    node1 = np.array([0, 0])
    node2 = np.array([1, 0])
    node3 = np.array([0, 1])
    area = surface_triangle(node1, node2, node3)
    assert np.isclose(area, 0.5)

def test_centre_of_points():
    nodes = np.array([[0, 0], [2, 0], [0, 2]])
    c = centre(nodes)
    assert np.allclose(c, [2/3, 2/3])

def test_point_toward_cell():
    assert points_toward_cell(np.array([0.5,0.5]),np.array([0.5,1]),np.array([0,-1])) == True

    C0 = mesh.cells[0]
    C1 = mesh.cells[1]
    f1 = mesh.faces[1]
    assert points_toward_cell(C0.centroid,f1.centroid,f1.get_normal(mesh.nodes)) == True
    assert points_toward_cell(C1.centroid,f1.centroid,f1.get_normal(mesh.nodes)) == False

# Face tests

def test_face_normal_and_length():
    nodes = np.array([[0, 0], [1, 0], [0, 1]])
    f = Face(0, [0, 1], [0], nodes)
    normal = f.get_normal(nodes)
    assert np.allclose(normal, [0, 1])
    length = f.length(nodes)
    assert np.isclose(length, 1.0)



def test_mesh_span_and_volume():
    nodes = np.array([[0, 0], [1, 0], [0, 1]])
    faces = np.array([Face(0, [0, 1], [0], nodes), Face(1, [1, 2], [0], nodes), Face(2, [0, 2], [0], nodes)])
    cells = np.array([Cell(0, [0, 1, 2], [0, 1, 2], [], nodes)])
    mesh = Mesh(cells=cells, nodes=nodes, faces=faces)
    xmin, xmax, ymin, ymax = mesh.span()
    assert xmin == 0 and xmax == 1 and ymin == 0 and ymax == 1
    assert np.isclose(mesh.cell_volume[0], 0.5)

def test_face_commune():
    nodes = np.array([[0, 0], [1, 0], [0, 1]])
    faces = np.array([
        Face(0, [0, 1], [0, 1], nodes), 
        Face(1, [1, 2], [0]   , nodes), 
        Face(2, [0, 2], [1]   , nodes)])
    cells = np.array([
        Cell(0, [0, 1], [0, 1, 2], [1], nodes),
        Cell(1, [0, 2], [0, 1, 2], [0], nodes)
    ])
    mesh2 = Mesh(cells=cells, nodes=nodes, faces=faces)
    f = face_commune(cells[0], cells[1], mesh2)
    assert f.indice_global == 0
    
    C2 = mesh.cells[2]
    C1 = mesh.cells[1]
    f4 = mesh.faces[4]
    assert face_commune(C2, C1, mesh) == f4


def test_mesh_error():
    with pytest.raises(MeshError):
        nodes = np.array([[0, 0], [1, 0]])
        Face(0, [0, 1], [], nodes)

def test_face_getCD():
    nodes = np.array([[0, 0], [1, 0], [0, 1],[1, 1]])
    c0 = Cell(0, [0, 1, 2], [0, 1, 2], [], nodes)
    f0 = Face(0, [0, 1], [0], nodes)
    cells = np.array([c0])
    C, D = f0.getCD(cells)
    assert np.allclose(C, c0.centroid)
    assert np.allclose(D, f0.centroid)
    f1 = Face(1,[0,2],[0],nodes)
    f2 = Face(2,[1,2],[0,1],nodes)
    f3 = Face(3,[1,3],[1],nodes)
    f4 = Face(4,[2,3],[1],nodes)
    c1 = Cell(1,[2,3,4],[1,2,3],[0],nodes)
    cells = np.array([c0,c1])

    C, D = f2.getCD(cells)
    assert np.allclose(C, c0.centroid)
    assert np.allclose(D, c1.centroid)
    assert np.isclose(f2.distance_des_centres(cells),2**0.5/3)

def test_Ef_Tf():
    nodes = np.array([[0,0],[0,2],[2,0],[2,1]])
    f0 = Face(0,[0,1],[0],nodes)
    f1 = Face(1,[0,2],[0],nodes)
    f2 = Face(2,[1,2],[0,1],nodes)
    f3 = Face(3,[1,3],[1],nodes)
    f4 = Face(4,[2,3],[1],nodes)
    c0 = Cell(0,[0,1,2],[0,1,2],[1],nodes)
    c1 = Cell(1,[2,3,4],[1,2,3],[0],nodes)
    cells = np.array([c0,c1])
    S2 = f2.surface
    assert np.allclose(S2,np.array([2,2]))
    Ef,Tf = f2.Ef_Tf(cells)
    assert np.allclose(Ef+Tf,S2)
    assert np.isclose(np.dot(Ef,S2),np.linalg.norm(Ef)*np.linalg.norm(S2))

### Tests for Cell methods

def test_cell_volume_triangle():
    nodes = np.array([[0, 0], [1, 0], [0, 1]])
    c = Cell(0, [0, 1, 2], [0, 1, 2], [], nodes)
    assert np.isclose(c.volume, 0.5)

def test_cell_contains_point():
    nodes = np.array([[0, 0], [1, 0], [0, 1]])
    c = Cell(0, [0, 1, 2], [0, 1, 2], [], nodes)
    assert c.contains(0.2, 0.2, nodes)
    assert not c.contains(2, 2, nodes)

def test_cell_sort_nodes():
    nodes = np.array([[0, 0], [1, 0], [0, 1]])
    c = Cell(0, [0, 1, 2], [0, 1, 2], [], nodes)
    # Should be sorted counter-clockwise
    assert set(c.nodes_index) == {0, 1, 2}