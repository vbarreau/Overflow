import sys
import numpy as np
sys.path.append(r"D:/OneDrive/Documents/11-Codes/overflow/02_RANS")
from sim import *
from maillage_test import mesh

import pytest
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)


# l'objet mesh est importé depuis le fichier maillage.py
#  __
# |__|__
# |_\|__|

# mesh.complete_plot() pour afficher le maillage

vecteur_nul = np.zeros(2)

def test_init():
    """Test de l'initialisation de la classe Sim"""
    sim = Sim(mesh=mesh)
    assert isinstance(sim, Sim)
    assert sim.mesh.size == len(mesh.cells)
    assert len(sim.cell_param) == len(mesh.cells)
    assert sim.cell_param[0].T == 200
    assert sim.cell_param[0].p == 1e5
    np.testing.assert_array_equal(sim.cell_param[0].v, vecteur_nul)


def test_set_var():
    """Test de la fonction set_var"""
    sim = Sim(mesh=mesh)
    sim.set_var("T", 0, 0.2, 0.1)
    sim.set_var("p", 2e5, 0.1, 0.1)
    sim.set_var("v", np.array([1, 1]), 0.1, 0.1)
    assert np.allclose(sim.cell_param[0].get_var("v"), np.array([1, 1]))
    assert sim.cell_param[0].get_var("p") == pytest.approx(2e5)
    assert sim.cell_param[0].get_var("T") == pytest.approx(0)

    for i in range(1, 3):
        assert pytest.approx(sim.cell_param[i].get_var("T")) == 200
        assert pytest.approx(sim.cell_param[i].get_var("p")) == 1e5
        np.testing.assert_array_almost_equal(sim.cell_param[i].get_var("v"), vecteur_nul)

    # On remet les variables à leur valeur initiale
    sim.set_var("T", 200,0 )
    sim.set_var("p", 1e5,[0.1,0.1])
    sim.set_var("v", np.array([0, 0]), 0.1, 0.1)
    for i in range(0, 3):   
        assert pytest.approx(sim.cell_param[i].get_var("T")) == 200
        assert pytest.approx(sim.cell_param[i].get_var("p")) == 1e5
        np.testing.assert_array_almost_equal(sim.cell_param[i].get_var("v"), vecteur_nul)

def test_get_face_param():
    """Test de la fonction get_face_param"""
    sim = Sim(mesh=mesh)
    face_param = sim.get_face_param(0)
    assert isinstance(face_param, Parametres)
    assert face_param.T == 200
    assert face_param.p == 1e5
    np.testing.assert_array_almost_equal(face_param.v, vecteur_nul)

    sim.set_var("T", 3, 0)
    sim.set_var("T", 2, 1)
    sim.set_var("T", 1, 2)
    sim.set_var("T", 1, 3)

    face1_param = sim.get_face_param(1)
    face3_param = sim.get_face_param(3)
    face4_param = sim.get_face_param(4)

    assert pytest.approx(face1_param.T) == 5 / 2
    assert pytest.approx(face3_param.T) == 5 / 3
    assert pytest.approx(face4_param.T) == 5 / 3

def test_get_grad_cell():
    sim = Sim(mesh=mesh)

    grad1_init = sim.get_grad_cell(1,'T')
    np.testing.assert_array_almost_equal(grad1_init,vecteur_nul)

    sim.set_var("T", 3, 0)
    sim.set_var("T", 2, 1)
    sim.set_var("T", 1, 2)
    sim.set_var("T", 1, 3)

    grad1 = sim.get_grad_cell(1,'T')
    expected = np.array([-5/3,-5/3])
    np.testing.assert_array_almost_equal(grad1,expected)

def test_compute_gradient():
    sim = Sim(mesh=mesh)
    gradT = sim.compute_gradient('T')
    for i in range(len(gradT)):
        assert (gradT[i]==vecteur_nul).all()
    sim.set_var("T", 3, 0)
    sim.set_var("T", 2, 1)
    sim.set_var("T", 1, 2)
    sim.set_var("T", 1, 3)

    gradT = sim.compute_gradient('T')
    expected = np.array([-5/3,-5/3])
    assert (gradT[1] == expected).all()

# if __name__=="__main__" :
    
#     mesh.complete_plot()
#     plt.show()




