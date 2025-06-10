import sys
import numpy as np
sys.path.append(r"D:/OneDrive/Documents/11-Codes/overflow/02_RANS")
from etat import *
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


# Tests de la classe Parametres
def test_param_init():
    """Test de l'initialisation de la classe Parametres"""
    param = Parametres()
    assert isinstance(param, Parametres)
    assert param.T == 300
    assert param.p == 1e5
    assert param.vx == 0
    assert param.vy == 0
    assert (param.gradT == np.ones(2)).all()
    assert (param.gradp == np.zeros(2)).all()
    assert (param.gradvx == np.zeros(2)).all()
    assert (param.gradvy == np.zeros(2)).all()
    assert (param.S          == np.zeros((2, 2)) ).all() # Tenseur des deformations
    assert (param.Omega      == np.zeros((2, 2)) ).all() # Tenseur de vorticité
    assert (param.tau        == np.zeros((2, 2)) ).all() # tenseur de turbulences
    
    # New attributes
    assert param.k          == 1.0          # Turbulent kinetic energy
    assert param.w          == 1.0          # Specific dissipation rate
    assert param.w_bar      == 1.0
    assert param.Nu_t       == 0.0           # Turbulent viscosity
    assert param.f_beta     == 0.0           # Beta function
    assert param.xi_omega   == 0.0           # Xi omega parameter
    assert param.sigma_d    == 0.0           # Scalar parameter
    assert (param.gradk     == np.zeros(2)).all()   # Gradient of turbulent kinetic energy
    assert (param.gradw     == np.zeros(2)).all()   # Gradient of specific dissipation rate


def test_set_var():
    """Test de la fonction set_var"""
    param = Parametres()
    param.set_var("T", 300)
    param.set_var("p", 2e5)
    param.set_var("vx", np.array([1, 1]))
    param.set_var("vy", np.array([2, 2]))
    param.set_var("gradT", np.array([0.1, 0.1]))
    param.set_var("gradP", np.array([0.2, 0.2]))
    param.set_var("gradVx", np.array([0.3, 0.3]))
    param.set_var("gradVy", np.array([0.4, 0.4]))
    param.set_var("grad_rho", np.array([0.5, 0.5]))
    param.set_var("S", np.array([[1, 2], [3, 4]]))
    param.set_var("Omega", np.array([[5, 6], [7, 8]]))
    param.set_var("tau", np.array([[9, 10], [11, 12]]))

    assert param.get_var("T") == 300
    assert param.get_var("p") == 2e5
    np.testing.assert_array_equal(param.get_var("vx"), np.array([1, 1]))
    np.testing.assert_array_equal(param.get_var("vy"), np.array([2, 2]))
    np.testing.assert_array_equal(param.get_var("gradT"), np.array([0.1, 0.1]))
    np.testing.assert_array_equal(param.get_var("gradP"), np.array([0.2, 0.2]))
    np.testing.assert_array_equal(param.get_var("gradVx"), np.array([0.3, 0.3]))
    np.testing.assert_array_equal(param.get_var("gradVy"), np.array([0.4, 0.4]))
    np.testing.assert_array_equal(param.get_var("grad_rho"), np.array([0.5, 0.5]))
    np.testing.assert_array_equal(param.get_var("S"), np.array([[1, 2], [3, 4]]))
    np.testing.assert_array_equal(param.get_var("Omega"), np.array([[5, 6], [7, 8]]))
    np.testing.assert_array_equal(param.get_var("tau"), np.array([[9, 10], [11, 12]]))


def test_set_cell_tensor():
    """Test the set_cell_tensor method for setting tensors for a specific cell."""
    param = Parametres()
    param.vx = 1
    param.vy = 2
    param.gradvx = np.array([1, 2])
    param.gradvy = np.array([2, 3])
    param.set_cell_tensor()

    expected_S = np.array([[1, 2], [2, 3]])
    expected_Omega = np.zeros((2, 2))

    assert np.allclose(param.S, expected_S)
    assert np.allclose(param.Omega, expected_Omega)

    param.gradvy = np.array([3, 4])
    param.set_cell_tensor()
    expected_S = np.array([[1, 2.5], [2.5, 4]])
    expected_Omega[0,1] = -0.5
    expected_Omega[1,0] =  0.5

    assert np.allclose(param.S, expected_S)
    assert np.allclose(param.Omega, expected_Omega)


def test_update_values() :
    param = Parametres()
    param.k = 1
    param.w = 2
    param.gradk = np.array([1, 2])
    param.gradw = np.array([3, 4])

    param.update_values()

    expected_Nu_t = 0.5
    expected_sigma_d = SIGMA_DO
    expected_tau = -2/3*np.eye(2)

    assert param.w_bar == param.w
    assert param.Nu_t == pytest.approx(expected_Nu_t, rel=1e-2)
    assert param.sigma_d == pytest.approx(expected_sigma_d, rel=1e-2)
    np.testing.assert_array_almost_equal(param.tau, expected_tau)

    param.gradvx = np.array([1, 2])
    param.gradvy = np.array([2, 3])
    param.set_cell_tensor() # S = [[1,2], [2,3]]
    param.update_values()
    assert param.w_bar == CLIM*20
    assert param.Nu_t == 1/param.w_bar
    expected_tau = -2/3*np.eye(2) + 4/35*param.S
    np.testing.assert_array_almost_equal(param.tau, expected_tau)



 # Tests de la classe Sim

def test_init():
    """Test de l'initialisation de la classe Sim"""
    state = Etat(mesh=mesh)
    assert isinstance(state, Etat)
    assert state.mesh.size == len(mesh.cells)
    assert len(state.cell_param) == len(mesh.cells)
    assert state.cell_param[0].T == 300
    assert state.cell_param[0].p == 1e5
    np.testing.assert_array_equal(state.cell_param[0].vx, 0)


def test_set_var():
    """Test de la fonction set_var"""
    state = Etat(mesh=mesh)
    state.set_var("T", 0, 0.2, 0.1)
    state.set_var("p", 2e5, 0.1, 0.1)
    state.set_var('vx',1,0.1,0.1)
    state.set_var('vy',1,0.1,0.1)
    assert state.cell_param[0].vx == 1
    assert state.cell_param[0].vy == 1    
    assert state.cell_param[0].p == pytest.approx(2e5)
    assert state.cell_param[0].T == pytest.approx(0)

    for i in range(1, 3):
        assert pytest.approx(state.cell_param[i].T) == 300
        assert pytest.approx(state.cell_param[i].p) == 1e5
        assert state.cell_param[i].vx == 0
        assert state.cell_param[i].vy == 0

    # On remet les variables à leur valeur initiale
    state.set_var("T", 300,0 )
    state.set_var("p", 1e5,[0.1,0.1])
    state.set_var('vx',0,0.1,0.1)
    state.set_var('vy',0,0.1,0.1)
    for i in range(0, 3):   
        assert pytest.approx(state.cell_param[i].T) == 300
        assert pytest.approx(state.cell_param[i].p) == 1e5
        assert state.cell_param[i].vx == 0
        assert state.cell_param[i].vy == 0

def test_get_face_param():
    """Test de la fonction get_face_param"""
    state = Etat(mesh=mesh)
    face_param = state.get_face_param(0)
    assert isinstance(face_param, Parametres)
    assert face_param.T == 300
    assert face_param.p == 1e5
    assert face_param.vx == 0

    state.set_var("T", 3, 0)
    state.set_var("T", 2, 1)
    state.set_var("T", 1, 2)
    state.set_var("T", 1, 3)

    face1_param = state.get_face_param(1)
    face3_param = state.get_face_param(3)
    face4_param = state.get_face_param(4)

    assert pytest.approx(face1_param.T) == 5 / 2
    assert pytest.approx(face3_param.T) == 5 / 3
    assert pytest.approx(face4_param.T) == 5 / 3

def test_get_grad_cell():
    state = Etat(mesh=mesh)

    grad1_init = state.get_grad_cell(1,'T')
    np.testing.assert_array_almost_equal(grad1_init,vecteur_nul)

    state.set_var("T", 3, 0)
    state.set_var("T", 2, 1)
    state.set_var("T", 1, 2)
    state.set_var("T", 1, 3)

    grad1 = state.get_grad_cell(1,'T')
    expected = np.array([-5/3,-5/3])
    np.testing.assert_array_almost_equal(grad1,expected)

def test_compute_gradient():
    state = Etat(mesh=mesh)
    gradT = state.compute_gradient('T')
    for i in range(len(gradT)):
        assert (gradT[i]==vecteur_nul).all()
    state.set_var("T", 3, 0)
    state.set_var("T", 2, 1)
    state.set_var("T", 1, 2)
    state.set_var("T", 1, 3)

    gradT = state.compute_gradient('T')
    expected = np.array([-5/3,-5/3])
    np.testing.assert_array_almost_equal(gradT[1] , expected)

    state.set_var("vx", 3, 0)
    state.set_var("vx", 2, 1)
    state.set_var("vx", 1, 2)
    state.set_var("vx", 1, 3)
    gradVx = state.compute_gradient('vx')
    np.testing.assert_array_almost_equal(gradVx[1] , expected)


def test_tensors():
    state = Etat(mesh=mesh)
    state.set_var("vx", 3, 0)
    state.set_var("vx", 2, 1)
    state.set_var("vx", 1, 2)
    state.set_var("vx", 1, 3)
    state.compute_gradient()
    for i in range(1,state.mesh.size):
        grad = getattr(state.cell_param[i], 'gradvx')
        assert (grad!=0).any() ,f"Gradient nul en cellule {i}\n"
    state.compute_tensors()

    s1 = getattr(state.cell_param[1], 'S')
    print(f'S en cellule 1 :\n{s1}\n ')
    W1 = getattr(state.cell_param[1], 'Omega')
    print(f'Omega en cellule 1 :\n{W1}\n ')
    assert (s1!=0).any()
    #  tester avec des vrais valeurs


def test_update_all_param():
    """Test the update_all_param method of the Sim class."""
    state = Etat(mesh=mesh)
    state.set_var("vx", 0, 0)
    state.set_var("vx", 0.5, 1)
    state.set_var("vx", 0, 2)
    state.set_var("vx", 1, 3)

    state.set_var("vy", 0, 0)
    state.set_var("vy", -0.5, 1)
    state.set_var("vy", -1, 2)
    state.set_var("vy", 0, 3)

    state.set_var("k", 10, 0)
    state.set_var("k", 0.5, 1)
    state.set_var("k", 0, 2)
    state.set_var("k", 0.5, 3)

    state.set_var("w", 2, 0)
    state.set_var("w", 0.5, 1)
    state.set_var("w", 1, 2)
    state.set_var("w", 1, 3)

    state.update_all_param()
    print("    Vx,   Vy,     k,    w,     gradVx,       gradVy,         gradk  ,         gradw :")
    for i in range( state.mesh.size):
        cell = state.cell_param[i]
        print(f"{i}   {cell.vx:.2f}, {cell.vy:.2f}, {cell.k:.2f}, {cell.w:.2f} "
              f"  [{' '.join(f'{val:.2f}' for val in getattr(cell,'gradvx'))}] "
              f"  [{' '.join(f'{val:.2f}' for val in getattr(cell,'gradvy'))}] "
              f"  [{' '.join(f'{val:.2f}' for val in getattr(cell,'gradk'))}] "
              f"  [{' '.join(f'{val:.2f}' for val in getattr(cell,'gradw'))}]")

    tau1 = getattr(state.cell_param[1], 'tau')
    assert tau1[0][0] == pytest.approx(-0.162, rel=1e-2)
    assert tau1[0][1] == pytest.approx(0, rel=1e-2)
    assert tau1[1][0] == pytest.approx(0, rel=1e-2)
    assert tau1[1][1] == pytest.approx(-0.50, rel=1e-2)

def test_sub():
    """Test the sub method of the Sim class."""
    state = VarEtat(mesh=mesh)
    state.set_var("T", 3, 0)
    state.set_var("T", 2, 1)
    state.set_var("T", 1, 2)
    state.set_var("T", 1, 3)

    other_sim = VarEtat(mesh=mesh)
    other_sim.set_var("T", 0, 0)
    result = other_sim - state

    assert isinstance(result, Etat)
    assert len(result.cell_param) == len(state.cell_param)
    assert result.cell_param[0].T == -3

if __name__=="__main__" :
    
    fig,ax = plt.subplots(1, 1, figsize=(10, 10))
    etat_ini = Etat(mesh=mesh)
    etat_ini.set_var("T", 3, 0)
    etat_ini.set_var("T", 2, 1)
    etat_ini.set_var("T", 1, 2)
    etat_ini.set_var("T", 1, 3)
    etat_ini.plot('T',ax=ax,point_size=100)
    mesh.complete_plot(ax=ax)
    plt.show()




