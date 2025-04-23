from objets import *


def test_set_V() :
    V = zeros([5,5,2])

    V[1,:] = array([1,0])
    grid = Grid(5,5)

    grid.set_V_grid(V)
    assert (grid.get_v()[1,0] == array([1,0])).all()
    

    
test_set_V()