import sys
sys.path.append(r"D:/OneDrive/Documents/11-Codes/overflow/02_RANS")
from maillage import *


class Parametres:
    """Classe pour les parametres de la cellule"""
    def __init__(self,T:float=200,p:float=1e5,v:np.array=np.array([0,0]))->None:
        self.T = T
        self.p = p
        self.v = v
        self.rho = p/(287.05*T)
    def get_var(self,var:str)->float:
        """Renvoie la variable de la cellule"""
        if var == "T":
            return self.T
        elif var == "p":
            return self.p
        elif var == "v":
            return self.v
        elif var == "rho":
            return self.rho
        else:
            raise ValueError(f"Variable {var} non reconnue")


class Sim():
    """Classe pour la simulation"""
    def __init__(self, filename:str)->None:
        self.mesh = Mesh(filename)
        self.cell_param = [Parametres() for _ in range(len(self.mesh.cells))]


# Face
    def get_face_param(self,face_index:int)->Parametres: 
        """Calcul des parametres de la face"""
        cells = self.mesh.cells
        face = self.mesh.faces[face_index]
        index_owner = face.owner
        index_neighbour = face.neighbour
        Vo = cells[face.owner].volume
        Vn = cells[face.neighbour].volume
        gn = Vo/(Vo + Vn)
        go = 1-gn
        T = gn*self.cell_param[index_neighbour].T + go*self.cell_param[index_owner].T
        v = gn*self.cell_param[index_neighbour].v + go*self.cell_param[index_owner].v
        p = gn*self.cell_param[index_neighbour].p + go*self.cell_param[index_owner].p
        return Parametres(T,p,v)

# Cell
    def get_grad_cell(self,cell_index, var)->np.array:
        """Calcul du gradient d'une variable dans la cellule"""

        grad = np.zeros(2)
        cell_faces = self.mesh.cells[cell_index].faces
        for i in range(len(self.faces)):
            f = self.mesh.faces[cell_faces[i]]
            sign = f.owner == cell_index
            grad += sign * f.surface * self.get_face_param(f.indice_global).get_var(var)
        grad /= self.volume
            
        return grad


# Mesh

    def compute_gradient(self, var:str)->np.array:
        """Calcul du gradient d'une variable dans la cellule"""

        grad = np.zeros((len(self.mesh.cells),2))
        for i in range(len(self.mesh.faces)):
            f = self.mesh.faces[i]
            flux_f = self.get_face_param(i).get_var(var)*f.surface
            if f.owner != -1:
                grad[f.owner] += flux_f
                grad[f.neighbour] -= flux_f
            else:
                grad[f.neighbour] -= flux_f
        grad[:,0] /= self.mesh.cell_volume
        grad[:,1] /= self.mesh.cell_volume
                
        return grad 
    
    def plot(self, var:str,ax = plt.subplots()[1])->None:
        """Plot the mesh with the variable"""
        for i in range(len(self.mesh.cells)):
            cell = self.mesh.cells[i]
            x,y = cell.centroid
            ax.scatter(x,y,s=10,c='k')
    

if __name__ == "__main__":
    
    sim = Sim("D:/OneDrive/Documents/11-Codes/overflow/02_RANS/circle_mesh.dat")
    gradient = sim.compute_gradient("T")
    sim.plot("T")
    plt.show()