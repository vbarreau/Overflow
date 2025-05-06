import sys
sys.path.append(r"D:/OneDrive/Documents/11-Codes/overflow/02_RANS")
from maillage import *
import time

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
        
    def set_var(self,var:str,value:float)->None:
        """Set the value of a variable in the cell"""
        if var == "T":
            self.T = value
        elif var == "p":
            self.p = value
        elif var == "v":
            self.v = value
        elif var == "rho":
            self.rho = value
        else:
            raise ValueError(f"Variable {var} non reconnue")


class Sim():
    """Classe pour la simulation"""
    def __init__(self, **kwargs)->None:
        if len(kwargs) == 1 and isinstance(kwargs['filename'], str):
            filename = kwargs['filename']
            t = time.time()
            self.mesh = Mesh(filename = filename)
            print(f"Temps de chargement du maillage : {time.time()-t} s")
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

    def compute_gradient(self, var:str,chrono=False)->np.array:
        """Calcul du gradient d'une variable dans la cellule"""
        if chrono:
            t = time.time()
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
        if chrono:
            print(f"Temps de calcul du gradient : {time.time()-t} s")
                
        return grad 
    
    def set_var(self,var:str,values:np.array,x,y)->None:
        """Set the value of a variable in the mesh at a given point"""
        return None
    
    def set_CI(self,var,*args)-> None:
        """ Set the initial condition of the simulation from either a matrix or a function"""
        if len(args) == 1 and isinstance(args[0],np.ndarray):
            xmin,xmax,ymin,ymax = self.mesh.span()
            DX = (xmax-xmin)/args[0].shape[0]
            DY = (ymax-ymin)/args[0].shape[1]
            for cell in self.mesh.cells:
                x = cell.centroid[0]
                y = cell.centroid[1]
                nx = int((x-xmin)/DX)
                ny = int((y-ymin)/DY)
                if nx < 0 or nx >= args[0].shape[0] or ny < 0 or ny >= args[0].shape[1]:
                    raise ValueError(f"Cellule {cell.indice_global} en dehors de la matrice")
                self.cell_param[cell.indice_global].set_var(var,args[0][nx,ny])
        elif len(args) == 1 and callable(args[0]):
            for cell in self.mesh.cells:
                x = cell.centroid[0]
                y = cell.centroid[1]
                self.cell_param[cell.indice_global].set_var(var,args[0](x,y))
        return None

    # Plotting
    
    def plot(self, var:str,ax = None,chrono=False)->None:
        """Plot the mesh with the variable"""
        if chrono:
            t = time.time()
        if ax is None:
            fig, ax = plt.subplots()
        
        x = [cell.centroid[0] for cell in self.mesh.cells]
        y = [cell.centroid[1] for cell in self.mesh.cells]
        values = [self.cell_param[i].get_var(var) for i in range(len(self.mesh.cells))]
        
        sc = ax.scatter(x, y, c=values, cmap='viridis', s=10)
        plt.colorbar(sc, ax=ax, label=var)
            
        if chrono:
            print(f"Temps de réalisation du graphique : {time.time()-t} s")

def radial_CI(x,y,A=1)->float:
    """Fonction de condition initiale radiale"""

    r = np.sqrt((x-20)**2 + (y-10)**2)
    return A*np.exp(-(r-2)**2/4)
    

if __name__ == "__main__":
    
    sim = Sim(filename = "D:/OneDrive/Documents/11-Codes/overflow/02_RANS/circle_mesh.dat")
    sim.set_CI("T",radial_CI)
    gradient = sim.compute_gradient("T")
    fig , ax = plt.subplots()
    # sim.mesh.plot_mesh(ax=ax)
    sim.plot("T",ax=ax)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title("Gradient de T")

    plt.show()