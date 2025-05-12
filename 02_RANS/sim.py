import sys
sys.path.append(r"D:/OneDrive/Documents/11-Codes/overflow/02_RANS")
from maillage import *
import time

XI          = 0
f_beta      = 1
alpha       = 13 / 25
beta_0      = 0.0708
beta        = beta_0
beta_star   = 0.09
sigma       = 0.5
sigma_star  = 0.6
sigma_do    = 1 / 8
C_lim       = 7 / 8


class Parametres:
    """Classe pour les parametres de la cellule"""
    def __init__(self, T: float = 200, p: float = 1e5, vx: float = 0, vy: float = 0) -> None:
        self.T          = T
        self.p          = p
        self.vx         = vx
        self.vy         = vy
        self.rho        = p / (287.05 * T)
        self.gradT      = np.ones(2)
        self.gradP      = np.zeros(2)
        self.gradVx     = np.zeros(2)
        self.gradVy     = np.zeros(2)
        self.grad_rho   = np.zeros(2)
        self.S          = np.zeros((2, 2))  # Tenseur des deformations
        self.Omega      = np.zeros((2, 2))  # Tenseur de vorticité
        self.tau        = np.zeros((2, 2))  # tenseur de turbulences
        
        # New attributes
        self.k          = 0.0          # Turbulent kinetic energy
        self.w          = 0.0          # Specific dissipation rate
        self.w_bar      = 1.0
        self.Nu_t       = 0.0           # Turbulent viscosity
        self.f_beta     = 0.0           # Beta function
        self.xi_omega   = 0.0           # Xi omega parameter
        self.epsilon    = 0.0           # Dissipation rate
        self.l          = 0.0           # Turbulence length scale
        self.sigma_d    = 0.0           # Scalar parameter
        self.grad_k     = np.zeros(2)   # Gradient of turbulent kinetic energy
        self.grad_w     = np.zeros(2)   # Gradient of specific dissipation rate

    def get_var(self, var: str):
        """Renvoie la variable de la cellule"""
        if var == "T":
            return self.T
        elif var == "p":
            return self.p
        elif var == "vx":
            return self.vx
        elif var == "vy":
            return self.vy
        elif var == "v" or var == "V":
            return np.array([self.vx, self.vy])
        elif var == "rho":
            return self.rho
        elif var == "gradT":
            return self.gradT
        elif var == "gradP":
            return self.gradP
        elif var == "gradVx":
            return self.gradVx
        elif var == "gradVy":
            return self.gradVy
        elif var == "grad_rho":
            return self.grad_rho
        elif var == "S":
            return self.S
        elif var == "Omega":
            return self.Omega
        elif var == "tau":
            return self.tau
        elif var == "k":
            return self.k
        elif var == "w":
            return self.w
        elif var == "Nu_t":
            return self.Nu_t
        elif var == "f_beta":
            return self.f_beta
        elif var == "xi_omega":
            return self.xi_omega
        elif var == "grad_k":
            return self.grad_k
        elif var == "grad_omega" or var == 'grad_w' or var == 'gradw':
            return self.grad_w
        elif var == "epsilon":
            return self.epsilon
        elif var == "l":
            return self.l
        elif var == "sigma_d":
            return self.sigma_d
        else:
            raise ValueError(f"Variable {var} non reconnue")
        
    def w_bar_min(self) -> float:
        """Renvoie la valeur minimale de w_bar"""
        s2 = (self.S*self.S).sum()
        return C_lim*np.sqrt(2*s2/beta_star)

    def set_var(self, var: str, value) -> None:
        """Set the value of a variable in the cell"""
        if isinstance(value, (float, int, np.float64)):
            if var == "T":
                self.T = value
            elif var == "p":
                self.p = value
            elif var == "rho":
                self.rho = value
            elif var == "vx":
                self.vx = value
            elif var == "vy":
                self.vy = value
            elif var == "k":
                self.k = value
            elif var == "w":
                self.w = value
            elif var == "Nu_t":
                self.Nu_t = value
            elif var == "f_beta":
                self.f_beta = value
            elif var == "xi_omega":
                self.xi_omega = value
            elif var == "epsilon":
                self.epsilon = value
            elif var == "l":
                self.l = value
            elif var == "sigma_d":
                self.sigma_d = value
            else:
                raise ValueError(f"Variable {var} non reconnue, type {type(value)}")

        elif isinstance(value, np.ndarray) and value.shape == (2,):
            if var == "gradT":
                self.gradT = value
            elif var == "gradP" or var == "gradp":
                self.gradP = value
            elif var == "gradVx" or var == "gradvx":
                self.gradVx = value
            elif var == "gradVy" or var == "gradvy":
                self.gradVy = value
            elif var == "grad_rho" or var == "gradrho":
                self.grad_rho = value
            elif var == "grad_k" or var == "gradk":
                self.grad_k = value
            elif var == "grad_omega" or var == 'grad_w' or var == 'gradw':
                self.grad_w = value
            elif var == "v" or var == "V":
                self.vx = value[0]
                self.vy = value[1]
            else:
                raise ValueError(f"Variable {var} non reconnue, type {type(value)}")

        elif isinstance(value, np.ndarray) and value.shape == (2, 2):
            if var == "S":
                self.S = value
            elif var == "Omega":
                self.Omega = value
            elif var == "tau":
                self.tau = value
            else:
                raise ValueError(f"Variable {var} non reconnue, type {type(value)}")

        else:
            raise ValueError(f"type {type(value)} non reconnue pour {var}")
        
    def set_cell_tensor(self):
        gradV = np.zeros((2,2))
        gradV[0] = self.gradVx
        gradV[1] = self.gradVy
        S = np.zeros((2,2))
        W = np.zeros((2,2))
        for i in range(2):
            for j in range(2):
                t1 = gradV[i,j]
                t2 = gradV[j,i]
                S[i,j] = t1 + t2 
                if i!=j :
                    W[i,j] = t1-t2
        S = 0.5*S 
        W = 0.5*W
        self.S = S 
        self.Omega = W 
        return

    def update_values(self)-> None:    
        k = self.k
        w = self.w
        self.epsilon = beta_star*w*k
        self.l = np.sqrt(k)/w 
        check = 0
        grad_k = self.grad_k
        grad_w = self.grad_w
        check  = np.dot(grad_k,grad_w).sum()
        if check <= 0 :
            self.sigma_d = 0
        else :
            self.sigma_d = sigma_do
        
        self.w_bar = max(self.w,self.w_bar_min())
        self.Nu_t = self.k / self.w_bar
        self.tau = 2*self.Nu_t*self.S - 2/3 * self.k * np.eye(2)


class Sim():
    """Classe pour la simulation"""
    def __init__(self, *args, **kwargs)->None:
        if (len(args) == 1 and isinstance(args[0], str)) or (len(kwargs) == 1 and 'filename' in kwargs and isinstance(kwargs['filename'], str)):
            filename = kwargs['filename'] if 'filename' in kwargs else args[0]
            t = time.time()
            self.mesh = Mesh(filename=filename)
            print(f"Temps de chargement du maillage : {time.time()-t} s")
        elif len(kwargs) == 1 and 'mesh' in kwargs and isinstance(kwargs['mesh'], Mesh):
            self.mesh = kwargs['mesh']
        else:
            raise ValueError("Invalid arguments: expected 'filename' or 'mesh' as input.")
        self.cell_param = [Parametres() for _ in range(len(self.mesh.cells))]

# Face
    def get_face_param(self,face_index:int)->Parametres: 
        """Calcul des parametres de la face"""

        cells = self.mesh.cells
        face = self.mesh.faces[face_index]
        index_owner = face.owner
        index_neighbour = face.neighbour
        if index_owner == -1 :
            return self.cell_param[index_neighbour]
        elif index_neighbour == -1 :
            return self.cell_param[index_owner]
        else :

            Vo = cells[face.owner].volume
            Vn = cells[face.neighbour].volume
            gn = Vo/(Vo + Vn)
            go = 1-gn
            T = gn*self.cell_param[index_neighbour].T + go*self.cell_param[index_owner].T
            vx = gn*self.cell_param[index_neighbour].vx + go*self.cell_param[index_owner].vx
            vy = gn*self.cell_param[index_neighbour].vy + go*self.cell_param[index_owner].vy
            p = gn*self.cell_param[index_neighbour].p + go*self.cell_param[index_owner].p
            return Parametres(T,p,vx,vy)

# Cell
    def get_grad_cell(self,cell_index, var)->np.array:
        """Calcul du gradient d'une variable dans la cellule"""

        grad = np.zeros(2,dtype=float)
        cell = self.mesh.cells[cell_index]
        cell_faces_index = cell.faces
        for i in range(len(cell_faces_index)):
            f = self.mesh.faces[cell_faces_index[i]]
            outward_face = (f.owner == cell_index)
            if outward_face :
                sign = 1
            else : 
                sign = -1
            grad += sign * f.surface * self.get_face_param(f.indice_global).get_var(var)
        grad /= cell.volume
            
        return grad

# Mesh
    def compute_gradient(self, *args)->np.array:
        """Calcul du gradient d'une variable dans la cellule"""
        if len(args)==1 :
            var = args[0]
            grad = np.zeros((len(self.mesh.cells),2))
            for i,f in enumerate(self.mesh.faces):
                flux_f = self.get_face_param(i).get_var(var)*f.surface
                if f.owner != -1 :
                    grad[f.owner] += flux_f
                if f.neighbour != -1 :
                    grad[f.neighbour] -= flux_f
            grad[:,0] /= self.mesh.cell_volume
            grad[:,1] /= self.mesh.cell_volume
            var_grad = "grad"+var
            for i in range(len(grad)):
                self.cell_param[i].set_var(var_grad,grad[i]) 
            return grad
        elif len(args)==0 :
            all_var = ['vx','vy','T','p','rho','k','w']
            return [self.compute_gradient(s) for s in all_var]
    
    def compute_tensors(self) : 
        for cell_index in range(self.mesh.size) :
            self.cell_param[cell_index].set_cell_tensor()
        return
    
    def update_all_param(self):
        self.compute_gradient()
        self.compute_tensors()

        for cell in self.cell_param:
            cell.update_values()
            

    
    def set_var(self,var:str,value:np.array,*args)->None:
        """Set the value of a variable in the mesh at a given point"""
        if len(args)==1 and isinstance(args[0],int):
            # On utilise l'indice de la cellule
            cell_index = args[0]
            self.cell_param[cell_index].set_var(var,value)
            return None
        
        # On utilise les coordonnees de la cellule
        elif len(args) == 1 and isinstance(args[0],np.ndarray) and args[0].shape == (2,):
            x = args[0][0]
            y = args[0][1]
        if len(args) == 1 and isinstance(args[0],list) and len(args[0]) == 2:
            x = args[0][0]
            y = args[0][1]
        elif len(args) == 2 and isinstance(args[0],float) and isinstance(args[1],float):
            x = args[0]
            y = args[1]
        cell_index = self.mesh.find_cell(x,y)
        if cell_index != None:
            self.cell_param[cell_index].set_var(var,value)
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
                try :
                    self.cell_param[cell.indice_global].set_var(var,args[0](x,y))
                except ValueError :
                    if var != "vy" and var != "Vx" :
                        self.cell_param[cell.indice_global].set_var(var,args[0](x,y)[0])
                    else :
                        self.cell_param[cell.indice_global].set_var(var,args[0](x,y)[1])
        return None

    # Plotting
    def plot(self, var:str,ax = None,point_size=10)->None:
        """Plot the mesh with the variable"""
        if ax is None:
            fig, ax = plt.subplots()
        
        x = [cell.centroid[0] for cell in self.mesh.cells]
        y = [cell.centroid[1] for cell in self.mesh.cells]
        
        values = [np.linalg.norm(self.cell_param[i].get_var(var)) for i in range(len(self.mesh.cells))]
        
        sc = ax.scatter(x, y, c=values, cmap='viridis', s=point_size)
        plt.colorbar(sc, ax=ax, label=var)

    def ldc(self,seed_number:int = 20, specific_seed = None, ax = None) :
        """ Trace les lignes de courant de la simulation"""
        # Clairement à optimiser
        if ax is None:
            fig, ax = plt.subplots()
        xmin,xmax,ymin,ymax = self.mesh.span()
        dx = np.sqrt((xmax-xmin)*(ymax-ymin)/self.mesh.size)/2
        if seed_number >0 :
            seed_y = np.linspace(0,ymax,seed_number+2)[1:-1]
            for s in range(seed_number):
                xs = [0]
                ys = [seed_y[s]]
                while xmin <= xs[-1] < xmax and ymin <= ys[-1] < ymax :
                    cell_index = self.mesh.find_cell(xs[-1],ys[-1])
                    if cell_index == None :
                        while xmin <= x < xmax and cell_index == None :
                            x = xs[-1] + dx
                            y = ys[-1]
                            cell_index = self.mesh.find_cell(x,y)
                    cell = self.mesh.cells[cell_index]
                    V = sim.cell_param[cell.indice_global].get_var("v")
                    V = V/V[0] 
                    xs.append(xs[-1] + V[0]*dx)
                    ys.append(ys[-1] + V[1]*dx)
                ax.plot(xs,ys,'k-')
        if specific_seed != None :
            xs = [specific_seed[1]]
            ys = [specific_seed[1]]
            while xmin <= xs[-1] < xmax and ymin <= ys[-1] < ymax :
                cell_index = self.mesh.find_cell(xs[-1],ys[-1])
                if cell_index == None :
                    while xmin <= x < xmax and cell_index == None :
                        x = xs[-1] + dx
                        y = ys[-1]
                        cell_index = self.mesh.find_cell(x,y)
                cell = self.mesh.cells[cell_index]
                V = sim.cell_param[cell.indice_global].get_var("v")
                V = V/V[0] 
                xs.append(xs[-1] + V[0]*dx)
                ys.append(ys[-1] + V[1]*dx)
            ax.plot(xs,ys,'k-')

def radial_CI(x,y,A=1)->float:
    """Fonction de condition initiale radiale"""
    r = np.sqrt((x-20)**2 + (y-10)**2)
    return A*np.exp(-(r-2)**2/10)
    
def lin_en_y(x,y,A=1):
    return A*y

def CI_qui_deforme(x,y,A=1) :
    return lin_en_y(x,y,A/5)  + radial_CI(x,y,A*10)

def CI_cylindre(x,y,A=1,a=2) :
    r = np.sqrt((x-20)**2 + (y-10)**2)
    teta = np.arctan2(y-10,x-20)
    e_r = np.array([np.cos(teta),np.sin(teta)])
    e_teta = np.array([-np.sin(teta),np.cos(teta)])
    V  = A*(np.cos(teta)*(1-(a/r)**2)*e_r - np.sin(teta)*(1+(a/r)**2)*e_teta)
    Vx = V[0]
    Vy = V[1]
    return Vx, Vy

if __name__ == "__main__":
    sim = Sim(filename = "D:/OneDrive/Documents/11-Codes/overflow/02_RANS/circle_mesh.dat")
    sim.set_CI("T",radial_CI)
    gradient = sim.compute_gradient("T")
    fig , ax = plt.subplots(2)
    # sim.mesh.plot_mesh(ax=ax)

    sim.set_CI("vx",CI_cylindre)
    sim.set_CI("vy",CI_cylindre)
    x = [cell.centroid[0] for cell in sim.mesh.cells]
    y = [cell.centroid[1] for cell in sim.mesh.cells]
    vx = [sim.cell_param[i].get_var("vx") for i in range(len(sim.mesh.cells))]
    vy = [sim.cell_param[i].get_var("vy") for i in range(len(sim.mesh.cells))]
    ax[0].quiver(x, y, vx, vy, scale=1, scale_units='xy', angles='xy', color='r')
    sim.compute_gradient("vx")
    sim.compute_gradient("vy")
    sim.compute_tensors()
    sim.plot("S",ax=ax[1])
    sim.ldc(0,specific_seed=[1,5],ax=ax[0])
    ax[0].set_aspect('equal', adjustable='box')
    ax[1].set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.show()

    