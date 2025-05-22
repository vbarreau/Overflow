import sys
sys.path.append(r"D:/OneDrive/Documents/11-Codes/overflow/02_RANS")
from maillage import *
from fonctions import *
import time

XI          = 0
f_beta      = 1
ALPHA       = 13 / 25
BETA_0      = 0.0708
BETA        = BETA_0
BETA_STAR   = 0.09
SIGMA       = 0.5
SIGMA_STAR  = 0.6
SIGMA_DO    = 1 / 8
CLIM       = 7 / 8

RHO = 1000
NU = 1e-3


class Parametres:
    """Classe pour les parametres de la cellule"""
    def __init__(self, T: float = 200, p: float = 1e5, vx: float = 0, vy: float = 0) -> None:
        self.T          = T
        self.p          = p
        self.vx         = vx
        self.vy         = vy
        self.B          = self.p + 0.5*RHO*(self.vx**2 + self.vy**2)  # Bernoulli constant
        self.gradT      = np.ones(2)
        self.gradP      = np.zeros(2)
        self.gradVx     = np.zeros(2)
        self.gradVy     = np.zeros(2)
        self.grad_rho   = np.zeros(2)
        self.S          = np.zeros((2, 2))  # Tenseur des deformations
        self.Omega      = np.zeros((2, 2))  # Tenseur de vorticité
        self.tau        = np.zeros((2, 2))  # tenseur de turbulences
        
        self.k          = 0.0          # Turbulent kinetic energy
        self.w          = 1.0          # Specific dissipation rate
        self.w_bar      = 1.0
        self.Nu_t       = 0.0           # Turbulent viscosity
        self.f_beta     = 0.0           # Beta function
        self.xi_omega   = 0.0           # Xi omega parameter
        # self.epsilon    = 0.0           # Dissipation rate
        # self.l          = 0.0           # Turbulence length scale
        self.sigma_d    = 0.0           # Scalar parameter
        self.grad_k     = np.zeros(2)   # Gradient of turbulent kinetic energy
        self.grad_w     = np.zeros(2)   # Gradient of specific dissipation rate
        self.gamma_k     = (NU + SIGMA_STAR*self.k/self.w)*self.grad_k
        self.gamma_w     = (NU + SIGMA * self.k/self.w)*self.grad_w  

        self.condition = [] 

    def reset_CL(self):
        """Reset the condition of the cell"""
        for cl in self.condition: 
            setattr(self, cl.var, cl.value)
        else :
            return

    
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
        elif var == "sigma_d":
            return self.sigma_d
        elif var == "gamma_k":
            return self.gamma_k
        elif var == "gamma_w":
            return self.gamma_w
        else:
            raise ValueError(f"Variable {var} non reconnue")
        
    def w_bar_min(self) -> float:
        """Renvoie la valeur minimale de w_bar"""
        s2 = (self.S*self.S).sum()
        return CLIM*np.sqrt(2*s2/BETA_STAR)

    def set_var(self, var: str, value) -> None:
        """Set the value of a variable in the cell"""
        if isinstance(value, (float, int, np.float64)):
            if var == "T":
                self.T = value
            elif var == "p":
                self.p = value
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
            elif var == 'gamma_k':
                self.gamma_k = value
            elif var == 'gamma_w':
                self.gamma_w = value
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
        # k,w et S doivent etre mis a jour avant   
        self.p = self.B - 0.5*RHO*(self.vx**2 + self.vy**2)
        self.T = self.p / (287.05 * RHO)
        k = self.k
        w = self.w
        self.epsilon = BETA_STAR*w*k
        self.l = np.sqrt(k)/w 
        check = 0
        grad_k = self.grad_k
        grad_w = self.grad_w
        check  = np.dot(grad_k,grad_w).sum()
        if check <= 0 :
            self.sigma_d = 0
        else :
            self.sigma_d = SIGMA_DO
        
        self.w_bar = max(self.w,self.w_bar_min())
        self.Nu_t = self.k / self.w_bar
        self.tau = 2*self.Nu_t*self.S - 2/3 * self.k * np.eye(2)
        self.gamma_k     = (NU + SIGMA_STAR*self.k/self.w)*self.grad_k         # Turbulent kinetic energy production
        self.gamma_w     = (NU + SIGMA * self.k/self.w)*self.grad_w           # Specific dissipation rate production

    def update_B(self)-> None:
        """Met a jour la constante de Bernoulli"""
        self.B = self.p + 0.5*RHO*(self.vx**2 + self.vy**2)
        return None

    def add_CL(self,condition)->None:
        """Ajoute une condition limite à la cellule"""
        self.condition.append(condition)
        return None
    

class CL():
    """Classe pour les conditions limites
        contient la valeur de la condition limite et la variable"""
    def __init__(self,var,value):
        self.var = var
        self.value = value

class Etat():
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
            all_var = ['vx','vy','T','p','k','w']
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
        for cell in self.cell_param:
            cell.update_values()
            cell.set_cell_tensor()
            cell.update_B()
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

    def plot_v(self,ax=None):
        """Plot the mesh with the velocity"""
        if ax is None:
            fig, ax = plt.subplots()
        
        x = [cell.centroid[0] for cell in self.mesh.cells]
        y = [cell.centroid[1] for cell in self.mesh.cells]
        u = [self.cell_param[i].vx for i in range(len(self.mesh.cells))]
        v = [self.cell_param[i].vy for i in range(len(self.mesh.cells))]
        
        ax.quiver(x, y, u, v, angles='xy', scale_units='xy', scale=1, color='blue')


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
                    V = self.cell_param[cell.indice_global].get_var("v")
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
                V = self.cell_param[cell.indice_global].get_var("v")
                V = V/V[0] 
                xs.append(xs[-1] + V[0]*dx)
                ys.append(ys[-1] + V[1]*dx)
            ax.plot(xs,ys,'k-')

    def __mul__(self,other):
        """Renvoie le produit d'une simulation par un objet VarEtat ou un scalaire"""
        if isinstance(other, float) or isinstance(other, int):
            new_sim = Etat(mesh = self.mesh)
            for var in self.cell_param[0].__dict__.keys():
                for i in range(len(self.mesh.cells)):
                    setattr(new_sim.cell_param[i], var, getattr(self.cell_param[i], var) * other)
            return new_sim
        elif isinstance(other, VarEtat) :
            new_sim = Etat(mesh = self.mesh)
            for var in self.cell_param[0].__dict__.keys():
                for i in range(len(self.mesh.cells)):
                    setattr(new_sim.cell_param[i], var, getattr(self.cell_param[i], var) * getattr(other.cell_param[i], var))
            return new_sim
        else:
            raise ValueError(f"Type {type(other)} non reconnu")
        
    def __truediv__(self,other):
        """Renvoie le produit d'une simulation par un scalaire"""
        if isinstance(other, float) or isinstance(other, int):
            new_sim = Etat(mesh = self.mesh)
            for var in self.cell_param[0].__dict__.keys():
                for i in range(len(self.mesh.cells)):
                    setattr(new_sim.cell_param[i], var, getattr(self.cell_param[i], var) / other)
            return new_sim
        else:
            raise ValueError(f"Type {type(other)} non reconnu")

    def __add__(self,other):
        """Renvoie la somme de deux simulations"""
        if isinstance(other, Etat):
            new_sim = Etat(mesh = self.mesh)
            for var in self.cell_param[0].__dict__.keys():
                for i in range(len(self.mesh.cells)):
                    setattr(new_sim.cell_param[i], var, getattr(self.cell_param[i], var) + getattr(other.cell_param[i], var))
            return new_sim
        else:
            raise ValueError(f"Type {type(other)} non reconnu")
           
    def copy(self):
        """Renvoie une copie de la simulation"""
        new_sim = Etat(mesh = self.mesh)
        for var in self.cell_param[0].__dict__.keys():
            for i in range(len(self.mesh.cells)):
                setattr(new_sim.cell_param[i], var, getattr(self.cell_param[i], var))
        return new_sim

class VarEtat(Etat):
    """Classe contenant la variation temporelle d'un etat à l'autre"""
    def __init__(self, *args, **kwargs)->None:
        super().__init__(*args, **kwargs)
        for var in self.cell_param[0].__dict__.keys():
            if type(getattr(self.cell_param[0], var)) == float or type(getattr(self.cell_param[0], var)) == int:
                for cell in self.cell_param:
                    setattr(cell, var, 0)
            elif type(getattr(self.cell_param[0], var)) == np.ndarray:
                s = getattr(self.cell_param[0], var).shape
                for cell in self.cell_param:
                    setattr(cell, var, np.zeros(s))

    def __sub__(self,other):
        """Renvoie la somme de deux simulations"""
        if isinstance(other, Etat):
            new_sim = Etat(mesh = self.mesh)
            for var in self.cell_param[0].__dict__.keys():
                if var == "condition":
                    continue
                for i in range(len(self.mesh.cells)):
                    setattr(new_sim.cell_param[i], var, getattr(self.cell_param[i], var) - getattr(other.cell_param[i], var))
            return new_sim
        else:
            raise ValueError(f"Type {type(other)} non reconnu")
        
    
    def sum(self,var:str)->float:
        """Renvoie la somme d'une variable dans la simulation"""
        s = 0
        for cell in self.cell_param:
            s += getattr(cell, var)
        return s
    
    def mean(self,var:str)->float:
        """Renvoie la moyenne d'une variable dans la simulation"""
        s = 0
        for cell in self.cell_param:
            s += getattr(cell, var)
        return s/len(self.cell_param)
          

# def NS(cp : Parametres, t:float,prod_k, prod_w)->Parametres:
#     """Renvoie les variations temporelles de la cellule"""
#     if len(cp.condition) > 0 :
#         return 0,0,0,0
#     dVx = -cp.vx*cp.gradVx[0]- cp.vy*cp.gradVx[1] *(-cp.gradP + 0 )/RHO  # Manque les gradients de sigma et tau
#     dVy = -cp.vx*cp.gradVy[0]- cp.vy*cp.gradVy[1] *(-cp.gradP + 0 )/RHO  # Manque les gradients de sigma et tau
#     A =0
#     for i in range(2):
#         for j in range(2):
#             if i == 0 :
#                 gv = cp.gradVx
#             else :
#                 gv = cp.gradVy
#             A += cp.tau[i,j]*gv[j] 
#     dk = -cp.vx*cp.grad_k[0]- cp.vy*cp.grad_k[1] + A - BETA_STAR*cp.k*cp.w + prod_k.sum() 
#     dw = -cp.vx*cp.grad_w[0]- cp.vy*cp.grad_w[1] + ALPHA * cp.w/cp.k * A - BETA*cp.w**2 + cp.sigma_d*(cp.grad_k*cp.grad_w).sum()  + prod_w.sum()
    
#     return dVx, dVy, dk, dw


class Sim():
    def __init__(self, filename:str)->None:
        self.etat = Etat(filename=filename)
        self.etat.set_CI("T",radial_CI)
        self.etat.update_all_param()
        self.etat.compute_gradient()
        self.etat.compute_tensors()
        return None
    
    def set_CL(self, var:str, value:float, face_index:str = "in")->None:
        xmin,xmax,_,_ = self.etat.mesh.span()
        check_step = self.etat.mesh.mean_step*4/5
        condition = CL(var,value)

        if face_index == "in":
            def is_good(cell):
                return cell.centroid[0] < xmin + check_step
        elif face_index == "out":
            def is_good(cell):
                return cell.centroid[0] > xmax - check_step
            
        for i,cell in enumerate(self.etat.mesh.cells):
            if not cell.is_boundary:
                continue
            if is_good(cell):
                self.etat.cell_param[i].add_CL(condition)
                setattr(self.etat.cell_param[i], var, value)
    
    # def NS_sim(self,Q:Etat,t:float)->Etat:
    #     """Résolution des equations de Navier-Stokes sur l'etat Q
    #     Renvoie les vecteurs de variations temporelles de Vx, Vy, k, w et leurs gradients
    #     Les autres facteurs de variation sont unitaires"""
    #     dQ_dt = VarEtat(mesh=self.etat.mesh)
    #     prod_k = Q.compute_gradient("gamma_k")
    #     prod_w = Q.compute_gradient("gamma_w")
    #     residu = np.zeros(4)

    #     for i in tqdm(range(len(self.etat.mesh.cells), desc="Calcul des variations")):
    #         if self.etat.mesh.cells[i].is_boundary :
    #             continue # 
    #         c = self.etat.cell_param[i]

    #         dVx, dVy, dk, dw = NS(c,t,prod_k[i],prod_w[i])

    #         dQ_dt.cell_param[i].vx = dVx
    #         dQ_dt.cell_param[i].vy = dVy
    #         dQ_dt.cell_param[i].k = dk
    #         dQ_dt.cell_param[i].w = dw 

    #     residu[0] = Q.sum("vx")/Q.mean("vx")
    #     residu[1] = Q.sum("vy")/Q.mean("vy")
    #     residu[2] = Q.sum("k")/Q.mean("k")
    #     residu[3] = Q.sum("w")/Q.mean("w")

    #     dQ_dt.compute_gradient("vx") # Les dérivées commutent donc je calcul d²/dx.dt plutot que d²/dt.dx
    #     dQ_dt.compute_gradient("vy")
    #     dQ_dt.compute_gradient("k")
    #     dQ_dt.compute_gradient("w")

    #     return dQ_dt, residu


    # def step(self, F:callable, dt:float, t:float, ordre:int)-> Etat:
    #     """ Effectue une étape de la méthode de Runge-Kutta dans la simulation
    #         F       : fonction de Q et t qui renvoie le VarEtat dQ/dt = F(Q,t)[0]
    #         dt      : pas de temps désiré
    #         t       : temps actuel
    #         ordre   : 1,2,3 ou 4 
    #         """
    #     if ordre == 1:
    #         k1, residu =  F(self.etat, t)
    #         self.etat =  self.etat + dt * k1
    #     elif ordre == 2:
    #         k1 = F(self.etat, t)[0]
    #         k2, residu = F(self.etat + dt/2 * k1, t + dt/2)[0]
    #         self.etat = self.etat + dt * k2
    #     elif ordre == 3:
    #         k1 = F(self.etat, t)[0]
    #         k2 = F(self.etat + dt/2 * k1, t + dt/2)[0]
    #         k3, residu = F(self.etat - dt * k1 + 2 * dt * k2, t + dt)[0]
    #         self.etat = self.etat + dt/6 * (k1 + 4 * k2 + k3)
    #     elif ordre == 4:
    #         k1 = F(self.etat, t)[0]
    #         k2 = F(self.etat + dt/2 * k1, t + dt/2)[0]
    #         k3 = F(self.etat + dt/2 * k2, t + dt/2)[0]
    #         k4, residu = F(self.etat + dt * k3, t + dt)[0]
    #         self.etat = self.etat + dt/6 * (k1 + 2 * k2 + 2 * k3 + k4)
    #     return residu
        
    # def integrator(self,F:callable,dt:float,ti=0,tf=1,ordre=1):
    #     """ q       : Etat initiale Q[ti]
    #         F       : fonction de Q et t telle que dQ/dt = F(Q,t)[0]
    #         dt      : pas de temps désiré
    #         ti      : temps initial d'intégration
    #         tf      : temps de fin d'intégration
    #         ordre   : 1,2,3 ou 4 
    #         """
    #     t = ti 
    #     i = 0
    #     residu = np.ones(4)
    #     while residu.max() > 1e-3 and t < tf :
    #         t+=dt 
    #         i+=1
    #         residu = self.step(F,dt,t,ordre)

if __name__ == "__main__":
    sim = Sim(filename = "D:/OneDrive/Documents/11-Codes/overflow/02_RANS/circle_mesh.dat")
    sim.set_CL("vx",10,"in")
    sim.set_CL("vy",0,"in")
    sim.set_CL("p",1e5,"out")
    # for i in range(sim.etat.mesh.size):
    #     if sim.etat.cell_param[i].condition != None :
    #         cell_p = sim.etat.cell_param[i]
    #         cell = sim.etat.mesh.cells[i]
    #         var = cell_p.condition.var
    #         print(f'Cellule {i}, x = {cell.centroid[0]} : {var} = {getattr(sim.etat.cell_param[i],var)}')
    sim.etat.plot_v()
    plt.show()