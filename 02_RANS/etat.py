from maillage import *

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
        self.gradp      = np.zeros(2)
        self.gradvx     = np.zeros(2)
        self.gradvy     = np.zeros(2)
        self.S          = np.zeros((2, 2))  # Tenseur des deformations
        self.Omega      = np.zeros((2, 2))  # Tenseur de vorticité
        self.tau        = np.zeros((2, 2))  # tenseur de turbulences
        
        self.k          = 1.0          # Turbulent kinetic energy
        self.w          = 1.0          # Specific dissipation rate
        self.w_bar      = 1.0
        self.Nu_t       = 0.0           # Turbulent viscosity
        self.f_beta     = 0.0           # Beta function
        self.xi_omega   = 0.0           # Xi omega parameter
        # self.epsilon    = 0.0           # Dissipation rate
        # self.l          = 0.0           # Turbulence length scale
        self.sigma_d    = 0.0           # Scalar parameter
        self.gradk     = np.zeros(2)   # Gradient of turbulent kinetic energy
        self.gradw     = np.zeros(2)   # Gradient of specific dissipation rate
        self.gamma_k     = (NU + SIGMA_STAR*self.k/self.w)*self.gradk
        self.gamma_w     = (NU + SIGMA * self.k/self.w)*self.gradw  

        self.condition = [] 

    def reset_CL(self):
        """Reset the condition of the cell"""
        for cl in self.condition: 
            setattr(self, cl.var, cl.value)
        else :
            return

    
    def w_bar_min(self) -> float:
        """Renvoie la valeur minimale de w_bar"""
        s2 = (self.S*self.S).sum()
        return CLIM*np.sqrt(2*s2/BETA_STAR)

    def set_cell_tensor(self):
        gradV = np.zeros((2,2))
        gradV[0] = self.gradvx
        gradV[1] = self.gradvy
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
        # self.epsilon = BETA_STAR*w*k
        # self.l = np.sqrt(k)/w 
        check = 0
        gradk = self.gradk
        gradw = self.gradw
        check  = np.dot(gradk,gradw).sum()
        if check <= 0 :
            self.sigma_d = 0
        else :
            self.sigma_d = SIGMA_DO
        
        self.w_bar = max(self.w,self.w_bar_min())
        self.Nu_t = self.k / self.w_bar
        self.tau = 2*self.Nu_t*self.S - 2/3 * self.k * np.eye(2)
        self.gamma_k     = (NU + SIGMA_STAR*self.k/self.w)*self.gradk         # Turbulent kinetic energy production
        self.gamma_w     = (NU + SIGMA * self.k/self.w)*self.gradw           # Specific dissipation rate production

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
            val = getattr(self.get_face_param(f.indice_global),var)
            grad += sign * f.surface * val
        grad /= cell.volume
            
        return grad

# Mesh
    def compute_gradient(self, *args)->np.array:
        """Calcul du gradient d'une variable dans la cellule"""
        if len(args)==1 :
            var = args[0]
            grad = np.zeros((len(self.mesh.cells),2))
            for i,f in enumerate(self.mesh.faces):
                val = getattr(self.get_face_param(i),var)
                flux_f = val*f.surface
                if f.owner != -1 :
                    grad[f.owner] += flux_f
                if f.neighbour != -1 :
                    grad[f.neighbour] -= flux_f
            grad[:,0] /= self.mesh.cell_volume
            grad[:,1] /= self.mesh.cell_volume
            var_grad = "grad"+var
            if var_grad in self.cell_param[0].__dict__.keys():    
                for i in range(len(grad)):
                    setattr(self.cell_param[i],var_grad,grad[i])
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
            setattr(self.cell_param[cell_index], var, value)
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
            setattr(self.cell_param[cell_index], var, value)
        return None
    
    def set_CI(self,var,*args)-> None:
        """ Set the initial condition of the simulation from either a matrix or a function"""
        if len(args) == 1 and isinstance(args[0],np.ndarray):
            xmin,xmax,ymin,ymax = self.mesh.span()
            DX = (xmax-xmin)/args[0].shape[0]
            DY = (ymax-ymin)/args[0].shape[1]
            for cell in self.mesh.cells:
                limits = self.cell_param[cell.indice_global].condition
                if  var in [cl.var for cl in limits] :
                    # Si la cellule a une condition limite, on ne change pas la valeur
                    continue
                x = cell.centroid[0]
                y = cell.centroid[1]
                nx = int((x-xmin)/DX)
                ny = int((y-ymin)/DY)
                if nx < 0 or nx >= args[0].shape[0] or ny < 0 or ny >= args[0].shape[1]:
                    raise ValueError(f"Cellule {cell.indice_global} en dehors de la matrice")
                setattr(self.cell_param[cell.indice_global], var, args[0][nx,ny])
        elif len(args) == 1 and callable(args[0]):
            for cell in self.mesh.cells:
                limits = self.cell_param[cell.indice_global].condition
                if  var in [cl.var for cl in limits] :
                    # Si la cellule a une condition limite, on ne change pas la valeur
                    continue
                x = cell.centroid[0]
                y = cell.centroid[1]
                setattr(self.cell_param[cell.indice_global], var, args[0](x,y))
                
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
        
        values = [np.linalg.norm(getattr(self.cell_param[i], var)) for i in range(len(self.mesh.cells))]
        
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
                if var == "condition":
                    continue
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

    def sum(self,var:str)->float:
        """Renvoie la somme d'une variable dans l'état"""
        s = 0
        for cell in self.cell_param:
            s += getattr(cell, var)
        return s
    
    def mean(self,var:str)->float:
        """Renvoie la moyenne non pondérée d'une variable dans l'état"""
        s = 0
        for cell in self.cell_param:
            s += getattr(cell, var)
        return s/len(self.cell_param)

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
        
          