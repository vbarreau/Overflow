from geom import *
from Param import *
from model import psi_OSPRE


RELAX = 0.1


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
            self.mesh = Mesh(filename=filename)
        elif len(kwargs) == 1 and 'mesh' in kwargs and isinstance(kwargs['mesh'], Mesh):
            self.mesh = kwargs['mesh']
        else:
            raise ValueError("Invalid arguments: expected 'filename' or 'mesh' as input.")
        self.cell_param = [Parametres() for _ in range(len(self.mesh.cells))]
        self.face_param = [Parametres() for _ in range(len(self.mesh.faces))]

# Face

    def face_average(self,gn,go,owner,neighbour,var):
        """Calcul la moyenne de la variable var entre les deux cellules owner et neighbour"""
        if owner is None:
            return gn*getattr(self.cell_param[neighbour],var)
        elif neighbour is None:
            return go*getattr(self.cell_param[owner],var)
        else:
            # On suppose que les deux cellules sont valides
            # On utilise la moyenne pondérée par le volume des cellules
            # gn = Vn/(Vo+Vn) et go = Vo/(Vo+Vn)
            # On utilise getattr pour acceder à la variable de la cellule

            return gn*getattr(self.cell_param[neighbour],var) + go*getattr(self.cell_param[owner],var)

    def get_face_param(self,face_index:int)->Parametres: 
        """Calcul des parametres de la face"""

        cells = self.mesh.cells
        face = self.mesh.faces[face_index]
        index_owner = face.owner
        index_neighbour = face.neighbour
        if index_owner is None :
            return self.cell_param[index_neighbour]
        elif index_neighbour is None :
            return self.cell_param[index_owner]
        else :
            Vo = cells[face.owner].volume
            Vn = cells[face.neighbour].volume
            gn = Vo/(Vo + Vn)
            go = 1-gn
            T  = self.face_average(gn,go,index_owner,index_neighbour,'T')
            k  = self.face_average(gn,go,index_owner,index_neighbour,'k')
            w  = self.face_average(gn,go,index_owner,index_neighbour,'w')
            vx = self.face_average(gn,go,index_owner,index_neighbour,'vx')
            vy = self.face_average(gn,go,index_owner,index_neighbour,'vy')
            p  = self.face_average(gn,go,index_owner,index_neighbour,'p')
            return Parametres(T,p,vx,vy,k,w)
        
    def compute_face_param_HR(self,face_index, var:str = 'T',psi = psi_OSPRE)->float:
        """Calcul le paramètre Haute Résolution de la face"""
        face = self.mesh.faces[face_index]
        UU,U,C,D,DD = face.stream(self.mesh)
        phi_U  = getattr(self.cell_param[U.indice_global] , var)
        phi_D  = getattr(self.cell_param[D.indice_global] , var)
        phi_UU = getattr(self.cell_param[UU.indice_global], var)
        if phi_D == phi_U:
            phi_f = phi_U
        else :
            r = (phi_U-phi_UU)/(phi_D-phi_U)
            phi_f = phi_U + 0.5*psi(r)*(phi_D-phi_U)
        setattr(self.face_param[face_index],var, phi_f)
        return phi_f
    
    def compute_face_param_U(self,face_index:int, var:str = 'T')->float:
        """Calcul le paramètre de la face en utilisant la moyenne de l'interface"""
        cells = self.mesh.cells
        face = self.mesh.faces[face_index]
        index_owner = face.owner
        index_neighbour = face.neighbour
        if index_owner is None :
            return getattr(self.cell_param[index_neighbour],var)
        elif index_neighbour is None :
            return getattr(self.cell_param[index_owner],var)
        else :
            Vo = cells[face.owner].volume
            Vn = cells[face.neighbour].volume
            gn = Vo/(Vo + Vn)
            go = 1-gn
            phi = self.face_average(gn,go,index_neighbour,index_owner,var)
            setattr(self.face_param[face_index],var ,phi)
            return phi

    def face_flow(self,face_index:int)->float:
        """Calcul le debit de fluide traversant la face
        Le débit est positif si le fluide traverse la face dans le sens de son vecteur surface"""
        # On suppose que l'array face_param est à jour
        S = self.mesh.faces[face_index].surface
        face_param = self.get_face_param(face_index)
        V = np.array([face_param.vx, face_param.vy])
        return np.dot(S, V)*RHO # On suppose que la vitesse est en m/s et la surface en m², donc le debit est en kg/s
    
    def FluxVf(self,face_index , T_f: np.array)->np.array:
        """Renvoie le flux de vitesse à la face
        voir p.590 du livre Fluid Mechanics and its Applications"""
        param_f = self.face_param[face_index]
        vHR_f = np.array([param_f.vx, param_f.vy])
        vU_f = np.array( [self.compute_face_param_U(face_index,'vx'),
                            self.compute_face_param_U(face_index,'vy')] )
        gradV_f = np.array([param_f.gradxvx, param_f.gradxvy])
        return -MU*np.dot(gradV_f,T_f) + self.face_flow(face_index)*(vHR_f - vU_f) 
    
    def diff_term_f(self,face_index:int,E_f:np.array)->np.array:
        """Calcul le terme de diffusion à la face"""
        E_f = self.mesh.faces[face_index].Ef_Tf(self.mesh.cells)[0]
        d_CF = self.mesh.faces[face_index].distance_des_centres(self.mesh.cells)
        return MU * E_f / d_CF
    

    
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
                if f.owner is not None :
                    grad[f.owner] += flux_f
                if f.neighbour is not None :
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
        
    def compute_sink(self):
        """ Calcul du terme puit des equations de vitesse
            d/dxj (Sij + tau_ij))"""
        grad = np.zeros((len(self.mesh.cells),2,2,2))
        for i,f in enumerate(self.mesh.faces):
            #val est la somme sigma + tau
            val = getattr(self.get_face_param(i),'tau') + getattr(self.get_face_param(i),'S')
            flux_f = np.zeros((2,2,2))
            flux_f[0] = val*f.surface[0]
            flux_f[1] = val*f.surface[1]
            if f.owner is not None :
                grad[f.owner] += flux_f
            if f.neighbour is not None :
                grad[f.neighbour] -= flux_f
        for i in range(len(grad)):
            # On divise par le volume de la cellule
            grad[i,0,0] /= self.mesh.cell_volume[i]
            grad[i,0,1] /= self.mesh.cell_volume[i]
            grad[i,1,0] /= self.mesh.cell_volume[i]
            grad[i,1,1] /= self.mesh.cell_volume[i]
        for i in range(len(grad)):
            self.cell_param[i].v_sink = grad[i]
    
    def compute_tensors(self) : 
        for cell_index in range(self.mesh.size) :
            self.cell_param[cell_index].set_cell_tensor()
        return
    
    def update_all_param(self):
        self.compute_gradient()
        self.compute_tensors()
        self.compute_sink()

        for cell in self.cell_param:
            cell.update_values()
            cell.reset_CL()
            
    def get_var(self,var:str)-> np.array:
        """Renvoie la valeur de la variable var dans l'état"""
        if var in self.cell_param[0].__dict__.keys():
            return np.array([getattr(cell, var) for cell in self.cell_param])
        else:
            raise ValueError(f"Variable {var} not found in cell parameters")

    def set_var(self,var:str,value:np.array,*args)->None:
        """Set the value of a variable in the whole mesh or at a given point if specified"""
        # Maillage entier
        if len(args)==0 and isinstance(value, np.ndarray) and value.shape == (len(self.mesh.cells),):
            # On utilise la valeur pour chaque cellule
            for i in range(len(self.mesh.cells)):
                setattr(self.cell_param[i], var, value[i])
            return None

        # Cellule unique
        elif len(args)==1 and isinstance(args[0],int):
            # On utilise l'indice de la cellule
            cell_index = args[0]
            setattr(self.cell_param[cell_index], var, value)
            return None
        
        else :
        # On utilise les coordonnees de la cellule
            if len(args) == 1 and isinstance(args[0],np.ndarray) and args[0].shape == (2,):
                x = args[0][0]
                y = args[0][1]
            elif len(args) == 1 and isinstance(args[0],list) and len(args[0]) == 2:
                x = args[0][0]
                y = args[0][1]
            elif len(args) == 2 and isinstance(args[0],float) and isinstance(args[1],float):
                x = args[0]
                y = args[1]
            cell_index = self.mesh.find_cell_index(x,y)

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
    def plot(self, var:str,ax = None,point_size=10,cbar=True)->None:
        """Plot the mesh with the variable"""
        if ax is None:
            fig, ax = plt.subplots()
        
        x = [cell.centroid[0] for cell in self.mesh.cells]
        y = [cell.centroid[1] for cell in self.mesh.cells]
        
        values = [np.linalg.norm(getattr(self.cell_param[i], var)) for i in range(len(self.mesh.cells))]
        
        sc = ax.scatter(x, y, c=values, cmap='viridis', s=point_size)
        if cbar :
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
                    cell_index = self.mesh.find_cell_index(xs[-1],ys[-1])
                    if cell_index == None :
                        while xmin <= x < xmax and cell_index == None :
                            x = xs[-1] + dx
                            y = ys[-1]
                            cell_index = self.mesh.find_cell_index(x,y)
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
                cell_index = self.mesh.find_cell_index(xs[-1],ys[-1])
                if cell_index == None :
                    while xmin <= x < xmax and cell_index == None :
                        x = xs[-1] + dx
                        y = ys[-1]
                        cell_index = self.mesh.find_cell_index(x,y)
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
                if var == "condition":
                    continue
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
    def __init__(self, etat_ini:Etat)->None:
        """Initialize VarEtat class by copying the structure of initial State"""
        self.mesh = etat_ini.mesh.copy()
        self.cell_param = etat_ini.cell_param.copy()
        self.face_param = etat_ini.face_param.copy()
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
        
          