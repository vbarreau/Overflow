from numpy import *
from matplotlib.pyplot import *
from scipy.interpolate import interp1d,UnivariateSpline
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

def step(q, F, dt:float, t:float, ordre:int):
    """ q       : valeur initiale Q[ti]
        F       : fonction de Q et t telle que dQ/dt = F(Q,t)
        dt      : pas de temps désiré
        t       : temps actuel
        ordre   : 1,2,3,4 ou "trapeze"
        """
    if ordre == 1:
        return q + dt * F(q, t)
    elif ordre == 2:
        k1 = F(q, t)
        k2 = F(q + dt/2 * k1, t + dt/2)
        return q + dt * k2
    elif ordre == 3:
        k1 = F(q, t)
        k2 = F(q + dt/2 * k1, t + dt/2)
        k3 = F(q - dt * k1 + 2 * dt * k2, t + dt)
        return q + dt/6 * (k1 + 4 * k2 + k3)
    elif ordre == 4:
        k1 = F(q, t)
        k2 = F(q + dt/2 * k1, t + dt/2)
        k3 = F(q + dt/2 * k2, t + dt/2)
        k4 = F(q + dt * k3, t + dt)
        return q + dt/6 * (k1 + 2 * k2 + 2 * k3 + k4)

def integrator(q,F,dt,ti=0,tf=1,ordre=1):
    """ q       : valeur initiale Q[ti]
        F       : fonction de Q et t telle que dQ/dt = F(Q,t)
        dt      : pas de temps désiré
        ti      : temps initial d'intégration
        tf      : temps de fin d'intégration
        ordre   : 1,2,3,4 ou "trapeze"
        """
    Q = [q]
    t = ti 
    T = [t]
    i = 0
    while t < tf :
        t+=dt 
        T.append(t)
        i+=1
        Q_new = step(Q[-1],F,dt,t,ordre)
        Q.append(Q_new)

    return array(Q)

def fonction_ex(q,t):
    A = zeros((2,2))
    A[0,1] = 1
    A[1,0] = -1 
    return dot(A,q)

def test_integrator():
    ti = 0
    tf = 10
    dt = 1e-3
    q0 = array([0,1])
    fig,ax = subplots(2)

    Q1 = integrator(q0 , fonction_ex,dt,ti=ti,tf=tf,ordre=1)[:,0]
    Q2 = integrator( q0, fonction_ex,dt,ti=ti,tf=tf,ordre=2)[:,0]
    Q3 = integrator( q0, fonction_ex,dt,ti=ti,tf=tf,ordre=3)[:,0]
    Q4 = integrator( q0, fonction_ex,dt,ti=ti,tf=tf,ordre=4)[:,0]
    T = linspace(ti,tf,len(Q1))
    Q_sol = array([sin(t) for t in T])


    ax[0].plot(T, Q_sol , label="solution",color='k')
    ax[0].plot(T,Q1,label="RK1",color='orange')
    ax[0].plot(T,Q2,label="RK2",color='g')
    ax[0].plot(T,Q3,label="RK3",color='blueviolet')
    ax[0].plot(T,Q4,label="RK4",color='r')
    ax[0].legend()
    ax[0].set_title("Intégrations de l'oscilateur harmonique")


    ax[1].plot(T,abs(Q_sol-Q1),label="RK1",color='orange')
    ax[1].plot(T,abs(Q_sol-Q2),label="RK2",color='g')
    ax[1].plot(T,abs(Q_sol-Q3),label="RK3",color='blueviolet')
    ax[1].plot(T,abs(Q_sol-Q4),label="RK4",color='r')
    ax[1].set_yscale('log')
    ax[1].set_title("Erreurs des intégrations")
    fig.set_tight_layout(True)
    
    # assert all(abs(Q_sol-Q1)<0.1)
    # assert all(abs(Q_sol-Q2)<0.1)
    # assert all(abs(Q_sol-Q3)<0.1)
    # assert all(abs(Q_sol-Q4)<0.1)

def anim(matSol: ndarray, nx: int, T:ndarray = None, fps: int = 10,name="animation.gif"):
    """Crée une animation de la solution donnée en format GIF."""
    x = linspace(0, 1, nx)
    fig = figure()  # initialise la figure
    line, = plot([], [])
    xlim(0, 1)
    ylim(-1, max(matSol[0] * 1.1))
    grid()
    legend(loc="upper right")

    def animate(i):
        y = matSol[i]
        line.set_data(x, y)
        title(f"t = {T[i]:.2f}")
        return line,

    # Create a tqdm progress bar
    with tqdm(total=len(matSol), desc="Creating animation") as pbar:
        def update(frame):
            animate(frame)
            pbar.update(1)
            return line,

        ani = FuncAnimation(fig, update, frames=len(matSol), interval=1000 / fps, blit=True, repeat=False)
        ani.save(name, fps=fps)

def gaussian(x, mean, std):
    return exp(-(x-mean)**2/(2*std**2))/sqrt(2*pi*std**2)

def initial_conditions(nx):
    """définit les conditions initiales de la simulation : une gaussienne de c"""
    x = linspace(0, 1, nx)
    return gaussian(x,0.5,0.05)

def initial_conditions_2D(nx):
    """définit les conditions initiales de la simulation : une gaussienne de c"""
    out = zeros((nx,nx))
    for i in range(nx) :
        for j in range(nx) :
            out[i,j] = gaussian(i/nx,0.5,0.1)*gaussian(j/nx,0.5,0.1)
    return out

def extend_per(X) :
    """renvoie 'larray X avec des conditions périodiques aux limites de chaque dimension
    quelque soit la dimension de X"""

    S = X.shape 
    S_per = tuple(array(S,int) + 2*ones(len(S),int))
    X_per = zeros(S_per)
    X_per[tuple([slice(1, i+1) for i in S])] = X
    for i in range(len(S)):
        X_per = swapaxes(X_per, 0, i)
        X_per[0] = X_per[-2]
        X_per[-1] = X_per[1]
        X_per = swapaxes(X_per, 0, i)
    return X_per

def extend_closed(X) :
    """renvoie 'larray X avec des conditions fermées aux limites de chaque dimension
    quelque soit la dimension de X"""
    S = X.shape 
    S_ext = tuple(array(S,int) + 2*ones(len(S),int))
    X_closed = zeros(S_ext)
    X_closed[tuple([slice(1, i+1) for i in S])] = X
    for i in range(len(S)):
        X_closed = swapaxes(X_closed, 0, i)
        X_closed[0] = X_closed[1]
        X_closed[-1] = X_closed[-2]
        X_closed = swapaxes(X_closed, 0, i)
    return X_closed

def grad(X:ndarray,flag = "") :
    """renvoie le gradient de X,
    soit autant d'array que X a de dimensions, contenant les dérivées partielles de X selon cette direction"""
    
    periodic = 'periodic' in flag
    closed = 'closed' in flag

    if periodic and closed :
        raise ValueError("Les conditions ne peuvent être à la fois périodiques et fermées")
    
    if periodic :
        X_ext = extend_per(X)
    elif closed :
        X_ext = extend_closed(X)

    if periodic or closed :        
        dX = gradient(X_ext) # renvoi liste de gradients dans chaque direction

        # On enlève ensuite les bords, que l'on avait pour la condition aux limites
        S = X.shape
        if type(dX) == tuple or type(dX)==list:
            for i in range(len(dX)) :
                dX[i] = dX[i][tuple([slice(1, i+1) for i in S])]
        else :
            dX = dX[tuple([slice(1, i+1) for i in S])]

    else :
        dX = gradient(X)

    return array(dX)

def div(X_vec:ndarray,flag = "") :
    """renvoie la divergence de X"""
    S = X_vec.shape
    div = zeros(S[1:])
    for i in range(S[0]):
        div += grad(X_vec[i],flag=flag)[i]
    return div

def array_to_func(X:ndarray, t_end:float, dt:float):
    """renvoie une fonction interpolant les valeurs de X"""
    T = linspace(0-dt,t_end+dt,len(X))
    return interp1d(T,X)

def V_dot(V,q):
    assert V.shape[:-1] == q.shape
    S=V.shape
    if len(S)>1:
        out = zeros([S[-1],S[0],S[1]])
        for i in range(S[-1]):
            out[S[-1]-i-1,:,:] = V[:,:,i]*q
        return out 
    else : 
        return V * q

class Cellule:
    def __init__(self, x, y=None, z=None, dx=0.1, dy=0.1,dz=0.1,V=None):
        """
        Initialise une cellule dans une grille discrète.
        
        :param x: Position x de la cellule (centre).
        :param y: Position y de la cellule (centre).
        :param dx: Taille de la cellule en x.
        :param dy: Taille de la cellule en y.
        """
        self.x = x
        self.dx = dx

        if y is None :
            self.dim = 1
            self.volume = dx
        elif z is None :
            self.dim = 2
            self.y = y
            self.dy = dy
            self.volume = dx*dy
        else :
            self.y = y
            self.dy = dy
            self.z = z 
            self.dz = dz 
            self.dim = 3 
            self.volume = dx*dy*dz
        
        # Propriétés physiques de la cellule
        self.velocity = V
            
        self.P = 1e5
        self.rho = 1.0
        self.T = 300.0  # En Kelvin
        self.pol = 0 

        # Constantes physiques (par ex., pour l'enthalpie)
        self.capacité_thermique = 1005.0  # Cp en J/(kg·K) pour l'air sec
        
    def h(self):
        """
        Calcule l'enthalpie massique de la cellule (h = Cp * T + P/rho).
        """
        return self.capacité_thermique*self.T + self.P/self.rho

    def update(self, vitesse=None, pression=None, densité=None, température=None):
        """
        Met à jour les propriétés physiques de la cellule.
        """
        if vitesse is not None:
            self.vitesse = array(vitesse)
        if pression is not None:
            self.pression = pression
        if densité is not None:
            self.densité = densité
        if température is not None:
            self.température = température
    
    def F(self):
        """
        Calcule le flux d'une cellule.
        autre_cellule: Instance de Cellule voisine.
        n            : vecteur normal à la surface considérée

        return       : Flux entre les deux cellules.
        """
        flux = [0,0,0]
        flux[0] = self.rho * self.velocity.transpose()
        flux[1] = self.rho * dot(self.velocity,self.velocity.transpose()) + self.P * eye(self.dim) 
        flux[2] = flux[0] * self.h() 
        return flux

class Grid:
    def __init__(self, nx, ny=None, nz=None, dx=0.1, dy=0.1, dz=0.1):
        """
        Initialise une grille de cellules. 1D par défaut. 2D ou 3D si les paramètres associés sont indiqués.
        
        :param nx: Nombre de cellules en x.
        :param ny: Nombre de cellules en y.
        :param dx: Taille de chaque cellule en x.
        :param dy: Taille de chaque cellule en y.
        """
        self.nx = nx
        self.dx = dx
        self.Lx = nx * dx
        self.dim = 1
        self.shape = (nx,)
        X = linspace(0, self.Lx, nx)
        self.X = X

        if ny is not None:
            self.dim = 2
            self.ny = ny
            self.dy = dy
            self.Ly = ny * dy
            self.shape = (nx, ny)
            Y = linspace(0, self.Ly, ny)
            self.X, self.Y = meshgrid(X, Y)

        if nz is not None:
            self.dim = 3
            self.nz = nz
            self.dz = dz
            self.Lz = nz * dz
            self.shape = (nx, ny, nz)
            Z = linspace(0, self.Lz, nz)
            self.X, self.Y, self.Z = meshgrid(X, Y, Z)

        if self.dim == 1:
            self.cells = zeros(nx, dtype=Cellule)
            for x in range(nx):
                self.cells[x] = Cellule(x * dx, dx=dx)

        elif self.dim == 2:
            self.cells = zeros((nx, ny), dtype=Cellule)
            for x in range(nx):
                for y in range(ny):
                    self.cells[x, y] = Cellule(x * dx, y * dy, dx=dx, dy=dy)

        else:
            self.cells = zeros((nx, ny, nz), dtype=Cellule)
            for x in range(nx):
                for y in range(ny):
                    for z in range(nz):
                        self.cells[x, y, z] = Cellule(x * dx, y * dy, z * dz, dx=dx, dy=dy, dz=dz)

    def appliquer_conditions_limites(self):
        D = self.dim

        if D == 1 :
            self.cells[0].velocity  = 0
            self.cells[-1].velocity = 0

        elif D==2 :
            self.cells[:,0].velocity  = array([0.0, 0.0])  # Bas
            self.cells[:,-1].velocity = array([0.0, 0.0])  # Haut
            self.cells[0,:].velocity  =  array([0.0, 0.0])  # Gauche
            self.cells[-1,:].velocity = array([0.0, 0.0])  # Droite

        elif D==3 :
            print("fini de coder abruti")

    def mat_F(self) :
        """
        Renvoie la matrice des flux
        """
        D = self.dim 
        if D == 1 :
            MF_m = zeros(self.nx)
            MF_v = zeros((self.nx,D))
            MF_e = zeros(self.nx)
            for i in range(self.nx):
                    f = self.cells[i].F()
                    MF_m[i] = f[0]
                    MF_v[i] = f[1]
                    MF_e[i] = f[2]

        elif D==2 :
            MF = zeros((self.nx,self.ny))
            for i in range(self.nx):
                for j in range(self.ny):
                    MF[i,j] = self.cells[i,j].F()
        return MF_m,MF_v,MF_e 

    def set_V(self,V:ndarray,x=None,y=None) :
        if self.dim == 1 :
            self.cells[x].velocity = V
        elif self.dim == 2 :
            if x is not None :
                for j in range(self.ny) :
                    self.cells[x,j].velocity = V
            if y is not None :
                for i in range(self.nx) :
                    self.cells[i,y].velocity = V

    def set_V_grid(self,V:ndarray) :
        assert list(V.shape) == list(self.shape) + [self.dim]
        it = nditer(self.cells,flags=['multi_index','refs_ok'],op_flags=['readwrite'])
        for cell in it :
            cell.item().velocity = V[it.multi_index]



    def get_v(self) :
        S = list(self.shape) + [self.dim]
        V = zeros(S)
        it = nditer(self.cells,flags=['multi_index','refs_ok'],op_flags=['readwrite'])
        for cell in it :
            V[it.multi_index] = cell.item().velocity
        return V 
    
    def set_pol(self,pol:ndarray,x=None,y=None) :
        if x==None and y==None :
            it = nditer(self.cells,flags=['multi_index','refs_ok'],op_flags=['readwrite'])
            for cell in it :
                cell.item().pol = pol[it.multi_index]
            
        elif self.dim == 1 and x is not None :
            self.cells[x].pol = pol
        elif self.dim == 2 :
            if x is not None and y is not None :
                self.cells[x,y].pol = pol
            elif x is not None :
                for j in range(self.ny) :
                    self.cells[x,j].pol = pol
            elif y is not None :
                for i in range(self.nx) :
                    self.cells[i,y].pol = pol
    
    def get_pol(self) :
        if self.dim == 1 :
            pol = zeros(self.nx)
            for i in range(self.nx) :
                pol[i] = self.cells[i].pol
        if self.dim == 2 :
            pol = zeros((self.nx,self.ny))
            for i in range(self.nx) :
                for j in range(self.ny) :
                    pol[i,j] = self.cells[i,j].pol
        return pol
    
    def get_tot_c(self):
        tot_c = 0
        for cell in self.cells :
            tot_c += cell.pol * cell.volume
        return tot_c

class solver:
    def __init__(self, grid : Grid, dt : float, t_end: float, mu : float,V:ndarray , ordre=1, CL=""):
        self.grid = grid
        self.dt = dt
        self.t_end = t_end
        self.mu = mu
        self.ordre = ordre
        self.CL = CL
        # pour l'instant, vitesse homogène mais variable dans le temps
        self.V = V

    def stay_positive(self,q, out):
        it = nditer([q, out], flags=['multi_index'], op_flags=[['readonly'], ['readwrite']])
        for q_val, out_val in it:
            if q_val < 1e-12 and out_val < 0:
                out[it.multi_index] = 0


    def dQdt_cell(self,q,t) :
        grad_c = grad( q,flag=self.CL)
        out = div(self.mu*grad_c-V_dot(self.V,q),flag=self.CL)
        self.stay_positive(q, out)
            # out[0] = 0  # concentration nulle aux bords
        # out[-1] = out[0] # périodicité
        
        return out

    
    def solve(self):
        """Simule l'advection-convection d'un poluant dans l'écoulement stationnaire"""
        nx = self.grid.nx
        shape_sol_t = zeros(self.grid.dim + 1,int)
        shape_sol_t[0] = int(self.t_end//self.dt)+3
        shape_sol_t[1:] = self.grid.shape
        sol_t = zeros(shape_sol_t)
        sol_t[0] = self.grid.get_pol()

        t = 0.0
        i_t = 0

        while t < self.t_end :
            t += self.dt
            i_t += 1
            sol_t[i_t] = step(sol_t[i_t-1],self.dQdt_cell,self.dt,t,self.ordre)
            self.grid.set_pol(sol_t[i_t])
        return sol_t



def anim_surface(matSol: ndarray, grid: Grid, T: ndarray = None, fps: int = 10, name="animation.gif"):
    """Crée une animation de la solution donnée en format GIF."""
    fig, ax = subplots()  # initialise la figure
    X, Y = grid.X, grid.Y
    Z = matSol[0]

    def animate(i):
        ax.clear()
        Z = matSol[i]
        surf = ax.imshow( Z, cmap='viridis',interpolation='nearest')
        if T is not None:
            ax.set_title(f"t = {T[i]:.2f}")
        return surf,

    # Create a tqdm progress bar
    with tqdm(total=len(matSol), desc="Creating animation") as pbar:
        def update(frame):
            result = animate(frame)
            pbar.update(1)
            return result

        ani = FuncAnimation(fig, update, frames=len(matSol), interval=1000 / fps, blit=True, repeat=False)
        ani.save(name, fps=fps)


if __name__=="__main__" :
    A = zeros(4)
    A[0] = 2
    A[1] = 2
    A[3] = 1
    A[2] = 3
    grad_A = grad(A)
    print("A = ",A)
    print("grad(A) = ",grad_A)
    print("A = ",A)
    print("div(A) = ",div(A))


    B = zeros((4,4))
    B[0] = A
    B[1] = 2*A
    B[3] = A
    grad_B = grad(B)
    print("B = ",B)
    print("grad(B) = ",grad_B)
    print("B = ",B)
    print("div(B) = ",div(B))

    assert all(div(B) == grad_B[0]+grad_B[1])

    def handle_out(q, out):
        it = nditer([q, out], flags=['multi_index'], op_flags=['readwrite'])
        for q_val, out_val in it:
            if q_val < 1e-12 and out_val < 0:
                out[it.multi_index] = 0

    # Example usage
    q = array([[1e-13, 2], [3, 4]])
    out = array([[-1, 2], [3, -4]])

    handle_out(q, out)
    print(out)
