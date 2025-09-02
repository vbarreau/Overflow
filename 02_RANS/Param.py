from geom import *
RHO = 1000
NU = 1e-3
CV = 4186
CP = CV # incompressible 
LAMBDA = 0.606
MU = NU * RHO

XI          = 0
f_beta      = 1
ALPHA       = 13 / 25
C_BETA1     = 0.075
C_ALPHA1    = 0.5976
BETA_0      = 0.0708
BETA        = BETA_0
BETA_STAR   = 0.09
SIGMA       = 0.5
SIGMA_STAR  = 0.6
SIGMA_DO    = 1 / 8
CLIM        = 7 / 8


class Parametres:
    """Classe pour les parametres de la cellule"""
    def __init__(self, T: float = 300, p: float = 1e5, vx: float = 0, vy: float = 0,k = 1, w = 1) -> None:
        self.T          = T
        self.p          = p #TODO : implémenter pression turbulente ? (=p+2/3*rho.k)
        self.vx         = vx
        self.vy         = vy
        self.gradT      = np.ones(2)
        self.gradgradtT = np.zeros(2) 
        self.gradp      = np.zeros(2)
        self.gradvx     = np.zeros(2)
        self.gradvy     = np.zeros(2)
        self.S          = np.zeros((2, 2))  # Tenseur des deformations
        self.Omega      = np.zeros((2, 2))  # Tenseur de vorticité
        self.tau        = np.zeros((2, 2))  # tenseur de turbulences
        
        self.k          = k          # Turbulent kinetic energy
        self.w          = w          # Specific dissipation rate
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
        self.v_sink      = np.zeros((2,2,2))   # tenseur de puit de vitesse

        self.condition = [] 

    def reset_CL(self):
        """Reset the condition of the cell"""
        for cl in self.condition: 
            setattr(self, cl.var, cl.value)
        else :
            return

    def div_v(self) -> float:
        """Calcul de la divergence de la vitesse"""
        return self.gradvx[0] + self.gradvy[1]
    
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
        # self.p = self.T * (287.05 * RHO)
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
        if self.w_bar == 0 :
            self.w_bar = 1e-10
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
    
    def solve_momentum(self) -> tuple:
        """Renvoie les variations temporelles de la cellule"""
        dVx = -self.vx*self.gradvx[0]- self.vy*self.gradvx[1] + (-self.gradp[0] + self.v_sink[0,0,0] + self.v_sink[1,0,1] )/RHO  
        dVy = -self.vx*self.gradvy[0]- self.vy*self.gradvy[1] + (-self.gradp[1] + self.v_sink[0,1,0] + self.v_sink[1,1,1] )/RHO 
        return dVx, dVy
    
    def solve_turbulence(self,prod_k,prod_w) -> tuple:
        """Renvoie les variations temporelles de k et w"""
        A =0 #  A = sum_{i,j}(tau[i,j] * dUi/dxj)
        for i in range(2):
            for j in range(2):
                if i == 0 :
                    gv = self.gradvx
                else :
                    gv = self.gradvy
                A += self.tau[i,j]*gv[j] # l'indice i de gv n'est pas nécessaire car selectionné avec les if précédents
        dk = -self.vx*self.gradk[0]- self.vy*self.gradk[1] + A - BETA_STAR*self.k*self.w + prod_k.sum() 
        dw = -self.vx*self.gradw[0]- self.vy*self.gradw[1] + ALPHA * self.w/self.k * A - BETA*self.w**2 + self.sigma_d*(self.gradk*self.gradw).sum()  + prod_w.sum()
        return dk, dw
    
    def solve_energy(self) -> float:
        dT = (-self.p*self.div_v() + LAMBDA*self.gradgradtT.sum() + 2*MU*(self.S*self.S).sum()-2/3*MU*self.div_v()**2)/RHO
        return dT
    
    def gradv(self) -> np.ndarray:
        """Renvoie le gradient de la vitesse"""
        gradV = np.zeros((2,2))
        gradV[0] = self.gradvx
        gradV[1] = self.gradvy
        return gradV
    
    def P_k(self) -> float:
        """Renvoie la production de k"""
        grad_v = self.gradv()
        return (2*(grad_v[0,0]**2 + grad_v[1,1,]**2) + (grad_v[0,1]+grad_v[1,0])**2)*self.Nu_t
        