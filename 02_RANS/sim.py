import sys
sys.path.append(r"D:/OneDrive/Documents/11-Codes/overflow/02_RANS")
from geom import *
from fonctions import *
import time
from etat import *


def NS(cp : Parametres, t:float,prod_k, prod_w)->Parametres:
    """Renvoie les variations temporelles de la cellule"""

    dVx, dVy = cp.solve_momentum()
    dk, dw = cp.solve_turbulence(prod_k, prod_w)
    dT = cp.solve_energy()
    
    return dVx, dVy, dk, dw, dT

def RHS(sim_ini,t:float = 0)->VarEtat: 
    """Renvoie les variations temporelles de l'état"""
    sim = sim_ini.copy()
    etat = sim.etat.copy()
    etat.update_all_param()
    etat.compute_gradient()
    etat.compute_tensors()

    # On calcule les coefficients pour chaque variable pour les equaitons
    # d(phi)/dt = -ac*phi_C - sum(af*phi_F) + bc
    sim.coeff_vx.compute_for_v("vx", etat)
    sim.coeff_vy.compute_for_v("vy", etat)
    sim.coeff_k.compute_for_k(etat)
    sim.coeff_w.compute_for_w(etat)
    sim.coeff_T.compute_for_T(etat)

    Vx = etat.get_var('vx')
    Vy = etat.get_var('vy')
    K =  etat.get_var('k')
    W =  etat.get_var('w')
    T =  etat.get_var('T')

    # On agrège les coefficients pour obtenir les variations temporel les
    rhs = VarEtat(sim_ini.etat)
    rhs.set_var('vx', -(sim.coeff_vx.aC + sim.coeff_vx.aF) @ Vx + sim.coeff_vx.bC)
    rhs.set_var('vy', -(sim.coeff_vy.aC + sim.coeff_vy.aF) @ Vy + sim.coeff_vy.bC)
    rhs.set_var('k' , -(sim.coeff_k.aC + sim.coeff_k.aF) @ K + sim.coeff_k.bC)
    rhs.set_var('w' , -(sim.coeff_w.aC + sim.coeff_w.aF) @ W + sim.coeff_w.bC)
    rhs.set_var('T' , -(sim.coeff_T.aC + sim.coeff_T.aF) @ T + sim.coeff_T.bC)

    return rhs

class Coefficients():
    """Classe qui stocke les coefficients de l'equation d'intégration de phi sous forme matricielle.
    Quelque soit phi et C, on a :
    d(phi)/dt = -ac*phi_C - sum(af*phi_F) + bc
     avec F les celllules voisines de C et f les faces de C
     Donc on a :
     VarEtat = -(Ac+Af)*PHI + B 
     ATTENTION : divisé par RHO par rapport au bouquin, chap 17.7 p.708
     """
    def __init__(self,size:int)->None:
        self.size = size
        self.aC = np.zeros((self.size,self.size)) # matrice diagonale [m²/s] en 2D
        self.aF = np.zeros((self.size,self.size)) # matrice de connectivité [m²/s]
        self.bC = np.zeros(self.size) # vecteur [m4.s-2] en 2D

    def increment_phi(self, phi:str, face_index:int, etat:Etat, gamma_phi:float)->None:
        """Incrémente les coefficients des cellules voisines de cette face.\n
        **phi** : attribut pour lequel on calcul les coefficients\n
        **face_index** : indice de la face considérée\n
        **etat** : l'état de la simulation\n
        **gamma_phi** : coefficient de diffusion pour phi\n
        **coeff_debit** : Coefficient multiplicateur du débit. 1 par défaut, Cp pour l'equation d'énergie.
        """

        face = etat.mesh.faces[face_index]
        Ef, Tf = face.Ef_Tf(etat.mesh.cells)
        c = face.owner
        f = face.neighbour
        gDiff = face.gDiff_f(etat.mesh.cells)
        coeff_debit = CP if phi == "T" else 1
        debit = etat.face_flow(face_index)*coeff_debit # le coeff sert pour l'equation d'energie, pour les autres c'est 1
        terme_C = gamma_phi * gDiff + max(0,debit)/RHO
        terme_B = (gamma_phi * np.dot(getattr(etat.face_param[face_index],"grad"+phi), Tf) - debit*(etat.compute_face_param_HR(face_index, phi) - etat.compute_face_param_U(face_index, phi)))/RHO
        terme_F = (- max(-debit,0) - gamma_phi * gDiff)/RHO
        if c is not None :
            self.aC[c,c] += terme_C # TODO : terme C n'a pas toujours la bonne dimension
            self.bC[c] += terme_B
        if f is not None :
            self.aC[f,f] -= terme_C
            self.aF[c,f] -= terme_B
        if f is not None and c is not None:
            self.aF[c,f] += terme_F
            self.aF[f,c] -= terme_F
            # TODO : Voir ce qui se passe en bordure, si c ou f sont None

    def compute_for_v(self,dir, etat:Etat)->None:
        """Calcul les coefficients pour l'integration du vecteur vitesse"""
        if dir in ["vx", "vy"]:
            phi = dir
        elif dir in ['x', 'y']:
            phi = 'v'+dir
        Nf = len(etat.mesh.faces)
        for i in range(Nf):
            gamma_v = (etat.face_param[i].Nu_t + NU) * RHO  # Je redivise par RHO apres mais c'est plus sur
            self.increment_phi(phi, i, etat, gamma_v)
        # Pas de terme source volumique pour v
    
    
    def compute_for_k(self, etat:Etat)->None:
        """Calcul les coefficients pour l'integration de l'énergie cinétique turbulente"""
        Nf = len(etat.mesh.faces)
        for i in range(Nf):
            nu_eff = NU + etat.face_param[i].Nu_t # TODO à corriger
            self.increment_phi("k", i, etat, nu_eff)
        for i in range(etat.mesh.size):
            self.aC[i,i] += BETA_STAR * etat.cell_param[i].w * etat.mesh.cells[i].volume 
            grad_v = etat.cell_param[i].gradv()
            P_k = (2*(grad_v[0,0]**2 + grad_v[1,1,]**2) + (grad_v[0,1]+grad_v[1,0])**2)*etat.face_param[i].Nu_t
            self.bC[i] += P_k * etat.mesh.cells[i].volume

    def compute_for_w(self, etat:Etat)->None:
        """Calcul les coefficients pour l'integration du taux de dissipation"""
        Nf = len(etat.mesh.faces)
        for i in range(Nf):
            nu_eff = NU + etat.face_param[i].Nu_t # TODO à corriger
            self.increment_phi("w", i, etat, nu_eff)
        for i in range(etat.mesh.size):
            self.aC[i,i] += C_BETA1 * etat.cell_param[i].w * etat.mesh.cells[i].volume 
            P_k = etat.cell_param[i].P_k() # Production de k
            self.bC[i] += C_ALPHA1 * etat.mesh.cells[i].volume * etat.cell_param[i].w /etat.cell_param[i].k * P_k

    def compute_for_T(self, etat:Etat)->None:
        """Calcul les coefficients pour l'integration de T\n
        **aT_F** = -k_f*E_f/d_CF - ||-debit_f,0||*Cp \n
        **aT_C** = -∑( aT_F) + debit_f*Cp \n
        **bT_C** = ∑(k_f*gradT_f.Tf) - Cp *∑(debit_f * (T^HR - T^U)_f )\n
        """
        Nf = len(etat.mesh.faces)
        for i in range(Nf):
            gamma_T = etat.face_param[i].k   
            self.increment_phi("T", i, etat, gamma_T)
        # Les termes sources sont négligés ou temporels donc nul en stationaire




class Sim():
    def __init__(self, **kwargs)->None:
        if len(kwargs) == 1 and 'filename' in kwargs and isinstance(kwargs['filename'], str):
            filename = kwargs['filename']
            self.etat = Etat(filename=filename)
        elif len(kwargs) == 1 and 'mesh' in kwargs and isinstance(kwargs['mesh'], Mesh):
            self.etat = Etat(mesh=kwargs['mesh'])

        self.etat.update_all_param()
        self.etat.compute_gradient()
        self.etat.compute_tensors()

        self.coeff_vx = Coefficients(self.etat.mesh.size)
        self.coeff_vy = Coefficients(self.etat.mesh.size)
        self.coeff_k  = Coefficients(self.etat.mesh.size)
        self.coeff_w  = Coefficients(self.etat.mesh.size)
        self.coeff_T  = Coefficients(self.etat.mesh.size)

        return None
    
    def copy(self):
        """Renvoie une copie de la simulation"""
        new_sim = Sim(mesh=self.etat.mesh.copy())
        new_sim.etat = self.etat.copy()
        new_sim.coeff_vx = self.coeff_vx
        new_sim.coeff_vy = self.coeff_vy
        new_sim.coeff_k  = self.coeff_k
        new_sim.coeff_w  = self.coeff_w
        new_sim.coeff_T  = self.coeff_T
        return new_sim
    
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

    def step(self, F:callable, dt:float, t:float, ordre:int)-> Etat:
        """ Effectue une étape de la méthode de Runge-Kutta dans la simulation
            F       : fonction de Q et t qui renvoie le VarEtat dQ/dt = F(Q,t)
            dt      : pas de temps désiré
            t       : temps actuel
            ordre   : 1,2,3 ou 4 
            """
        if ordre == 1:
            k1 =  F(self, t)
            self.etat =  self.etat + k1*dt
        elif ordre == 2:
            k1 = F(self, t)
            k2 = F(self + k1 * dt/2 , t + dt/2)
            self.etat = self.etat + k2 * dt
        elif ordre == 3:
            k1 = F(self, t)
            k2 = F(self + k1  * dt/2, t + dt/2)
            k3 = F(self - k1 * dt + k2 * 2*dt, t + dt)
            self.etat = self.etat +  (k1 +  k2*4 + k3)*dt/6
        elif ordre == 4:
            k1 = F(self, t)
            k2 = F(self + k1*dt/2 , t + dt/2)
            k3 = F(self + k2*dt/2, t + dt/2)
            k4 = F(self + k3*dt, t + dt)
            self.etat = self.etat + (k1 + k2*2 + k3*2 + k4)*dt/6
            # self.pressure_correction()
        
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
    
    from exemples import mesh_ligne
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    sim = Sim(mesh=mesh_ligne)
    sim.set_CL("vx",0,"in")
    sim.set_CL("vy",0,"in")
    sim.set_CL("p",2e5,"in")
    sim.set_CL("p",1e5,"out")

    sim.etat.update_all_param()
    # for i in range(sim.etat.mesh.size):
    #     if sim.etat.cell_param[i].condition != None :
    #         cell_p = sim.etat.cell_param[i]
    #         cell = sim.etat.mesh.cells[i]
    #         var = cell_p.condition.var
    #         print(f'Cellule {i}, x = {cell.centroid[0]} : {var} = {getattr(sim.etat.cell_param[i],var)}')
    
    print(f'Etat Initial : \nP : {sim.etat.cell_param[0].p:.2f} Pa, {sim.etat.cell_param[1].p} Pa, {sim.etat.cell_param[2].p} Pa \n')


    fig, ax = plt.subplots(2, 2, figsize=(12, 6))

    # Plot vx before
    sc_vx0 = sim.etat.plot('vx', ax=ax[0,0], point_size=500, cbar=False)
    # Plot p before
    sc_p0 = sim.etat.plot('p', ax=ax[1,0], point_size=500,cbar=False)

    for i in tqdm(range(100)):
        sim.step(RHS, 0.5, 0, 1)
        sim.etat.update_all_param()

    # Plot vx 
    sc_vx1 = sim.etat.plot('vx', ax=ax[0,1], point_size=500,cbar=False)
    # Plot p 
    sc_p1 = sim.etat.plot('p', ax=ax[1,1], point_size=500,cbar=False)

    # Place colorbars outside the plot area

    # vx colorbar (right of top row)
    divider0 = make_axes_locatable(ax[0,1])
    cax_vx = divider0.append_axes("right", size="5%", pad=0.15)
    cbar_vx = fig.colorbar(sc_vx1, cax=cax_vx)
    cbar_vx.set_label('vx (m/s)')

    # p colorbar (right of bottom row)
    divider1 = make_axes_locatable(ax[1,1])
    cax_p = divider1.append_axes("right", size="5%", pad=0.15)
    cbar_p = fig.colorbar(sc_p1, cax=cax_p)
    cbar_p.set_label('p')

    plt.tight_layout()
    print(f"""Etat final : \n
          P : {sim.etat.cell_param[0].p:.2f} Pa, {sim.etat.cell_param[1].p} Pa, {sim.etat.cell_param[2].p} Pa \n""")

    plt.show()