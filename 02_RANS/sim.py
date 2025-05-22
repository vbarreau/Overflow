import sys
sys.path.append(r"D:/OneDrive/Documents/11-Codes/overflow/02_RANS")
from maillage import *
from fonctions import *
import time
from etat import *

def NS(cp : Parametres, t:float,prod_k, prod_w)->Parametres:
    """Renvoie les variations temporelles de la cellule"""
    if len(cp.condition) > 0 :
        return 0,0,0,0
    dVx = -cp.vx*cp.gradvx[0]- cp.vy*cp.gradvx[1] *(-cp.gradp[0] + 0 )/RHO  # Manque les gradients de sigma et tau
    dVy = -cp.vx*cp.gradvy[0]- cp.vy*cp.gradvy[1] *(-cp.gradp[1] + 0 )/RHO  # Manque les gradients de sigma et tau
    A =0
    for i in range(2):
        for j in range(2):
            if i == 0 :
                gv = cp.gradvx
            else :
                gv = cp.gradvy
            A += cp.tau[i,j]*gv[j] 
    dk = -cp.vx*cp.gradk[0]- cp.vy*cp.gradk[1] + A - BETA_STAR*cp.k*cp.w + prod_k.sum() 
    dw = -cp.vx*cp.gradw[0]- cp.vy*cp.gradw[1] + ALPHA * cp.w/cp.k * A - BETA*cp.w**2 + cp.sigma_d*(cp.gradk*cp.gradw).sum()  + prod_w.sum()
    
    return dVx, dVy, dk, dw


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
    
    def NS_sim(self,Q:Etat,t:float)->Etat:
        """Résolution des equations de Navier-Stokes sur l'etat Q
        Renvoie les vecteurs de variations temporelles de Vx, Vy, k, w et leurs gradients
        Les autres facteurs de variation sont unitaires"""
        dQ_dt = VarEtat(mesh=self.etat.mesh)
        prod_k = Q.compute_gradient("gamma_k")
        prod_w = Q.compute_gradient("gamma_w")
        residu = np.zeros(4)

        for i in range(len(self.etat.mesh.cells)):
            
            c = self.etat.cell_param[i]
            limits = c.condition
            var_liimits = [cl.var for cl in limits]

            dVx, dVy, dk, dw = NS(c,t,prod_k[i],prod_w[i])

            if 'vx' not in var_liimits :
                dQ_dt.cell_param[i].vx = dVx
            if 'vy' not in var_liimits :
                dQ_dt.cell_param[i].vy = dVy    
            dQ_dt.cell_param[i].k = dk
            dQ_dt.cell_param[i].w = dw 

        residu[0] = Q.sum("vx")/(Q.mean("vx")+1)
        residu[1] = Q.sum("vy")/(Q.mean("vy")+1)
        residu[2] = Q.sum("k") /(Q.mean("k")+1)
        residu[3] = Q.sum("w") /(Q.mean("w")+1)

        dQ_dt.compute_gradient("vx") # Les dérivées commutent donc je calcul d²/dx.dt plutot que d²/dt.dx
        dQ_dt.compute_gradient("vy")
        dQ_dt.compute_gradient("k")
        dQ_dt.compute_gradient("w")

        return dQ_dt, residu


    def step(self, F:callable, dt:float, t:float, ordre:int)-> Etat:
        """ Effectue une étape de la méthode de Runge-Kutta dans la simulation
            F       : fonction de Q et t qui renvoie le VarEtat dQ/dt = F(Q,t)[0]
            dt      : pas de temps désiré
            t       : temps actuel
            ordre   : 1,2,3 ou 4 
            """
        if ordre == 1:
            k1, residu =  F(self.etat, t)
            self.etat =  self.etat + k1*dt
        elif ordre == 2:
            k1 = F(self.etat, t)[0]
            k2, residu = F(self.etat + dt/2 * k1, t + dt/2)[0]
            self.etat = self.etat + dt * k2
        elif ordre == 3:
            k1 = F(self.etat, t)[0]
            k2 = F(self.etat + dt/2 * k1, t + dt/2)[0]
            k3, residu = F(self.etat - dt * k1 + 2 * dt * k2, t + dt)[0]
            self.etat = self.etat + dt/6 * (k1 + 4 * k2 + k3)
        elif ordre == 4:
            k1 = F(self.etat, t)[0]
            k2 = F(self.etat + dt/2 * k1, t + dt/2)[0]
            k3 = F(self.etat + dt/2 * k2, t + dt/2)[0]
            k4, residu = F(self.etat + dt * k3, t + dt)[0]
            self.etat = self.etat + dt/6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return residu
        
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
    sim.etat.set_CI("vx",CI_uniforme)
    sim.etat.set_CI("vy",CI_uniforme)
    sim.etat.compute_gradient()
    # for i in range(sim.etat.mesh.size):
    #     if sim.etat.cell_param[i].condition != None :
    #         cell_p = sim.etat.cell_param[i]
    #         cell = sim.etat.mesh.cells[i]
    #         var = cell_p.condition.var
    #         print(f'Cellule {i}, x = {cell.centroid[0]} : {var} = {getattr(sim.etat.cell_param[i],var)}')
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    sim.etat.plot('vx',ax=ax[0])
    for i in tqdm(range(6)):
        sim.step(sim.NS_sim, 0.5, 0, 1)
        sim.etat.update_all_param()

    sim.etat.plot('vx',ax=ax[1])

    plt.show()