import sys
sys.path.append(r"D:/OneDrive/Documents/11-Codes/overflow/02_RANS")
from maillage import *
from fonctions import *
import time
from etat import *

def NS(cp : Parametres, t:float,prod_k, prod_w)->Parametres:
    """Renvoie les variations temporelles de la cellule"""

    dVx, dVy = cp.solve_momentum()
    dk, dw = cp.solve_turbulence(prod_k, prod_w)
    dT = cp.solve_energy()
    
    return dVx, dVy, dk, dw, dT


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

            dVx, dVy, dk, dw, dT = NS(c,t,prod_k[i],prod_w[i])

            if 'vx' not in var_liimits :
                dQ_dt.cell_param[i].vx = dVx
            if 'vy' not in var_liimits :
                dQ_dt.cell_param[i].vy = dVy    
            dQ_dt.cell_param[i].k = dk
            dQ_dt.cell_param[i].w = dw 
            if 'T' not in var_liimits :
                dQ_dt.cell_param[i].T = dT

        residu[0] = Q.sum("vx")/(Q.mean("vx")+1)
        residu[1] = Q.sum("vy")/(Q.mean("vy")+1)
        residu[2] = Q.sum("k") /(Q.mean("k")+1)
        residu[3] = Q.sum("w") /(Q.mean("w")+1)

        dQ_dt.compute_gradient("vx") # Les dérivées commutent donc je calcul d²/dx.dt plutot que d²/dt.dx
        dQ_dt.compute_gradient("vy")
        dQ_dt.compute_gradient("k")
        dQ_dt.compute_gradient("w")

        return dQ_dt, residu
    
    def pressure_correction(self, Q:Etat, t:float)->Etat:
        


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
            k2, residu = F(self.etat + k1 * dt/2 , t + dt/2)[0]
            self.etat = self.etat + k2 * dt
        elif ordre == 3:
            k1 = F(self.etat, t)[0]
            k2 = F(self.etat + k1  * dt/2, t + dt/2)[0]
            k3, residu = F(self.etat - k1 * dt + k2 * 2*dt, t + dt)[0]
            self.etat = self.etat +  (k1 +  k2*4 + k3)*dt/6
        elif ordre == 4:
            k1 = F(self.etat, t)[0]
            k2 = F(self.etat + k1*dt/2 , t + dt/2)[0]
            k3 = F(self.etat + k2*dt/2, t + dt/2)[0]
            k4, residu = F(self.etat + k3*dt, t + dt)
            self.etat = self.etat + (k1 + k2*2 + k3*2 + k4)*dt/6
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
    
    from exemples import mesh_ligne
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    sim = Sim(mesh=mesh_ligne)
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
        sim.step(sim.NS_sim, 0.5, 0, 1)
        sim.etat.update_all_param()

    # Plot vx after
    sc_vx1 = sim.etat.plot('vx', ax=ax[0,1], point_size=500,cbar=False)
    # Plot p after
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
    print(f'Etat final : \nP : {sim.etat.cell_param[0].p:.2f} Pa, {sim.etat.cell_param[1].p} Pa, {sim.etat.cell_param[2].p} Pa \n')

    plt.show()