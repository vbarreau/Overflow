from overflow import *
from objets import *
from matplotlib.animation import FuncAnimation
from tqdm import tqdm



nx = 20
dt = 1/(nx**2)/2
print(f"dt = {dt}")
t_end = 2
mu = 5
V = zeros(int(t_end//dt)+3)  
fps = 100

pol_ini = initial_conditions(nx)



fig,ax=subplots(3)
ax[0].plot(linspace(0,1,nx),pol_ini)
for ORDRE in [4] :
    test = Grid(nx,dx=1/nx)
    for i in range(nx) : 
        test.set_pol(array([pol_ini[i]]),x=i)
    tot_c_0 = test.get_tot_c()
    print(tot_c_0)
    res = solver(test,dt,t_end,mu,V,ORDRE).solve()
    tot_c_0 = test.get_tot_c()
    print(tot_c_0)

    ax[0].plot(linspace(0,1,nx),res[-5],label=f"RK{ORDRE}")
ax[0].legend()

ax[1].plot(linspace(0,t_end,int(t_end//dt)+3),res[:,7],label='7')
ax[1].plot(linspace(0,t_end,int(t_end//dt)+3),res[:,8],label='8')
ax[1].plot(linspace(0,t_end,int(t_end//dt)+3),res[:,9],label='9')
ax[1].legend()

ax[2].plot(linspace(0,t_end,int(t_end//dt)+3),gradient(res[:,7],dt),label='7')
ax[2].plot(linspace(0,t_end,int(t_end//dt)+3),gradient(res[:,8],dt),label='8')
ax[2].plot(linspace(0,t_end,int(t_end//dt)+3),gradient(res[:,9],dt),label='9')
ax[2].legend()
show()