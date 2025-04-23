from overflow import *
from objets import *
from matplotlib.animation import FuncAnimation
from tqdm import tqdm


if __name__ == "__main__" :
    nx = 100
    dt = 1/(nx**2)/2
    t_end = 1
    mu = 50
    V = zeros(int(t_end/dt)+3) + 10
    fps = 50
    ORDRE = 4

    pol_ini = initial_conditions(nx)

    test = Grid(nx,dx=10/nx)
    for i in range(nx) : 
        test.set_pol(array([pol_ini[i]]),x=i)
    

    res = solver(test,dt,t_end,mu,V,4).solve()
    plot(linspace(0,1,test.nx),res[0])
    plot(linspace(0,1,test.nx),res[-5])
    show()

    # plot(linspace(0,1,test.nx),abs(res[-5]-res[0]))
    # show()
    # anim(res,test.nx,fps=fps,name=f"nx{nx}-mu{mu}-V{V}-order{ORDRE}.mp4")



    # ti = 0
    # tf = 10
    # q0 = 1
    # fig,ax = subplots(2)

    # Q1 = integrator(q0 , fonction_ex,0.01,ti=ti,tf=tf,ordre=1)
    # Q2 = integrator( q0, fonction_ex,0.01,ti=ti,tf=tf,ordre=2)
    # Q3 = integrator( q0, fonction_ex,0.01,ti=ti,tf=tf,ordre=3)
    # Q4 = integrator( q0, fonction_ex,0.01,ti=ti,tf=tf,ordre=4)
    # T = linspace(ti,tf,len(Q1))
    # Q_sol = array([exp(t) for t in T])


    # ax[0].plot(T, Q_sol , label="solution",color='k')
    # ax[0].plot(T,Q1,label="RK1",color='orange')
    # ax[0].plot(T,Q2,label="RK2",color='g')
    # ax[0].plot(T,Q3,label="RK3",color='blueviolet')
    # ax[0].plot(T,Q4,label="RK4",color='r')
    # ax[0].legend()


    # ax[1].plot(T,abs(Q_sol-Q1)/Q_sol,label="RK1",color='orange')
    # ax[1].plot(T,abs(Q_sol-Q2)/Q_sol,label="RK2",color='g')
    # ax[1].plot(T,abs(Q_sol-Q3)/Q_sol,label="RK3",color='blueviolet')
    # ax[1].plot(T,abs(Q_sol-Q4)/Q_sol,label="RK4",color='r')
    # ax[1].legend()

    # show()