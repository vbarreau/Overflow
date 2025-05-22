import numpy as np

def radial_CI(x,y,A=1)->float:
    """Fonction de condition initiale radiale"""
    r = np.sqrt((x-20)**2 + (y-10)**2)
    return A*np.exp(-(r-2)**2/10)
    
def lin_en_y(x,y,A=1):
    return A*y

def CI_qui_deforme(x,y,A=1) :
    return lin_en_y(x,y,A/5)  + radial_CI(x,y,A*10)

def CI_cylindre(x,y,A=1,a=2) :
    r = np.sqrt((x-20)**2 + (y-10)**2)
    teta = np.arctan2(y-10,x-20)
    e_r = np.array([np.cos(teta),np.sin(teta)])
    e_teta = np.array([-np.sin(teta),np.cos(teta)])
    V  = A*(np.cos(teta)*(1-(a/r)**2)*e_r - np.sin(teta)*(1+(a/r)**2)*e_teta)
    Vx = V[0]
    Vy = V[1]
    return Vx, Vy

def CI_uniforme(x,y,A=1) :
    """Condition initiale uniforme"""
    return A